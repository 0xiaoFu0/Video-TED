#!/usr/bin/env python3
"""Export a completed AR FSDP checkpoint to an independent Hugging Face model.

Only local, trusted training checkpoints are accepted (torch.load uses pickle).
No network, GPU, optimizer loading, or source-checkpoint modification is needed.
Tensor dtypes are preserved. --dry-run checks paths/metadata without loading weights.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


ARCHITECTURES = {
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def discover(checkpoint, output=None):
    """Validate the completed step, complete rank set, and saved HF metadata."""
    supplied = Path(checkpoint).expanduser().resolve()
    actor = supplied if supplied.name == "actor" else supplied / "actor"
    step_dir = actor.parent
    match = re.fullmatch(r"global_step_(\d+)", step_dir.name)
    if match is None or not actor.is_dir():
        raise ValueError("CHECKPOINT must be an existing global_step_N or global_step_N/actor directory")
    step = int(match[1])
    tracker_path = step_dir.parent / "checkpoint_tracker.json"
    if not tracker_path.is_file():
        raise ValueError("Missing checkpoint_tracker.json; keep the original run directory to verify save completion")
    tracker = read_json(tracker_path)
    last = tracker.get("last_global_step")
    if type(last) is not int or last < step:
        raise ValueError("Checkpoint save is not confirmed complete by checkpoint_tracker.json")
    if not (step_dir / "dataloader.pt").is_file() or not (step_dir / "dataloader.pt").stat().st_size:
        raise ValueError("Missing/empty dataloader.pt; checkpoint save is incomplete")
    actual = set(actor.glob("model_world_size_*_rank_*.pt"))
    rank0 = [path for path in actual if re.fullmatch(r"model_world_size_[1-9]\d*_rank_0\.pt", path.name)]
    if len(rank0) != 1:
        raise ValueError("Expected exactly one rank 0 model shard")
    world = int(re.fullmatch(r"model_world_size_(\d+)_rank_0\.pt", rank0[0].name)[1])
    paths = [actor / f"model_world_size_{world}_rank_{rank}.pt" for rank in range(world)]
    if actual != set(paths) or any(not path.is_file() or not path.stat().st_size for path in paths):
        raise ValueError("Model shard set is incomplete, empty, or mixes different world sizes")
    metadata = actor / "huggingface"
    config = read_json(metadata / "config.json")
    family = config.get("model_type")
    # Transformers 4.57 Qwen2.5 config saving may move architectures into text_config.
    architecture = config.get("architectures") or config.get("text_config", {}).get("architectures")
    if family not in ARCHITECTURES or (architecture and architecture != [ARCHITECTURES[family]]):
        raise ValueError("Only dense Qwen2.5-VL / Qwen3-VL conditional-generation checkpoints are supported")
    for section in (config, config.get("text_config", {}), config.get("vision_config", {})):
        if not isinstance(section, dict) or section.get("quantization_config") is not None or section.get("num_experts") or section.get("num_local_experts"):
            raise ValueError("Quantized or MoE checkpoints are unsupported")
    for name in ("tokenizer_config.json", "preprocessor_config.json"):
        if not (metadata / name).is_file() or not (metadata / name).stat().st_size:
            raise ValueError(f"Missing saved processor/tokenizer metadata: {name}")
    if not (metadata / "tokenizer.json").is_file() and not all((metadata / name).is_file() for name in ("vocab.json", "merges.txt")):
        raise ValueError("Missing saved tokenizer vocabulary")
    tokenizer = read_json(metadata / "tokenizer_config.json")
    processor = read_json(metadata / "processor_config.json") if (metadata / "processor_config.json").is_file() else {}
    if not (tokenizer.get("chat_template") or processor.get("chat_template") or any((metadata / name).is_file() for name in ("chat_template.jinja", "chat_template.json"))):
        raise ValueError("Missing saved chat template")
    requested = Path(output).expanduser() if output else step_dir.parent / "exports" / (step_dir.name + "-hf")
    if os.path.lexists(requested):
        raise ValueError(f"Output already exists; choose a new directory: {requested}")
    destination = requested.resolve()
    if os.path.lexists(destination):
        raise ValueError(f"Output already exists; choose a new directory: {destination}")
    if destination == step_dir or step_dir in destination.parents:
        raise ValueError("Export outside global_step_N so checkpoint retention cannot delete the export")
    return actor, paths, metadata, destination, family


def mesh_metadata(mesh, world):
    """Handle the DeviceMesh serialization used by both older and newer Torch."""
    legacy = getattr(mesh, "__dict__", {})
    try:
        names = mesh.mesh_dim_names
    except AttributeError:
        names = legacy.get("mesh_dim_names", legacy.get("_mesh_dim_names"))
    names = tuple(names or ())
    try:
        ranks = mesh.mesh
    except AttributeError:
        ranks = legacy.get("mesh", legacy.get("_mesh"))
    if ranks is None:
        raise ValueError("Missing DeviceMesh rank ordering; use the same PyTorch version as training")
    if names not in (("fsdp",), ("ddp", "fsdp")):
        raise ValueError(f"Unsupported DeviceMesh: {names}")
    shape = tuple(ranks.shape)
    flat = [int(rank) for rank in ranks.reshape(-1).tolist()]
    if len(shape) != len(names) or sorted(flat) != list(range(world)):
        raise ValueError("DeviceMesh must describe exactly the saved ranks")
    return shape, flat


def combine(parts, placements, mesh_shape, mesh_ranks, global_shape):
    """Undo mesh dimensions from inner to outer; preserve replicated buffers."""
    import torch
    ordered = [parts[rank] for rank in mesh_ranks]
    for size, (kind, dim) in reversed(list(zip(mesh_shape, placements))):
        merged = []
        for start in range(0, len(ordered), size):
            group = ordered[start:start + size]
            if kind == "replicate":
                if any(value.dtype != group[0].dtype or not torch.equal(value, group[0]) for value in group[1:]):
                    raise ValueError("Replicated tensor values differ between ranks")
                value = group[0]
            elif kind == "shard":
                value = torch.cat(group, dim=dim)
            else:
                raise ValueError("Partial/unknown DTensor placement is unsupported")
            merged.append(value)
        ordered = merged
    if len(ordered) != 1 or tuple(ordered[0].shape) != tuple(global_shape):
        raise ValueError("Merged tensor shape does not match global DTensor metadata")
    return ordered[0].contiguous()


def merge_checkpoint(paths):
    import torch
    from torch.distributed.tensor import DTensor
    pieces, metadata, keys = {}, {}, None
    for rank, path in enumerate(paths):
        print(f"Loading model rank {rank + 1}/{len(paths)}: {path.name}", flush=True)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or not state or not all(isinstance(key, str) for key in state):
            raise ValueError(f"Invalid state dictionary: {path}")
        if keys is None:
            keys = set(state)
        elif set(state) != keys:
            raise ValueError("Model keys differ between ranks")
        for key, tensor in state.items():
            if isinstance(tensor, DTensor):
                mesh_shape, mesh_ranks = mesh_metadata(tensor.device_mesh, len(paths))
                placements = []
                for placement in tensor.placements:
                    if placement.is_replicate():
                        placements.append(("replicate", None))
                    elif placement.is_shard():
                        placements.append(("shard", placement.dim))
                    else:
                        raise ValueError(f"Unsupported placement for {key}: {placement}")
                if len(placements) != len(mesh_shape):
                    raise ValueError("Placement/DeviceMesh dimensions differ")
                local = tensor.to_local().detach().cpu()
                info = (tuple(placements), mesh_shape, tuple(mesh_ranks), tuple(tensor.shape), local.dtype)
            elif isinstance(tensor, torch.Tensor):
                local = tensor.detach().cpu()
                # Plain tensors may only be identical full replicas, never guessed shards.
                info = ((("replicate", None),), (len(paths),), tuple(range(len(paths))), tuple(local.shape), local.dtype)
            else:
                raise ValueError(f"Unsupported checkpoint value for {key}: {type(tensor).__name__}")
            if rank == 0:
                metadata[key], pieces[key] = info, []
            elif metadata[key] != info:
                raise ValueError(f"Sharding metadata differs between ranks: {key}")
            pieces[key].append(local)
        del state
    merged = {}
    for key in sorted(pieces):
        placements, shape, ranks, global_shape, _ = metadata[key]
        merged[key] = combine(pieces.pop(key), placements, shape, ranks, global_shape)
    return merged


def canonicalize_keys(state, expected, mapping):
    """Apply Transformers' legacy-to-current mappings only for missing keys."""
    canonical = {}
    for key, value in state.items():
        target = key
        if target not in expected:
            for pattern, replacement in mapping.items():
                target = re.sub(pattern, replacement, target)
        if target not in expected:
            raise ValueError(f"Unexpected checkpoint key after conversion: {key} -> {target}")
        if target in canonical:
            raise ValueError(f"Checkpoint key conversion collision: {target}")
        canonical[target] = value
    missing = set(expected) - set(canonical)
    if missing:
        raise ValueError(f"Missing checkpoint keys: {sorted(missing)[:10]}")
    for key in expected:
        if expected[key].shape != canonical[key].shape:
            raise ValueError(f"Checkpoint/model tensor shape differs: {key}")
    return canonical


def source_snapshot(paths):
    return [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]


def export(checkpoint, output=None, dry_run=False):
    actor, paths, metadata, destination, family = discover(checkpoint, output)
    summary = {"checkpoint": str(actor), "output": str(destination), "model_type": family,
               "shards": len(paths), "dtype": "preserved", "dry_run": dry_run}
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return destination
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    metadata_files = sorted(path for path in metadata.rglob("*") if path.is_file()
                            and path.suffix not in {".safetensors", ".bin", ".pt"}
                            and not path.name.endswith(".index.json"))
    monitored = paths + metadata_files
    before = source_snapshot(monitored)
    state = merge_checkpoint(paths)
    config = AutoConfig.from_pretrained(metadata, local_files_only=True, trust_remote_code=False)
    dtype = next(value.dtype for value in state.values() if value.is_floating_point())
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config, torch_dtype=dtype, trust_remote_code=False)
    state = canonicalize_keys(state, model.state_dict(), getattr(model, "_checkpoint_conversion_mapping", {}))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two exports to the same output from publishing over each other.
    lock = destination.with_name(destination.name + ".export.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if os.path.lexists(destination):
            raise ValueError(f"Output already exists: {destination}")
        with tempfile.TemporaryDirectory(prefix=destination.name + ".exporting-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "model"
            stage.mkdir()
            for path in metadata_files:
                target = stage / path.relative_to(metadata)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            generation = stage / "generation_config.json"
            saved_generation = generation.read_bytes() if generation.exists() else None
            # The model stays on meta: save the real merged state without allocating a second model.
            model.save_pretrained(stage, state_dict=state, safe_serialization=True, max_shard_size="5GB")
            if saved_generation is not None:
                generation.write_bytes(saved_generation)
            if before != source_snapshot(monitored):
                raise ValueError("Checkpoint changed during export; select a completed, stable checkpoint")
            summary.update(source_files=before, tensors=len(state), status="complete")
            (stage / "EXPORT_COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            if os.path.lexists(destination):
                raise ValueError(f"Output appeared during export: {destination}")
            stage.rename(destination)
    finally:
        lock.unlink()
    print(f"Exported Hugging Face model: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Completed global_step_N or global_step_N/actor")
    parser.add_argument("--output", help="New directory; default: RUN/exports/global_step_N-hf")
    parser.add_argument("--dry-run", action="store_true", help="Check metadata and shard completeness without loading tensors")
    args = parser.parse_args()
    try:
        export(args.checkpoint, args.output, args.dry_run)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
