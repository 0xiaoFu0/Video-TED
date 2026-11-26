#!/usr/bin/env python3
"""Export AR FSDP model shards to an independent Hugging Face directory on CPU."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path


def discover_shards(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    candidates = list(checkpoint.glob("model_world_size_*_rank_0.pt"))
    if len(candidates) != 1:
        raise ValueError("actor 目录应恰好有一份 rank 0 模型文件")
    match = re.fullmatch(r"model_world_size_(\d+)_rank_0.pt", candidates[0].name)
    if match is None:
        raise ValueError("Invalid rank 0 filename")
    world = int(match[1])
    paths = [checkpoint / f"model_world_size_{world}_rank_{i}.pt" for i in range(world)]
    if world < 1 or any(not p.is_file() or p.stat().st_size == 0 for p in paths):
        raise ValueError("模型分片不完整；请选择已保存完成的 checkpoint")
    if not (checkpoint / "huggingface/config.json").is_file():
        raise ValueError("Missing huggingface/config.json in checkpoint")
    return paths


def merge_checkpoint(paths):
    import torch
    from torch.distributed.tensor import DTensor
    pieces, metadata = {}, {}
    expected_keys = None
    for rank, path in enumerate(paths):
        print(f"Loading model shard {rank + 1}/{len(paths)}: {path.name}", flush=True)
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(shard, dict) or not shard:
            raise ValueError("Invalid model shard: " + str(path))
        if expected_keys is None:
            expected_keys = set(shard)
        elif set(shard) != expected_keys:
            raise ValueError("Model keys differ between ranks")
        for key, tensor in shard.items():
            if isinstance(tensor, DTensor):
                names = tensor.device_mesh.mesh_dim_names
                placements = tuple(tensor.placements)
                if names == ("ddp", "fsdp") and tensor.device_mesh.mesh.shape[0] == 1:
                    placements = placements[1:]
                elif names != ("fsdp",):
                    raise ValueError("Unsupported FSDP mesh: " + str(names))
                if len(placements) != 1:
                    raise ValueError("Expected a one-dimensional FSDP placement")
                placement = placements[0]
                if placement.is_replicate():
                    kind, dim = "replicate", None
                elif placement.is_shard():
                    kind, dim = "shard", placement.dim
                else:
                    raise ValueError("Unsupported tensor placement: " + str(placement))
                local = tensor.to_local().detach().cpu()
                info = (kind, dim, tuple(tensor.shape))
            elif isinstance(tensor, torch.Tensor) and len(paths) == 1:
                local = tensor.detach().cpu()
                info = ("replicate", None, tuple(tensor.shape))
            else:
                raise ValueError("Multi-rank export requires DTensor metadata: " + key)
            if rank == 0:
                metadata[key] = info
                pieces[key] = []
            elif metadata[key] != info:
                raise ValueError("Sharding metadata differs between ranks: " + key)
            if info[0] == "shard" or rank == 0:
                pieces[key].append(local.to(torch.bfloat16) if local.is_floating_point() else local)
        del shard
    merged = {}
    for key, parts in pieces.items():
        kind, dim, shape = metadata[key]
        value = torch.cat(parts, dim=dim) if kind == "shard" else parts[0]
        if tuple(value.shape) != shape:
            raise ValueError(f"Merged tensor shape differs: {key}, {tuple(value.shape)} != {shape}")
        merged[key] = value
    return merged


def export(checkpoint, destination):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    checkpoint, destination = Path(checkpoint).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("导出目标已存在，请选择新目录：" + str(destination))
    if checkpoint == destination or checkpoint in destination.parents:
        raise ValueError("请导出到训练 checkpoint 之外，避免随训练的保留策略被删除")
    paths = discover_shards(checkpoint)
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    config = AutoConfig.from_pretrained(checkpoint / "huggingface", local_files_only=True)
    architecture = (getattr(config, "architectures", None) or [""])[0]
    if "ForConditionalGeneration" in architecture:
        auto_model = AutoModelForImageTextToText
    elif "ForCausalLM" in architecture or architecture == "GPT2LMHeadModel":
        auto_model = AutoModelForCausalLM
    else:
        raise ValueError("Unsupported architecture: " + architecture)
    state = merge_checkpoint(paths)
    if before != [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]:
        raise ValueError("读取期间 checkpoint 发生变化，请选择已保存完成的版本")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + ".exporting-", dir=destination.parent) as work:
        stage = Path(work) / "model"
        stage.mkdir()
        for path in (checkpoint / "huggingface").iterdir():
            if path.is_file() and path.suffix not in {".safetensors", ".bin", ".pt"}:
                shutil.copy2(path, stage / path.name)
        generation = stage / "generation_config.json"
        saved_generation = generation.read_bytes() if generation.exists() else None
        with torch.device("meta"):
            model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
        expected = model.state_dict()
        if set(expected) != set(state):
            missing, extra = set(expected) - set(state), set(state) - set(expected)
            raise ValueError(f"Checkpoint/model keys differ; missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}")
        for key in expected:
            if expected[key].shape != state[key].shape:
                raise ValueError("Checkpoint/model tensor shape differs: " + key)
        del expected
        model.to_empty(device="cpu")
        model.save_pretrained(stage, state_dict=state, safe_serialization=True)
        if saved_generation is not None:
            generation.write_bytes(saved_generation)
        (stage / "EXPORT_COMPLETE.json").write_text(json.dumps({
            "checkpoint": str(checkpoint), "shards": len(paths), "dtype": "bfloat16",
            "source_files": [{"name": p.name, "size": s[0], "mtime_ns": s[1]} for p, s in zip(paths, before)]
        }, indent=2) + "\n", encoding="utf-8")
        stage.rename(destination)
    print("Exported Hugging Face model:", destination, flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="global_step_N/actor")
    parser.add_argument("destination", help="New independent Hugging Face directory")
    args = parser.parse_args()
    export(args.checkpoint, args.destination)
