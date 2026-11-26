#!/usr/bin/env python3
"""Validate local model metadata and weight shard completeness before training.

The standard-library check reads JSON and safetensors headers, never tensor
payloads. It detects missing/empty/truncated shards and metadata mismatches;
it is not a checksum audit, a model-weight correctness audit, or a GPU test.
``--check-processor`` also loads the local Transformers processor without
network access or model-weight loading, and renders image/video chat prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

try:
    from .model_profiles import resolve_model
except ImportError:  # Direct invocation: python tools/preflight_model.py
    from model_profiles import resolve_model


REQUIRED_CONTEXT_LENGTH = 18432 + 768
SPECIAL_TOKENS = {
    "<|im_start|>": 151644, "<|im_end|>": 151645,
    "<|vision_start|>": 151652, "<|vision_end|>": 151653,
    "<|image_pad|>": 151655, "<|video_pad|>": 151656,
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read model metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model metadata must be a JSON object: {path}")
    return value


def _nonempty_file(path: Path) -> int:
    if not path.is_file():
        raise ValueError(f"Required model file is missing: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Required model file is empty: {path}")
    return size


def _safe_shard_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid model shard name: {name!r}")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or PureWindowsPath(name).drive or "\\" in name
            or ".." in relative.parts or relative.suffix != ".safetensors"):
        raise ValueError(f"Model index contains an unsafe shard path: {name!r}")
    # HF snapshots use local symlinks to sibling blobs; lexical traversal is rejected.
    return root.joinpath(*relative.parts)


def _weight_header(path: Path) -> tuple[int, set[str], int]:
    size = _nonempty_file(path)
    try:
        with path.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            if not 2 <= length <= min(100_000_000, size - 8):
                raise ValueError("invalid or truncated safetensors header")
            header = json.loads(stream.read(length))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read weight shard header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Weight shard header must be a JSON object: {path}")
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise ValueError(f"Weight shard contains no tensors: {path}")
    ranges = []
    payload_bytes = size - 8 - length
    dtype_bytes = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2,
                   "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    for name, info in tensors.items():
        if not name or not isinstance(info, dict):
            raise ValueError(f"Invalid tensor entry in {path}: {name!r}")
        offsets, shape, dtype = info.get("data_offsets"), info.get("shape"), info.get("dtype")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= payload_bytes):
            raise ValueError(f"Weight shard is truncated or has invalid tensor offsets: {path}: {name}")
        if (not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape)
                or dtype not in dtype_bytes):
            raise ValueError(f"Invalid or unsupported tensor shape/dtype in {path}: {name}")
        if math.prod(shape) * dtype_bytes[dtype] != offsets[1] - offsets[0]:
            raise ValueError(f"Tensor shape and byte range disagree in {path}: {name}")
        ranges.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError(f"Tensor byte ranges have gaps/overlap in {path}")
        cursor = end
    if cursor != payload_bytes:
        raise ValueError(f"Weight shard has unexpected payload bytes: {path}")
    return size, set(tensors), payload_bytes


def _check_preprocessor(metadata: dict[str, Any], profile: dict[str, Any], source: str) -> None:
    expected = {
        "processor_class": profile["processor_class"],
        "patch_size": profile["vision_dimensions"]["patch_size"],
        "temporal_patch_size": 2, "merge_size": 2,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"{source}.{name} must be {value!r}, found {metadata.get(name)!r}")


def _validate_processor(root: Path, profile: dict[str, Any]) -> dict[str, Any]:
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise ValueError(f"Cannot load local {profile['name']} processor from {root}: {exc}") from exc
    if type(processor).__name__ != profile["processor_class"]:
        raise ValueError(f"Expected {profile['processor_class']}, found {type(processor).__name__}")
    expected_patch = profile["vision_dimensions"]["patch_size"]
    for component in (processor.image_processor, getattr(processor, "video_processor", None)):
        if component is not None and (
            getattr(component, "patch_size", None), getattr(component, "merge_size", None)
        ) != (expected_patch, 2):
            raise ValueError(f"Processor must use patch_size={expected_patch} and merge_size=2")
    tokenizer = processor.tokenizer
    for token, expected in SPECIAL_TOKENS.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected:
            raise ValueError(f"Tokenizer/model mismatch: {token} must have ID {expected}, found {actual}")
    if tokenizer.eos_token_id != 151645:
        raise ValueError("Instruct tokenizer must use eos_token_id=151645")
    for modality in ("image", "video"):
        messages = [{"role": "user", "content": [
            {"type": modality, modality: "preflight-placeholder"},
            {"type": "text", "text": "Describe the action."},
        ]}]
        try:
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            raise ValueError(f"Multimodal {modality} chat template is unusable: {exc}") from exc
        if (not isinstance(prompt, str) or f"<|{modality}_pad|>" not in prompt
                or not prompt.endswith("<|im_start|>assistant\n")):
            raise ValueError(f"Chat template must preserve {modality} placeholder and Instruct assistant prefix")
    return {"processor_class": type(processor).__name__, "tokenizer_class": type(tokenizer).__name__}


def validate_model(
    model_path: str | Path, model_key: str, *,
    required_context_length: int = REQUIRED_CONTEXT_LENGTH, check_processor: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable report, or raise ValueError with a launch blocker."""
    profile = resolve_model(model_key)
    if type(required_context_length) is not int or required_context_length <= 0:
        raise ValueError("required_context_length must be a positive integer")
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"MODEL_PATH must be an existing local model directory: {root}")
    config = _read_json(root / "config.json")
    architectures = config.get("architectures") or config.get("text_config", {}).get("architectures")
    if (config.get("model_type") != profile["model_type"]
            or architectures != [profile["architecture"]]):
        raise ValueError(f"Expected {profile['architecture']} (model_type={profile['model_type']})")
    # Qwen2.5-VL upstream config is flat; recent Transformers saves text_config.
    text = config.get("text_config", config if profile["model_type"] == "qwen2_5_vl" else {})
    vision = config.get("vision_config", {})
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ValueError("text_config and vision_config must be JSON objects")
    for section in (config, text, vision):
        if section.get("quantization_config") is not None:
            raise ValueError(f"Quantized models are unsupported; use unquantized {profile['name']} weights")
        if section.get("num_experts") or section.get("num_local_experts"):
            raise ValueError("Mixture-of-experts models are unsupported; use the selected dense model")
    for section_name, section, expected in (
        ("text_config", text, profile["text_dimensions"]),
        ("vision_config", vision, profile["vision_dimensions"]),
    ):
        for name, value in expected.items():
            if type(section.get(name)) is not int or section[name] != value:
                raise ValueError(f"Expected {profile['name']} {section_name}.{name}={value}, found {section.get(name)!r}")
    context = text.get("max_position_embeddings")
    if type(context) is not int or context < required_context_length:
        raise ValueError(f"Model context {context!r} is below required teacher prompt + response {required_context_length}")
    for name, value in {
        "image_token_id": 151655, "video_token_id": 151656,
        "vision_start_token_id": 151652, "vision_end_token_id": 151653,
    }.items():
        if config.get(name) != value:
            raise ValueError(f"Model config.{name} must be {value}, found {config.get(name)!r}")
    if text.get("eos_token_id") != 151645:
        raise ValueError("Instruct text configuration must use eos_token_id=151645")
    _check_preprocessor(_read_json(root / "preprocessor_config.json"), profile, "preprocessor_config.json")
    if profile["model_type"] == "qwen3_vl":
        video_config = _read_json(root / "video_preprocessor_config.json")
        _check_preprocessor(video_config, profile, "video_preprocessor_config.json")
        if video_config.get("video_processor_type") != "Qwen3VLVideoProcessor":
            raise ValueError("video_preprocessor_config.json requires Qwen3VLVideoProcessor")
    tokenizer = _read_json(root / "tokenizer_config.json")
    if tokenizer.get("eos_token") not in ("<|im_end|>", {"content": "<|im_end|>"}):
        eos = tokenizer.get("eos_token")
        if not isinstance(eos, dict) or eos.get("content") != "<|im_end|>":
            raise ValueError("Instruct tokenizer_config.json requires eos_token=<|im_end|>")
    decoder = tokenizer.get("added_tokens_decoder", {})
    if not isinstance(decoder, dict):
        raise ValueError("Tokenizer added_tokens_decoder must be a JSON object")
    for token, token_id in SPECIAL_TOKENS.items():
        entry = decoder.get(str(token_id), {})
        if not isinstance(entry, dict) or entry.get("content") != token:
            raise ValueError(f"Tokenizer metadata must map token ID {token_id} to {token}")
    if (root / "tokenizer.json").is_file():
        _nonempty_file(root / "tokenizer.json")
    else:
        _nonempty_file(root / "vocab.json")
        _nonempty_file(root / "merges.txt")
    if not tokenizer.get("chat_template"):
        candidates = (root / "chat_template.jinja", root / "chat_template.json")
        if not any(path.is_file() and path.stat().st_size for path in candidates):
            processor_config = root / "processor_config.json"
            if not processor_config.is_file() or not _read_json(processor_config).get("chat_template"):
                raise ValueError("Model tokenizer/processor has no saved chat template")

    index_path = root / "model.safetensors.index.json"
    declared_tensor_bytes, mapping = None, None
    if index_path.is_file():
        index = _read_json(index_path)
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(key, str) or not key for key in mapping):
            raise ValueError("model.safetensors.index.json must contain a nonempty weight_map")
        paths = {_safe_shard_path(root, name) for name in mapping.values()}
        metadata = index.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Model index metadata must be a JSON object")
        declared_tensor_bytes = metadata.get("total_size")
        if declared_tensor_bytes is not None and (type(declared_tensor_bytes) is not int or declared_tensor_bytes <= 0):
            raise ValueError("Model index metadata.total_size must be a positive integer")
    else:
        paths = {root / "model.safetensors"}
    weight_sizes, payload_bytes = {}, 0
    for path in sorted(paths):
        size, names, payload = _weight_header(path)
        relative_name = path.relative_to(root).as_posix()
        if mapping is not None:
            expected_names = {name for name, shard in mapping.items() if shard == relative_name}
            if names != expected_names:
                raise ValueError(f"Weight index/header tensor names disagree for {relative_name}")
        weight_sizes[relative_name] = size
        payload_bytes += payload
    if declared_tensor_bytes is not None and payload_bytes != declared_tensor_bytes:
        raise ValueError("Weight shard tensor bytes disagree with index metadata.total_size")
    report = {
        "model_path": str(root), "model_key": profile["key"], "model_name": profile["name"],
        "model_type": profile["model_type"], "architecture": profile["architecture"],
        "parameter_scale": profile["parameter_scale"], "max_position_embeddings": context,
        "required_context_length": required_context_length, "weight_sizes": weight_sizes,
        "total_weight_file_bytes": sum(weight_sizes.values()), "total_tensor_bytes": payload_bytes,
        "metadata_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.json")) if path.is_file()
        },
        "processor_checked": bool(check_processor),
    }
    if check_processor:
        report.update(_validate_processor(root, profile))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--required-context-length", type=int, default=REQUIRED_CONTEXT_LENGTH)
    parser.add_argument("--check-processor", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
    args = parser.parse_args()
    try:
        report = validate_model(args.model_path, args.model_key,
                                required_context_length=args.required_context_length,
                                check_processor=args.check_processor)
    except ValueError as exc:
        parser.exit(1, f"Model preflight FAILED: {exc}\n")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Model preflight: OK ({report['model_name']}; {report['model_path']}; "
              f"{len(report['weight_sizes'])} shard(s); context={report['max_position_embeddings']}; "
              f"processor_checked={report['processor_checked']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
