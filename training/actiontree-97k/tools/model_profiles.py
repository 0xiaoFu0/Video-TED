"""Supported dense Instruct models; metadata verified against official configs.

Sources (2026-10-23):
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json
https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json
https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json

Qwen2.5-VL-7B added and verified on 2026-10-30:
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/config.json

This registry does not download models. Paths are resolved only when requested.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ROOT = "/path/to/workspace/models/Qwen"
MODELS: dict[str, dict[str, Any]] = {
    "qwen25-3b": {
        "name": "Qwen2.5-VL-3B-Instruct",
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_type": "qwen2_5_vl",
        "architecture": "Qwen2_5_VLForConditionalGeneration",
        "processor_class": "Qwen2_5_VLProcessor",
        "parameter_scale": "3B",
        "tensor_parallel_size": 1,
        "text_dimensions": {
            "hidden_size": 2048, "intermediate_size": 11008,
            "num_hidden_layers": 36, "num_attention_heads": 16,
            "num_key_value_heads": 2, "vocab_size": 151936,
        },
        "vision_dimensions": {
            "depth": 32, "hidden_size": 1280, "intermediate_size": 3420,
            "num_heads": 16, "out_hidden_size": 2048,
            "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
        },
    },
    "qwen25-7b": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_type": "qwen2_5_vl",
        "architecture": "Qwen2_5_VLForConditionalGeneration",
        "processor_class": "Qwen2_5_VLProcessor",
        "parameter_scale": "7B",
        "tensor_parallel_size": 1,
        "text_dimensions": {
            "hidden_size": 3584, "intermediate_size": 18944,
            "num_hidden_layers": 28, "num_attention_heads": 28,
            "num_key_value_heads": 4, "vocab_size": 152064,
        },
        "vision_dimensions": {
            "depth": 32, "hidden_size": 1280, "intermediate_size": 3420,
            "num_heads": 16, "out_hidden_size": 3584,
            "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
        },
    },
    "qwen3-4b": {
        "name": "Qwen3-VL-4B-Instruct",
        "hf_id": "Qwen/Qwen3-VL-4B-Instruct",
        "model_type": "qwen3_vl",
        "architecture": "Qwen3VLForConditionalGeneration",
        "processor_class": "Qwen3VLProcessor",
        "parameter_scale": "4B",
        "tensor_parallel_size": 1,
        "text_dimensions": {
            "hidden_size": 2560, "intermediate_size": 9728,
            "num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
        },
        "vision_dimensions": {
            "depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
            "num_heads": 16, "out_hidden_size": 2560,
            "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
        },
    },
    "qwen3-8b": {
        "name": "Qwen3-VL-8B-Instruct",
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_type": "qwen3_vl",
        "architecture": "Qwen3VLForConditionalGeneration",
        "processor_class": "Qwen3VLProcessor",
        "parameter_scale": "8B",
        "tensor_parallel_size": 1,
        "text_dimensions": {
            "hidden_size": 4096, "intermediate_size": 12288,
            "num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
        },
        "vision_dimensions": {
            "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
            "num_heads": 16, "out_hidden_size": 4096,
            "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
        },
    },
}


def resolve_model(key: str) -> dict[str, Any]:
    """Return an independent profile for a supported short key or official name."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Model key must be a nonempty string")
    normalized = key.strip().lower()
    aliases = {
        alias.lower(): canonical
        for canonical, profile in MODELS.items()
        for alias in (canonical, profile["name"], profile["hf_id"])
    }
    if normalized in {"qwen3-3b", "qwen3-vl-3b", "qwen3-vl-3b-instruct", "qwen/qwen3-vl-3b-instruct"}:
        raise ValueError(
            "Qwen3-VL-3B-Instruct is not a supported official profile. "
            "Use qwen3-4b (Qwen3-VL-4B-Instruct), or qwen25-3b for Qwen2.5-VL-3B-Instruct."
        )
    if normalized not in aliases:
        raise ValueError(f"Unknown model {key!r}; choose one of: {', '.join(MODELS)}")
    canonical = aliases[normalized]
    return {"key": canonical, **copy.deepcopy(MODELS[canonical])}


def resolve_model_path(
    key: str, model_path: str | Path | None = None, model_root: str | Path | None = None,
) -> Path:
    """Explicit path > MODEL_PATH; explicit root > MODEL_ROOT > server default."""
    profile = resolve_model(key)
    override = model_path if model_path is not None else os.environ.get("MODEL_PATH")
    if override:
        return Path(override).expanduser()
    root = model_root if model_root is not None else os.environ.get("MODEL_ROOT", DEFAULT_MODEL_ROOT)
    return Path(root).expanduser() / profile["name"]
