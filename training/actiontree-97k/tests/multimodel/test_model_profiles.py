"""Offline tests for model identity and prelaunch failure boundaries.

Fixtures contain tiny synthetic tensors, not pretrained model weights.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.model_profiles import MODELS, resolve_model, resolve_model_path
from tools.preflight_model import REQUIRED_CONTEXT_LENGTH, SPECIAL_TOKENS, validate_model


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_weights(path, names=("model.test.weight",)):
    header = {
        name: {"dtype": "BF16", "shape": [2], "data_offsets": [idx * 4, (idx + 1) * 4]}
        for idx, name in enumerate(names)
    }
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * (4 * len(names)))


def create_model(tmp_path, key="qwen3-4b", *, nested_qwen25=False, sharded=False):
    root = tmp_path / key
    root.mkdir()
    profile = resolve_model(key)
    text = {**profile["text_dimensions"], "max_position_embeddings": 262144, "eos_token_id": 151645}
    config = {
        "model_type": profile["model_type"], "architectures": [profile["architecture"]],
        "vision_config": profile["vision_dimensions"], "image_token_id": 151655,
        "video_token_id": 151656, "vision_start_token_id": 151652, "vision_end_token_id": 151653,
    }
    if profile["model_type"] == "qwen2_5_vl" and not nested_qwen25:
        config.update(text)
    else:
        config["text_config"] = text
    write_json(root / "config.json", config)
    processor = {
        "processor_class": profile["processor_class"],
        "patch_size": profile["vision_dimensions"]["patch_size"], "merge_size": 2, "temporal_patch_size": 2,
    }
    write_json(root / "preprocessor_config.json", processor)
    if profile["model_type"] == "qwen3_vl":
        write_json(root / "video_preprocessor_config.json", {**processor, "video_processor_type": "Qwen3VLVideoProcessor"})
    write_json(root / "tokenizer_config.json", {
        "eos_token": "<|im_end|>", "chat_template": "synthetic template; never loaded",
        "added_tokens_decoder": {str(token_id): {"content": token} for token, token_id in SPECIAL_TOKENS.items()},
    })
    write_json(root / "tokenizer.json", {"test_fixture": True})
    if sharded:
        mapping = {}
        for i in range(2):
            shard = f"model-0000{i + 1}-of-00002.safetensors"
            name = f"model.test{i}.weight"
            write_weights(root / shard, (name,))
            mapping[name] = shard
        write_json(root / "model.safetensors.index.json", {"metadata": {"total_size": 8}, "weight_map": mapping})
    else:
        write_weights(root / "model.safetensors")
    return root


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    write_json(path, data)


@pytest.mark.parametrize("key", list(MODELS))
def test_accepts_each_official_model_profile(tmp_path, key):
    report = validate_model(create_model(tmp_path, key), key)
    assert report["model_name"] == MODELS[key]["name"]
    assert report["required_context_length"] == REQUIRED_CONTEXT_LENGTH
    assert report["total_tensor_bytes"] == 4
    assert report["processor_checked"] is False


@pytest.mark.parametrize("key", ["qwen25-3b", "qwen25-7b"])
def test_accepts_qwen25_serialized_nested_text_config(tmp_path, key):
    root = create_model(tmp_path, key, nested_qwen25=True)
    def move_architecture(config):
        config["text_config"]["architectures"] = config.pop("architectures")
    edit_json(root / "config.json", move_architecture)
    assert validate_model(root, key)["model_type"] == "qwen2_5_vl"


def test_qwen25_7b_dimensions_match_official_config():
    # Independent values from Qwen/Qwen2.5-VL-7B-Instruct/config.json.
    # In particular, its vocabulary and layer count differ from the 3B model.
    profile = resolve_model("qwen25-7b")
    assert profile["text_dimensions"] == {
        "hidden_size": 3584, "intermediate_size": 18944,
        "num_hidden_layers": 28, "num_attention_heads": 28,
        "num_key_value_heads": 4, "vocab_size": 152064,
    }
    assert profile["vision_dimensions"] == {
        "depth": 32, "hidden_size": 1280, "intermediate_size": 3420,
        "num_heads": 16, "out_hidden_size": 3584,
        "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
    }
    assert profile["model_type"] == "qwen2_5_vl"
    assert profile["architecture"] == "Qwen2_5_VLForConditionalGeneration"
    assert profile["processor_class"] == "Qwen2_5_VLProcessor"
    assert profile["parameter_scale"] == "7B"
    assert profile["tensor_parallel_size"] == 1


@pytest.mark.parametrize("alias", [
    "qwen25-7b", "Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct",
    "  QWEN25-7B  ",
])
def test_qwen25_7b_aliases_resolve_to_same_default_path(alias, monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    profile = resolve_model(alias)
    assert profile["key"] == "qwen25-7b"
    assert profile["hf_id"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert resolve_model_path(alias) == Path("/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("actual_key,requested_key,expected_hidden", [
    ("qwen25-3b", "qwen25-7b", 3584),
    ("qwen25-7b", "qwen25-3b", 2048),
])
def test_qwen25_path_override_cannot_substitute_another_size(
    tmp_path, nested, actual_key, requested_key, expected_hidden,
):
    root = create_model(tmp_path, actual_key, nested_qwen25=nested)
    with pytest.raises(ValueError, match=f"hidden_size={expected_hidden}"):
        validate_model(root, requested_key)


def test_resolves_paths_and_copies_profiles(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    assert resolve_model_path("qwen3-4b") == Path("/path/to/workspace/models/Qwen/Qwen3-VL-4B-Instruct")
    monkeypatch.setenv("MODEL_ROOT", "/models")
    assert resolve_model_path("qwen3-8b") == Path("/models/Qwen3-VL-8B-Instruct")
    monkeypatch.setenv("MODEL_PATH", "/exact")
    assert resolve_model_path("qwen3-4b") == Path("/exact")
    assert resolve_model_path("qwen3-4b", "/explicit") == Path("/explicit")
    copied = resolve_model("Qwen/Qwen3-VL-4B-Instruct")
    copied["text_dimensions"]["hidden_size"] = 1
    assert MODELS["qwen3-4b"]["text_dimensions"]["hidden_size"] == 2560


def test_qwen3_3b_never_silently_selects_another_size():
    with pytest.raises(ValueError, match="qwen3-4b"):
        resolve_model("qwen3-3b")
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model("other")


def test_model_path_override_does_not_bypass_architecture_or_size(tmp_path):
    root = create_model(tmp_path, "qwen3-8b")
    with pytest.raises(ValueError, match="hidden_size=2560"):
        validate_model(root, "qwen3-4b")
    with pytest.raises(ValueError, match="Qwen2_5_VLForConditionalGeneration"):
        validate_model(root, "qwen25-3b")


@pytest.mark.parametrize("target,field,value,match", [
    ("config", "quantization_config", {}, "Quantized"),
    ("text", "quantization_config", {"quant_method": "awq"}, "Quantized"),
    ("vision", "num_experts", 64, "Mixture-of-experts"),
    ("vision", "patch_size", 14, "vision_config.patch_size"),
    ("text", "max_position_embeddings", 8192, "below required"),
    ("config", "video_token_id", 1, "video_token_id"),
    ("text", "eos_token_id", 151643, "eos_token_id"),
])
def test_rejects_incompatible_configs(tmp_path, target, field, value, match):
    root = create_model(tmp_path)
    def mutate(data):
        section = {"config": data, "text": data["text_config"], "vision": data["vision_config"]}[target]
        section[field] = value
    edit_json(root / "config.json", mutate)
    with pytest.raises(ValueError, match=match):
        validate_model(root, "qwen3-4b")


@pytest.mark.parametrize("context", [0, -1, True, 1.5])
def test_context_requires_positive_integer(tmp_path, context):
    with pytest.raises(ValueError, match="positive integer"):
        validate_model(tmp_path, "qwen3-4b", required_context_length=context)


@pytest.mark.parametrize("filename", [
    "preprocessor_config.json", "video_preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
])
def test_missing_processor_assets_fail_before_gpu_launch(tmp_path, filename):
    root = create_model(tmp_path)
    (root / filename).unlink()
    with pytest.raises(ValueError, match="missing|Cannot read"):
        validate_model(root, "qwen3-4b")


def test_rejects_wrong_tokenizer_special_ids(tmp_path):
    root = create_model(tmp_path)
    edit_json(root / "tokenizer_config.json", lambda data: data["added_tokens_decoder"].pop("151656"))
    with pytest.raises(ValueError, match="151656"):
        validate_model(root, "qwen3-4b")


def test_template_can_be_standalone_but_cannot_be_missing(tmp_path):
    root = create_model(tmp_path)
    edit_json(root / "tokenizer_config.json", lambda data: data.pop("chat_template"))
    with pytest.raises(ValueError, match="no saved chat template"):
        validate_model(root, "qwen3-4b")
    (root / "chat_template.jinja").write_text("synthetic standalone template")
    assert validate_model(root, "qwen3-4b")["processor_checked"] is False


def test_accepts_complete_shards_and_rejects_missing_shard(tmp_path):
    root = create_model(tmp_path, sharded=True)
    assert len(validate_model(root, "qwen3-4b")["weight_sizes"]) == 2
    (root / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_model(root, "qwen3-4b")


@pytest.mark.parametrize("replacement,match", [(b"", "empty"), (b"version https://git-lfs", "header")])
def test_empty_or_lfs_pointer_shard_is_not_a_model(tmp_path, replacement, match):
    root = create_model(tmp_path)
    (root / "model.safetensors").write_bytes(replacement)
    with pytest.raises(ValueError, match=match):
        validate_model(root, "qwen3-4b")


def test_truncated_payload_is_detected_without_reading_tensor_payload(tmp_path):
    root = create_model(tmp_path)
    path = root / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        validate_model(root, "qwen3-4b")


@pytest.mark.parametrize("name", ["../outside.safetensors", "/absolute.safetensors", "C:/outside.safetensors", 1])
def test_rejects_unsafe_or_malformed_index_paths(tmp_path, name):
    root = create_model(tmp_path, sharded=True)
    edit_json(root / "model.safetensors.index.json", lambda data: data["weight_map"].update({"bad": name}))
    with pytest.raises(ValueError, match="unsafe|Invalid"):
        validate_model(root, "qwen3-4b")


def test_index_and_shard_must_agree_on_tensors_and_bytes(tmp_path):
    root = create_model(tmp_path, sharded=True)
    edit_json(root / "model.safetensors.index.json", lambda data: data["metadata"].update(total_size=16))
    with pytest.raises(ValueError, match="total_size"):
        validate_model(root, "qwen3-4b")
    edit_json(root / "model.safetensors.index.json", lambda data: data["weight_map"].update({"absent": "model-00001-of-00002.safetensors"}))
    with pytest.raises(ValueError, match="tensor names disagree"):
        validate_model(root, "qwen3-4b")


def test_cli_json_report_and_nonzero_failure(tmp_path):
    root = create_model(tmp_path)
    script = Path(__file__).resolve().parents[2] / "tools" / "preflight_model.py"
    command = [sys.executable, str(script), "--model-path", str(root), "--model-key", "qwen3-4b", "--json"]
    success = subprocess.run(command, text=True, capture_output=True, check=False)
    assert success.returncode == 0, success.stderr
    assert json.loads(success.stdout)["model_key"] == "qwen3-4b"
    (root / "model.safetensors").unlink()
    failure = subprocess.run(command, text=True, capture_output=True, check=False)
    assert failure.returncode == 1
    assert "FAILED" in failure.stderr


def test_preflight_accepts_official_qwen25_7b_config_snapshot(tmp_path):
    root = create_model(tmp_path, "qwen25-7b")
    snapshot = Path(__file__).parent / "fixtures" / "qwen25_7b_config.json"
    (root / "config.json").write_text(snapshot.read_text(), encoding="utf-8")
    report = validate_model(root, "qwen25-7b")
    assert report["model_key"] == "qwen25-7b"
    assert report["max_position_embeddings"] == 128000
