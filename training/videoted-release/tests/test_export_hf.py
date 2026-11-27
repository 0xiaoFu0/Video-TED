"""Checkpoint export tests with small CPU models."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import export_hf as exporter


SCRIPT = Path(__file__).resolve().parents[1] / "tools/export_hf.py"


def metadata_fixture(tmp_path, world=1):
    run = tmp_path / "run"
    actor = run / "global_step_10" / "actor"
    metadata = actor / "huggingface"
    metadata.mkdir(parents=True)
    (run / "checkpoint_tracker.json").write_text(json.dumps({"last_global_step": 10}))
    (actor.parent / "dataloader.pt").write_bytes(b"saved dataloader")
    config = {"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}
    (metadata / "config.json").write_text(json.dumps(config))
    (metadata / "tokenizer_config.json").write_text(json.dumps({"chat_template": "saved template"}))
    (metadata / "preprocessor_config.json").write_text("{}")
    (metadata / "tokenizer.json").write_text("{}")
    (metadata / "chat_template.jinja").write_text("saved exact template\n")
    for rank in range(world):
        (actor / f"model_world_size_{world}_rank_{rank}.pt").write_bytes(b"placeholder")
    return actor


def test_dry_run_reports_checkpoint_without_exporting(tmp_path):
    actor = metadata_fixture(tmp_path, world=2)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(actor.parent), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["model_type"] == "qwen3_vl" and report["shards"] == 2
    assert not Path(report["output"]).exists()


def test_export_rejects_incomplete_checkpoint(tmp_path):
    actor = metadata_fixture(tmp_path, world=2)
    (actor / "model_world_size_2_rank_1.pt").unlink()
    with pytest.raises(ValueError):
        exporter.export(actor.parent, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def tiny_model(family):
    import torch
    from transformers import (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                              Qwen3VLConfig, Qwen3VLForConditionalGeneration)
    torch.manual_seed(17)
    if family == "qwen2_5_vl":
        config = Qwen2_5_VLConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                 vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                                                "num_heads": 2, "patch_size": 2, "spatial_merge_size": 2,
                                                "temporal_patch_size": 2, "out_hidden_size": 16,
                                                "window_size": 8, "fullatt_block_indexes": [0]})
        model = Qwen2_5_VLForConditionalGeneration(config)
    else:
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             max_position_embeddings=128, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                             pad_token_id=0, eos_token_id=2),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                               in_channels=3, patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                               out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0]),
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
            tie_word_embeddings=False,
        )
        model = Qwen3VLForConditionalGeneration(config)
    model.config.architectures = [type(model).__name__]
    return model.to(torch.bfloat16)


@pytest.mark.parametrize("family,legacy", [("qwen2_5_vl", False), ("qwen2_5_vl", True), ("qwen3_vl", False)])
def test_real_tiny_model_export_load_roundtrip(tmp_path, family, legacy):
    import torch
    original = tiny_model(family)
    actor = metadata_fixture(tmp_path)
    metadata = actor / "huggingface"
    original.config.save_pretrained(metadata)
    original.generation_config.save_pretrained(metadata)
    saved_generation = (metadata / "generation_config.json").read_bytes()
    saved_template = (metadata / "chat_template.jinja").read_bytes()
    state = original.state_dict()
    checkpoint_state = {}
    for key, value in state.items():
        if legacy:
            key = key.replace("model.language_model.", "model.").replace("model.visual.", "visual.")
        checkpoint_state[key] = value
    torch.save(checkpoint_state, actor / "model_world_size_1_rank_0.pt")
    target = exporter.export(actor.parent)
    loaded = type(original).from_pretrained(target, torch_dtype=torch.bfloat16, local_files_only=True)
    assert set(loaded.state_dict()) == set(state)
    for key, tensor in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)
    assert (target / "generation_config.json").read_bytes() == saved_generation
    assert (target / "chat_template.jinja").read_bytes() == saved_template
    assert json.loads((target / "EXPORT_COMPLETE.json").read_text())["status"] == "complete"
    assert (actor / "model_world_size_1_rank_0.pt").exists()
    with pytest.raises(ValueError, match="already exists"):
        exporter.export(actor, target)


def test_nonzero_shard_dimension_uneven_sizes_and_permuted_mesh():
    import torch
    expected = torch.arange(15).reshape(3, 5)
    parts = [expected[:, 3:], expected[:, :3]]
    actual = exporter.combine(parts, (("shard", 1),), (2,), (1, 0), (3, 5))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hybrid_mesh_replicas_and_divergence():
    import torch
    first, second = torch.tensor([[1, 2]]), torch.tensor([[3, 4]])
    actual = exporter.combine([first, second, first, second], (("replicate", None), ("shard", 0)),
                              (2, 2), (0, 1, 2, 3), (2, 2))
    torch.testing.assert_close(actual, torch.cat([first, second]), rtol=0, atol=0)
    with pytest.raises(ValueError, match="Replicated tensor"):
        exporter.combine([first, second, first + 1, second], (("replicate", None), ("shard", 0)),
                         (2, 2), (0, 1, 2, 3), (2, 2))


def test_real_tensor_multirank_replicas_are_not_concatenated(tmp_path):
    import torch
    paths = [tmp_path / str(rank) for rank in range(2)]
    for path in paths:
        torch.save({"scalar": torch.tensor(3), "buffer": torch.tensor([4, 5])}, path)
    merged = exporter.merge_checkpoint(paths)
    assert merged["scalar"].shape == () and merged["buffer"].shape == (2,)
    torch.save({"scalar": torch.tensor(8), "buffer": torch.tensor([4, 5])}, paths[1])
    with pytest.raises(ValueError, match="Replicated tensor"):
        exporter.merge_checkpoint(paths)
