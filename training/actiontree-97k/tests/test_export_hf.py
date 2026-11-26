"""CPU-only export checks; tiny models do not establish full-model/GPU success."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/export_hf.py"
spec = importlib.util.spec_from_file_location("ar_export_hf", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


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


def test_metadata_dry_run_needs_no_dependencies_and_creates_no_output(tmp_path):
    actor = metadata_fixture(tmp_path, world=2)
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), str(actor.parent), "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["model_type"] == "qwen3_vl" and report["shards"] == 2
    assert not Path(report["output"]).exists()
    assert exporter.discover(actor)[0] == actor


@pytest.mark.parametrize("problem", ["missing_rank", "empty_rank", "extra_rank", "uncommitted", "missing_tracker", "missing_loader", "missing_template"])
def test_incomplete_checkpoint_rejected(tmp_path, problem):
    actor = metadata_fixture(tmp_path, world=2)
    if problem == "missing_rank":
        (actor / "model_world_size_2_rank_1.pt").unlink()
    elif problem == "empty_rank":
        (actor / "model_world_size_2_rank_1.pt").write_bytes(b"")
    elif problem == "extra_rank":
        (actor / "model_world_size_4_rank_3.pt").write_bytes(b"extra")
    elif problem == "uncommitted":
        (actor.parent.parent / "checkpoint_tracker.json").write_text('{"last_global_step": 9}')
    elif problem == "missing_tracker":
        (actor.parent.parent / "checkpoint_tracker.json").unlink()
    elif problem == "missing_loader":
        (actor.parent / "dataloader.pt").unlink()
    else:
        (actor / "huggingface/tokenizer_config.json").write_text("{}")
        (actor / "huggingface/chat_template.jinja").unlink()
    with pytest.raises(ValueError):
        exporter.discover(actor)


def test_output_is_independent_and_never_overwritten(tmp_path):
    actor = metadata_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        exporter.discover(actor, actor.parent / "model")
    target = tmp_path / "output"
    target.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        exporter.discover(actor, target)


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


def test_dtensor_merger_uses_metadata_and_checks_missing_keys(tmp_path, monkeypatch):
    import torch
    import torch.distributed.tensor
    class Placement:
        dim = 1
        def is_replicate(self): return False
        def is_shard(self): return True
    class FakeDTensor:
        def __init__(self, local):
            self.local = local
            self.shape = (2, 5)
            self.placements = [Placement()]
            self.device_mesh = SimpleNamespace(mesh_dim_names=("fsdp",), mesh=torch.tensor([1, 0]))
        def to_local(self): return self.local
    full = torch.arange(10).reshape(2, 5).float()
    shards = [{"weight": FakeDTensor(full[:, 3:])}, {"weight": FakeDTensor(full[:, :3])}]
    monkeypatch.setattr(torch.distributed.tensor, "DTensor", FakeDTensor)
    monkeypatch.setattr(torch, "load", lambda path, **kw: shards[int(path.name)])
    result = exporter.merge_checkpoint([Path("0"), Path("1")])
    torch.testing.assert_close(result["weight"], full, rtol=0, atol=0)
    shards[1]["other"] = torch.ones(1)
    with pytest.raises(ValueError, match="keys differ"):
        exporter.merge_checkpoint([Path("0"), Path("1")])


def test_bad_keys_shapes_and_conversion_collision_are_rejected():
    import torch
    expected = {"model.visual.weight": torch.zeros(2)}
    mapping = {"^visual": "model.visual"}
    with pytest.raises(ValueError, match="shape"):
        exporter.canonicalize_keys({"visual.weight": torch.zeros(3)}, expected, mapping)
    with pytest.raises(ValueError, match="collision"):
        exporter.canonicalize_keys({"visual.weight": torch.zeros(2), "model.visual.weight": torch.zeros(2)}, expected, mapping)
    with pytest.raises(ValueError, match="Missing checkpoint"):
        exporter.canonicalize_keys({}, expected, mapping)


def test_checkpoint_changed_during_export_never_publishes(tmp_path, monkeypatch):
    import torch
    actor = metadata_fixture(tmp_path)
    model = tiny_model("qwen3_vl")
    model.config.save_pretrained(actor / "huggingface")
    shard = actor / "model_world_size_1_rank_0.pt"
    torch.save(model.state_dict(), shard)
    original_merge = exporter.merge_checkpoint
    def changing_merge(paths):
        merged = original_merge(paths)
        (actor / "huggingface/chat_template.jinja").write_text("changed during export")
        return merged
    monkeypatch.setattr(exporter, "merge_checkpoint", changing_merge)
    target = tmp_path / "exported"
    with pytest.raises(ValueError, match="changed during export"):
        exporter.export(actor, target)
    assert not target.exists()
    assert not target.with_name(target.name + ".export.lock").exists()
