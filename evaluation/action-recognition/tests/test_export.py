import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_checkpoint import discover_shards, export


def test_missing_rank_rejected(tmp_path):
    (tmp_path / "model_world_size_2_rank_0.pt").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="分片不完整"):
        discover_shards(tmp_path)


@pytest.mark.parametrize("model_type", ["gpt2", "qwen2_5_vl"])
def test_cpu_export_roundtrip_and_independent_destination(tmp_path, model_type):
    import torch
    from transformers import (GPT2Config, GPT2LMHeadModel, Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration)
    torch.manual_seed(3)
    if model_type == "gpt2":
        config = GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=32, vocab_size=32)
        cls = GPT2LMHeadModel
    else:
        config = Qwen2_5_VLConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                 vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                                                "num_heads": 2, "patch_size": 2, "spatial_merge_size": 2,
                                                "temporal_patch_size": 2, "out_hidden_size": 16,
                                                "window_size": 8, "fullatt_block_indexes": [0]})
        cls = Qwen2_5_VLForConditionalGeneration
    original = cls(config).to(torch.bfloat16)
    original.config.architectures = [type(original).__name__]
    checkpoint = tmp_path / "global_step_10/actor"
    config_dir = checkpoint / "huggingface"
    config_dir.mkdir(parents=True)
    original.config.save_pretrained(config_dir)
    original.generation_config.save_pretrained(config_dir)
    state = original.state_dict()
    torch.save(state, checkpoint / "model_world_size_1_rank_0.pt")
    target = tmp_path / "exported"
    export(checkpoint, target)
    loaded = cls.from_pretrained(target, torch_dtype=torch.bfloat16)
    for key, value in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert (target / "EXPORT_COMPLETE.json").exists()
    assert (checkpoint / "model_world_size_1_rank_0.pt").exists()
    with pytest.raises(ValueError, match="已存在"):
        export(checkpoint, target)
    with pytest.raises(ValueError, match="之外"):
        export(checkpoint, checkpoint / "exported")


def test_multi_rank_concat_and_replicated_weights(monkeypatch):
    import torch
    import torch.distributed.tensor
    from types import SimpleNamespace
    from export_checkpoint import merge_checkpoint
    class Placement:
        dim = 0
        def __init__(self, replicated=False): self.replicated = replicated
        def is_replicate(self): return self.replicated
        def is_shard(self): return not self.replicated
    class FakeDTensor:
        def __init__(self, tensor, shape, replicated=False):
            self.tensor, self.shape = tensor, shape
            self.placements = [Placement(replicated)]
            self.device_mesh = SimpleNamespace(mesh_dim_names=("fsdp",))
        def to_local(self): return self.tensor
    shards = [
        {"weight": FakeDTensor(torch.arange(6).reshape(2, 3).float(), (4, 3)),
         "buffer": FakeDTensor(torch.tensor([4, 5]), (2,), True)},
        {"weight": FakeDTensor(torch.arange(6, 12).reshape(2, 3).float(), (4, 3)),
         "buffer": FakeDTensor(torch.tensor([4, 5]), (2,), True)},
    ]
    monkeypatch.setattr(torch.distributed.tensor, "DTensor", FakeDTensor)
    monkeypatch.setattr(torch, "load", lambda path, **kw: shards[int(path.name)])
    merged = merge_checkpoint([Path("0"), Path("1")])
    torch.testing.assert_close(merged["weight"], torch.arange(12).reshape(4, 3).to(torch.bfloat16))
    torch.testing.assert_close(merged["buffer"], torch.tensor([4, 5]))
