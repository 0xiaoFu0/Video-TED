"""Real tiny Transformers Qwen3-VL checks, not evidence of an 8B GPU run.

Run: python -m pytest tests/ar/test_qwen3_training.py -q
Dependencies: torch, transformers==4.57.3, torchvision, pillow, numpy.
"""

import copy
import types
import unittest

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from verl.models.transformers.qwen3_vl import get_rope_index, qwen3_vl_base_forward, qwen3_vl_model_forward
from verl.utils.position_ids import compute_position_ids


IMAGE, VIDEO, START, END = 60, 61, 62, 63


def tiny_model():
    torch.manual_seed(17)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            max_position_embeddings=512, rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
            pad_token_id=0, eos_token_id=2, use_cache=False,
        ),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
            in_channels=3, patch_size=4, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=64, num_position_embeddings=16, deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=IMAGE, video_token_id=VIDEO, vision_start_token_id=START,
        vision_end_token_id=END, tie_word_embeddings=False, attn_implementation="eager",
    )
    return Qwen3VLForConditionalGeneration(config).float()


class Qwen3VLProcessor:
    """Only the attributes used for RoPE, with a renamed image processor."""

    image_processor = types.SimpleNamespace(merge_size=2)
    image_token_id = IMAGE
    video_token_id = VIDEO
    vision_start_token_id = START


class WrappedProcessor(Qwen3VLProcessor):
    pass


def four_positions(ids, grids, mask=None):
    mask = torch.ones_like(ids) if mask is None else mask
    return compute_position_ids(ids, mask, grids, WrappedProcessor()).unsqueeze(1)


class Qwen3TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = tiny_model().eval()

    def test_rope_matches_upstream_image_video_and_text(self):
        cases = [
            ([9, START, IMAGE, IMAGE, IMAGE, IMAGE, END, 10], {"image_grid_thw": torch.tensor([[1, 4, 4]])}),
            ([9, START, IMAGE, IMAGE, IMAGE, IMAGE, END, 10, START, IMAGE, END, 11],
             {"image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]])}),
            ([9, START, VIDEO, VIDEO, VIDEO, VIDEO, END, 10, START, VIDEO, VIDEO, VIDEO, VIDEO, END, 11],
             {"video_grid_thw": torch.tensor([[2, 4, 4]])}),
            ([9, 10, 11], {}),
        ]
        for ids_list, grids in cases:
            for padded in (False, True):
                with self.subTest(grids=tuple(grids), padded=padded):
                    ids = torch.tensor(([0, 0] if padded else []) + ids_list)
                    mask = (ids != 0).long()
                    expected, _ = self.reference.model.get_rope_index(ids.unsqueeze(0), attention_mask=mask.unsqueeze(0), **grids)
                    actual = get_rope_index(WrappedProcessor(), ids, attention_mask=mask, **grids)
                    torch.testing.assert_close(actual, expected[:, 0])
                    dispatched = four_positions(ids, grids, mask)
                    self.assertEqual(dispatched.shape, (4, 1, len(ids)))
                    torch.testing.assert_close(dispatched[1:, 0], expected[:, 0])

    def test_text_rope_without_attention_mask(self):
        ids = torch.tensor([9, 10, 11])
        actual = get_rope_index(WrappedProcessor(), ids)
        expected, _ = self.reference.model.get_rope_index(ids.unsqueeze(0))
        torch.testing.assert_close(actual, expected[:, 0])
        with self.assertRaisesRegex(ValueError, "rank 1"):
            get_rope_index(WrappedProcessor(), ids.unsqueeze(0))

    def test_patched_logits_match_upstream_including_deepstack(self):
        for modality in ("image", "video", "mixed", "text"):
            with self.subTest(modality=modality):
                ids, vision = self.make_inputs(modality)
                positions = four_positions(ids[0], {key: value for key, value in vision.items() if "grid" in key})
                kwargs = dict(input_ids=ids, position_ids=positions, attention_mask=torch.ones_like(ids), use_cache=False, **vision)
                model = copy.deepcopy(self.reference)
                with torch.no_grad():
                    expected = model(**kwargs).logits
                    model.model.forward = types.MethodType(qwen3_vl_base_forward, model.model)
                    model.forward = types.MethodType(qwen3_vl_model_forward, model)
                    actual = model(**kwargs).logits
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    @staticmethod
    def make_inputs(modality):
        ids = [9]
        vision = {}
        if modality in ("image", "mixed"):
            ids += [START] + [IMAGE] * 4 + [END, 10]
            vision.update(pixel_values=torch.randn(16, 3 * 2 * 4 * 4), image_grid_thw=torch.tensor([[1, 4, 4]]))
        if modality in ("video", "mixed"):
            ids += [START] + [VIDEO] * 4 + [END, 10] + [START] + [VIDEO] * 4 + [END, 11]
            vision.update(pixel_values_videos=torch.randn(32, 3 * 2 * 4 * 4), video_grid_thw=torch.tensor([[2, 4, 4]]))
        ids += [12, 13]
        return torch.tensor([ids]), vision

    def test_patched_backward_reaches_vision_deepstack_and_lm_head(self):
        model = copy.deepcopy(self.reference).train()
        model.model.forward = types.MethodType(qwen3_vl_base_forward, model.model)
        model.forward = types.MethodType(qwen3_vl_model_forward, model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ids, vision = self.make_inputs("image")
        positions = four_positions(ids[0], {"image_grid_thw": vision["image_grid_thw"]})
        logits = model(input_ids=ids, position_ids=positions, attention_mask=torch.ones_like(ids), use_cache=False, **vision).logits
        loss = torch.nn.functional.cross_entropy(logits[:, -2].float(), ids[:, -1])
        loss.backward()
        modules = [model.lm_head, model.model.visual.patch_embed, *model.model.visual.deepstack_merger_list]
        for module in modules:
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_packed_rows_preserve_logits_and_deepstack_order(self):
        model = copy.deepcopy(self.reference)
        model.model.forward = types.MethodType(qwen3_vl_base_forward, model.model)
        model.forward = types.MethodType(qwen3_vl_model_forward, model)
        rows = [self.make_inputs("image"), self.make_inputs("image")]
        rows[1] = (torch.cat((torch.tensor([[19, 20, 21]]), rows[1][0]), dim=-1), rows[1][1])
        logits = []
        positions = []
        with torch.no_grad():
            for ids, vision in rows:
                pos = four_positions(ids[0], {"image_grid_thw": vision["image_grid_thw"]})
                positions.append(pos)
                logits.append(model(input_ids=ids, position_ids=pos, use_cache=False, **vision).logits)
            packed = model(
                input_ids=torch.cat([row[0] for row in rows], dim=-1),
                position_ids=torch.cat(positions, dim=-1),
                pixel_values=torch.cat([row[1]["pixel_values"] for row in rows]),
                image_grid_thw=torch.cat([row[1]["image_grid_thw"] for row in rows]),
                use_cache=False,
            ).logits
        torch.testing.assert_close(packed, torch.cat(logits, dim=1), atol=3e-6, rtol=3e-5)

    def test_hf_state_dict_preserves_qwen3_names_for_vllm_mapper(self):
        # vLLM 0.11.0 Qwen3VLForConditionalGeneration.load_weights applies its
        # own hf_to_vllm_mapper; Qwen2's legacy rename must not be applied.
        self.assertEqual(self.reference._checkpoint_conversion_mapping, {})
        keys = self.reference.state_dict()
        self.assertIn("model.visual.patch_embed.proj.weight", keys)
        self.assertIn("model.language_model.layers.0.self_attn.q_proj.weight", keys)
        self.assertIn("lm_head.weight", keys)


if __name__ == "__main__":
    unittest.main()
