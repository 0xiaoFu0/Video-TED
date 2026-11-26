"""Exercise AR teacher prompt tokenization with the real HF Qwen3 processor.

All tokenizer assets are created in memory; there is no model download.
"""

from pathlib import Path
import os
import tempfile
import unittest

from PIL import Image
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoProcessor, Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLProcessor, Qwen3VLVideoProcessor

from verl.trainer.opsd_algos import _tokenize_prompt
from verl.utils.video import prepare_video_data, tokenize_video, video_model_inputs, vllm_video_inputs
from verl.utils.position_ids import compute_position_ids
from verl.ar.check_video import check_qwen3_replay


def tiny_processor():
    vocabulary = {
        "<unk>": 0, "<pad>": 1, "<eos>": 2, "hello": 3,
        "<|image_pad|>": 4, "<|video_pad|>": 5, "<|vision_start|>": 6, "<|vision_end|>": 7,
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": list(vocabulary)[4:]})
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessorFast(
            patch_size=4, temporal_patch_size=2, merge_size=2, size={"shortest_edge": 256, "longest_edge": 256},
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=4, temporal_patch_size=2, merge_size=2, size={"shortest_edge": 256, "longest_edge": 1024},
        ),
    )


class ProcessorReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = tiny_processor()

    def assert_replay(self, prompt, multimodal, **processor_kwargs):
        expected = self.processor(text=[prompt], add_special_tokens=False, return_tensors="pt", **processor_kwargs)
        ids, mask, positions, raw_ids, raw_prompt = _tokenize_prompt(
            prompt, self.processor.tokenizer, self.processor, multimodal, max_prompt_length=512,
        )
        torch.testing.assert_close(ids[mask.bool()], expected["input_ids"][0])
        self.assertEqual(positions.shape, (4, 512))
        self.assertEqual(raw_prompt, prompt)
        self.assertEqual(raw_ids, self.processor.tokenizer.encode(prompt, add_special_tokens=False))
        return ids, mask, positions

    def test_image_teacher_and_continuation(self):
        images = [Image.new("RGB", (16, 16), (value, 20, 30)) for value in (10, 70)]
        visual = "<|vision_start|><|image_pad|><|vision_end|>"
        for suffix in (" answer evidence", " partial response candidate"):
            self.assert_replay(f"0.0 seconds {visual} 0.5 seconds {visual} hello{suffix}", {"images": images}, images=images)

    def test_native_video_worker_rollout_and_mrope_replay(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

        processor = self.processor
        prompt = "hello <|vision_start|><|video_pad|><|vision_end|> hello"
        side = processor.video_processor.patch_size * processor.video_processor.merge_size * 2
        frames = torch.arange(4 * 3 * side * side).reshape(4, 3, side, side).remainder(256).to(torch.uint8)
        prepared = prepare_video_data({
            "video": [frames], "video_metadatas": [{
                "fps": 4.0, "frames_indices": [0, 2, 4, 6], "total_num_frames": 8,
            }],
        }, processor)
        encoded = tokenize_video(processor, prompt, prepared)
        worker = video_model_inputs(processor, prepared)
        for key in ("pixel_values_videos", "video_grid_thw"):
            torch.testing.assert_close(worker[key], encoded[key], rtol=0, atol=0)
        mm_data, mm_kwargs = vllm_video_inputs(processor, prepared)
        positions = compute_position_ids(encoded["input_ids"][0], encoded["attention_mask"][0], encoded, processor)
        config = Qwen3VLConfig(
            vision_config={"spatial_merge_size": processor.image_processor.merge_size},
            image_token_id=processor.image_token_id, video_token_id=processor.video_token_id,
            vision_start_token_id=processor.vision_start_token_id,
        )
        check_qwen3_replay(processor, prompt, encoded, mm_data, mm_kwargs, positions[1:], config)

    def test_cached_native_video_matches_in_memory_replay(self):
        prompt = "hello <|vision_start|><|video_pad|><|vision_end|> hello"
        frames = torch.arange(4 * 3 * 16 * 16).reshape(4, 3, 16, 16).remainder(256).to(torch.uint8)
        metadata = {"fps": 4.0, "frames_indices": [0, 2, 4, 6], "total_num_frames": 8}
        kwargs = dict(videos=[frames], video_metadata=[metadata], do_resize=False, do_sample_frames=False)
        in_memory = self.assert_replay(prompt, {"video": [frames], "video_metadatas": [metadata]}, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.pt"
            torch.save({"frames": frames, "metadata": metadata}, path)
            cached = self.assert_replay(prompt, {"preprocessed_video_path": str(path)}, **kwargs)
        for actual, expected in zip(cached, in_memory):
            torch.testing.assert_close(actual, expected)


@unittest.skipUnless(os.environ.get("QWEN3VL_PROCESSOR_PATH"), "set QWEN3VL_PROCESSOR_PATH for local official assets")
class OfficialProcessorReplayTests(ProcessorReplayTests):
    """The same checks against a local Qwen3-VL-8B tokenizer/processor.

    Model weights are not loaded. Explicitly local-only, so setting this
    variable never initiates a download or authenticates to a model hub.
    """

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(
            os.environ["QWEN3VL_PROCESSOR_PATH"], local_files_only=True, trust_remote_code=False,
        )

    def test_cached_native_video_matches_in_memory_replay(self):
        prompt = "hello <|vision_start|><|video_pad|><|vision_end|> hello"
        frames = torch.zeros(4, 3, 64, 64, dtype=torch.uint8)
        metadata = {"fps": 4.0, "frames_indices": [0, 2, 4, 6], "total_num_frames": 8}
        kwargs = dict(videos=[frames], video_metadata=[metadata], do_resize=False, do_sample_frames=False)
        in_memory = self.assert_replay(prompt, {"video": [frames], "video_metadatas": [metadata]}, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.pt"
            torch.save({"frames": frames, "metadata": metadata}, path)
            cached = self.assert_replay(prompt, {"preprocessed_video_path": str(path)}, **kwargs)
        for actual, expected in zip(cached, in_memory):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
