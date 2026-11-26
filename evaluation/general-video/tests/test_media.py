from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from general_video_eval.backend import MockBackend, VLLMBackend, multimodal_input
from general_video_eval.media import decode_media, frame_files, frame_plan, smart_resize
from general_video_eval.schema import Sample


def sample(path, *, benchmark="mvbench", media_type="frames", start=None, end=None):
    return Sample(uid="fixture", benchmark=benchmark, question="What happened?",
                  choices=("a", "b"), answer="B", media_path=str(path),
                  media_type=media_type, start=start, end=end)


class SamplingTests(unittest.TestCase):
    def test_floor_sampler_long_clip_64_cap_keeps_endpoints_and_metadata(self):
        baseline = frame_plan(18000, 30, max_frames=64)
        plan = frame_plan(18000, 30, max_frames=64, sampling="uniform_fps_floor")
        self.assertEqual(plan["sampled_frames"], 64)
        self.assertEqual(plan["sampled_frame_indices"][0], 0)
        self.assertEqual(plan["sampled_frame_indices"][-1], 17999)
        self.assertAlmostEqual(plan["sampled_fps"], 64 / 600)
        self.assertEqual({k: v for k, v in plan.items() if k != "sampling"},
                         {k: v for k, v in baseline.items() if k != "sampling"})

    def test_floor_sampler_short_clip_uses_fewer_than_64_frames(self):
        plan = frame_plan(345, 30, max_frames=64, sampling="uniform_fps_floor")
        self.assertEqual(plan["sampled_frames"], 22)
        self.assertEqual(plan["sampled_frame_indices"][0], 0)
        self.assertEqual(plan["sampled_frame_indices"][-1], 344)
        self.assertEqual(frame_plan(345, 30, max_frames=64)["sampled_frames"], 24)
        self.assertAlmostEqual(plan["sampled_fps"], 22 / 11.5)

    def test_floor_sampler_respects_available_source_frames(self):
        plan = frame_plan(20, 1, max_frames=64, sampling="uniform_fps_floor")
        self.assertEqual(plan["sampled_frames"], 20)
        self.assertEqual(plan["sampled_frame_indices"], list(range(20)))

    def test_floor_sampler_very_short_clip_retains_repeat_compatibility(self):
        for count in (1, 2, 3):
            with self.subTest(source_frames=count):
                plan = frame_plan(count, 30, max_frames=64, sampling="uniform_fps_floor")
                self.assertEqual(plan["sampled_frames"], 4)
                self.assertEqual(plan["sampled_frame_indices"][0], 0)
                self.assertEqual(plan["sampled_frame_indices"][-1], count - 1)
                self.assertTrue(all(0 <= index < count for index in plan["sampled_frame_indices"]))

    def test_floor_sampler_keeps_annotated_clip_bounds_and_absolute_timing(self):
        plan = frame_plan(1000, 10, max_frames=64, start=20, end=31.4,
                          sampling="uniform_fps_floor")
        self.assertEqual(plan["sampled_frames"], 22)
        self.assertEqual(plan["sampled_frame_indices"][0], 200)
        self.assertEqual(plan["sampled_frame_indices"][-1], 314)
        self.assertTrue(all(200 <= index <= 314 for index in plan["sampled_frame_indices"]))
        self.assertEqual(plan["sampled_timestamps"][0], 20)
        self.assertEqual(plan["sampled_timestamps"][-1], 31.4)
        self.assertAlmostEqual(plan["sampled_fps"], 22 / 115 * 10)

    def test_uniform_capped_video_keeps_both_endpoints(self):
        plan = frame_plan(9000, 30, max_frames=64)
        self.assertEqual(plan["sampled_frames"], 64)
        self.assertEqual(plan["sampled_frame_indices"][0], 0)
        self.assertEqual(plan["sampled_frame_indices"][-1], 8999)
        self.assertAlmostEqual(plan["sampled_fps"], 64 / 300)

    def test_short_video_intentionally_repeats(self):
        plan = frame_plan(1, 30, max_frames=64)
        self.assertEqual(plan["sampled_frame_indices"], [0, 0, 0, 0])

    def test_mvbench_segment_formula_with_clip(self):
        plan = frame_plan(100, 10, max_frames=8, start=2, end=5, sampling="mvbench")
        expected = [int(20 + 30 / 8 / 2 + np.round(30 / 8 * index)) for index in range(8)]
        self.assertEqual(plan["sampled_frame_indices"], expected)
        self.assertTrue(all(20 <= i <= 50 for i in expected))
        self.assertAlmostEqual(plan["sampled_timestamps"][0], expected[0] / 10)

    def test_one_based_mvbench_directory(self):
        plan = frame_plan(20, 3, max_frames=4, sampling="mvbench", first_index=1, start=1, end=3)
        # Original TVQA image names are 1..N; decoder list positions are 0..N-1.
        expected_names = [int(3 + 6 / 4 / 2 + np.round(6 / 4 * index)) for index in range(4)]
        self.assertEqual(plan["sampled_frame_indices"], [name - 1 for name in expected_names])

    def test_invalid_or_outside_clip_fails(self):
        for args in ({"start": 20}, {"start": 2, "end": 1}, {"source_fps": float("nan")}, {"max_frames": 3}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                frame_plan(10, **{"source_fps": 3, **args})

    def test_resize_budget_and_patch_divisibility(self):
        for factor in (28, 32):
            h, w = smart_resize(2160, 3840, factor=factor)
            self.assertEqual(h % factor, 0)
            self.assertEqual(w % factor, 0)
            self.assertLessEqual(h * w, 262144)


class DecodingTests(unittest.TestCase):
    def test_explicit_floor_sampler_for_mvbench_frame_dir_preserves_clip_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(1, 21):
                Image.new("RGB", (64, 64), (index, 0, 0)).save(Path(directory) / f"{index:05d}.png")
            fixture = sample(directory, start=1, end=6)
            baseline_video, baseline = decode_media(fixture, max_frames=64)
            alias_video, alias = decode_media(fixture, max_frames=64, sampling="benchmark_default")
            np.testing.assert_array_equal(baseline_video, alias_video)
            self.assertEqual(baseline, alias)
            self.assertEqual(baseline["sampling"], "mvbench")
            video, info = decode_media(fixture, max_frames=64, sampling="uniform_fps_floor")
            self.assertEqual(info["sampling"], "uniform_fps_floor")
            self.assertEqual(video.shape[0], 10)
            self.assertEqual(info["sampled_frame_indices"][0], 2)
            self.assertEqual(info["sampled_frame_indices"][-1], 17)
            self.assertEqual([int(value) for value in video[:, 0, 0, 0]],
                             [i + 1 for i in info["sampled_frame_indices"]])
            self.assertAlmostEqual(info["sampled_fps"], 10 / 16 * 3)

    def test_frame_directory_crop_decodes_selected_colours(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(1, 21):
                Image.new("RGB", (64, 64), (index, 0, 0)).save(Path(directory) / f"{index:05d}.png")
            video, info = decode_media(sample(directory, start=1, end=3), max_frames=4)
            self.assertEqual(video.shape[0], 4)
            colours = [int(pixel) for pixel in video[:, 0, 0, 0]]
            self.assertEqual(colours, [i + 1 for i in info["sampled_frame_indices"]])
            self.assertEqual(info["decoder"], "pillow_frames")
            self.assertTrue(all(2 <= i <= 8 for i in info["sampled_frame_indices"]))

    def test_frame_directory_numeric_gaps_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in (1, 3):
                Image.new("RGB", (64, 64)).save(Path(directory) / f"{index:05d}.png")
            with self.assertRaisesRegex(ValueError, "missing numeric"):
                frame_files(directory)

    def test_gif_variable_duration_preserves_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gif"
            frames = [Image.new("RGB", (64, 64), colour)
                      for colour in ("red", "green", "blue", "yellow")]
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=[100, 200, 300, 400])
            for sampling in (None, "uniform_fps_floor"):
                with self.subTest(sampling=sampling):
                    video, info = decode_media(sample(path, media_type="gif"), max_frames=4, sampling=sampling)
                    self.assertEqual(video.shape[0], 4)
                    expected = [0, 0.1, 0.3, 0.6]
                    for index, timestamp in zip(info["sampled_frame_indices"], info["sampled_timestamps"]):
                        self.assertAlmostEqual(timestamp, expected[index])
                    mm, _ = multimodal_input(video, info, model_type="qwen3_vl")
                    metadata = mm["video"][1]
                    np.testing.assert_allclose(np.array(metadata["frames_indices"]) / metadata["fps"],
                                               info["sampled_timestamps"])

    def test_missing_media_never_skips(self):
        with self.assertRaises(FileNotFoundError):
            decode_media(sample("/does/not/exist"), max_frames=4)


class BackendContractTests(unittest.TestCase):
    def test_generate_never_puts_source_filename_or_ground_truth_in_chat(self):
        fixture = sample("/secret/ground_truth_action_B.mp4", media_type="video")
        video = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        info = {**frame_plan(60, 30, max_frames=4), "decoder": "decord"}
        rendered_messages, llm_requests = [], []

        def render(messages, **kwargs):
            rendered_messages.extend(messages)
            return json.dumps(messages)

        def generate(requests, **kwargs):
            llm_requests.extend(requests)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="<answer>A</answer>", finish_reason="stop")])]

        # Exercise request construction without importing CUDA/vLLM or loading
        # a checkpoint; actual token/video contracts are tested separately.
        backend = object.__new__(VLLMBackend)
        backend.config, backend.n, backend.resize_factor = {}, 1, 32
        backend.model_type, backend.sampling_params = "qwen3_vl", object()
        backend.processor = SimpleNamespace(apply_chat_template=render)
        backend.llm = SimpleNamespace(generate=generate)
        for configured in (None, "benchmark_default", "uniform_fps_floor"):
            with self.subTest(video_sampling=configured):
                backend.config = {} if configured is None else {"video_sampling": configured}
                with patch("general_video_eval.backend.decode_media", return_value=(video, info)) as decode:
                    result = backend.generate(fixture, "Read the video and choose a letter.", 4)
                self.assertEqual(decode.call_args.args[0].media_path, fixture.media_path)
                self.assertEqual(decode.call_args.kwargs["sampling"], configured)
        self.assertEqual(rendered_messages[0]["content"][0]["video"], "video.mp4")
        self.assertNotIn(fixture.media_path, llm_requests[0]["prompt"])
        self.assertNotIn("ground_truth_action_B", llm_requests[0]["prompt"])
        self.assertNotIn('"answer"', llm_requests[0]["prompt"])
        self.assertEqual(result["responses"], ["<answer>A</answer>"])

    def test_unsupported_architecture_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model_type"):
            multimodal_input(None, {}, model_type="qwen2_vl")

    def test_qwen3_metadata_keeps_original_source_timing(self):
        info = frame_plan(600, 30, max_frames=4, start=10, end=12, sampling="mvbench")
        info["decoder"] = "decord"
        video = np.zeros((4, 56, 56, 3), dtype=np.uint8)
        mm, kwargs = multimodal_input(video, info, model_type="qwen3_vl")
        self.assertIs(mm["video"][0], video)
        self.assertEqual(mm["video"][1]["frames_indices"], info["sampled_frame_indices"])
        self.assertEqual(mm["video"][1]["fps"], 30)
        self.assertFalse(kwargs["do_sample_frames"])
        self.assertFalse(kwargs["do_resize"])

    def test_qwen25_receives_sampled_fps(self):
        info = frame_plan(600, 30, max_frames=4)
        info["decoder"] = "decord"
        video = np.zeros((4, 56, 56, 3), dtype=np.uint8)
        mm, kwargs = multimodal_input(video, info, model_type="qwen2_5_vl")
        self.assertIs(mm["video"], video)
        self.assertAlmostEqual(kwargs["fps"], 0.2)

    def test_mock_does_not_leak_ground_truth(self):
        result = MockBackend({"think_mode": True, "n": 1}).generate(sample("fake"), "prompt", 4)
        self.assertEqual(result["responses"], ["<think>Mock output.</think><answer>A</answer>"])
        self.assertTrue(result["video_info"]["mock"])


if __name__ == "__main__":
    unittest.main()
