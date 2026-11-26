from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, UnidentifiedImageError

from general_video_eval.backend import multimodal_input
from general_video_eval.media import decode_media
from general_video_eval.schema import Sample


class SeedBenchMediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "question"
        self.root.mkdir()
        # Neither alphabetical nor numeric order matches this explicit order.
        self.names = [f"frame_{number}.jpg" for number in (30, 3, 50, 2, 44, 9, 21, 1)]
        self.values = [20 + 25 * index for index in range(8)]
        for name, value in zip(self.names, self.values):
            Image.new("RGB", (84, 56), (value, value, value)).save(
                self.root / name, quality=100, subsampling=0)
        self.sample = Sample(
            uid="seedbench:fixture", benchmark="seedbench", question="What happens?",
            choices=("first", "second", "third", "fourth"), answer="A",
            media_path=str(self.root), media_type="frames", frame_fps=2.0,
            metadata={"sampling": "seedbench_official_8", "frame_files": self.names},
        )

    def fixture_with_names(self, names):
        return replace(self.sample, metadata={**self.sample.metadata, "frame_files": names})

    def test_exact_eight_jpegs_preserve_explicit_order_without_resampling(self):
        # An unrelated ninth file is not part of this question's eight frames.
        (self.root / "frame_0.jpg").write_text("Not an image", encoding="utf-8")
        for sampling in (None, "benchmark_default", "uniform_fps_floor"):
            with self.subTest(sampling=sampling), patch(
                "general_video_eval.media.frame_plan", side_effect=AssertionError("must not resample")
            ):
                video, info = decode_media(
                    self.sample, max_frames=8, fps=0.01, min_frames=64, sampling=sampling,
                    min_pixels=100352, max_pixels=200704,
                )
            self.assertEqual(video.shape[0], 8)
            self.assertEqual(video.shape[-1], 3)
            self.assertEqual(video.dtype, np.uint8)
            np.testing.assert_allclose(video.mean(axis=(1, 2, 3)), self.values, atol=1)
            self.assertEqual(info["frame_files"], self.names)
            self.assertEqual(info["sampled_frame_indices"], list(range(8)))
            self.assertEqual(info["sampled_frames"], 8)
            self.assertEqual(info["sampling"], "seedbench_official_8")
            self.assertEqual(info["decoder"], "pillow_frames")
            self.assertEqual(video.shape[1] % 28, 0)
            self.assertEqual(video.shape[2] % 28, 0)
            self.assertGreaterEqual(video.shape[1] * video.shape[2], 100352)
            self.assertLessEqual(video.shape[1] * video.shape[2], 200704)

    def test_grayscale_jpeg_becomes_rgb(self):
        Image.new("L", (84, 56), 67).save(self.root / self.names[3], quality=100)
        video, _ = decode_media(self.sample, max_frames=8)
        self.assertEqual(video.shape[-1], 3)
        np.testing.assert_array_equal(video[3, :, :, 0], video[3, :, :, 1])
        np.testing.assert_array_equal(video[3, :, :, 0], video[3, :, :, 2])
        self.assertAlmostEqual(float(video[3].mean()), 67, delta=1)

    def test_synthetic_timing_is_explicit_and_satisfies_both_backend_contracts(self):
        video, info = decode_media(self.sample, max_frames=8)
        self.assertFalse(info["source_timing_known"])
        self.assertTrue(info["timing_is_synthetic"])
        self.assertEqual(info["timing_source"], "synthetic_ordinal_2fps")
        self.assertEqual(info["sampled_timestamps"], [index / 2 for index in range(8)])
        self.assertEqual(info["source_fps"], 2.0)
        self.assertEqual(info["sampled_fps"], 2.0)
        self.assertEqual(info["source_duration"], 4.0)
        self.assertIsNone(info["start"])
        self.assertIsNone(info["end"])
        mm25, kwargs25 = multimodal_input(video, info, model_type="qwen2_5_vl")
        self.assertIs(mm25["video"], video)
        self.assertEqual(kwargs25["fps"], 2.0)
        self.assertFalse(kwargs25["do_sample_frames"])
        mm3, kwargs3 = multimodal_input(video, info, model_type="qwen3_vl")
        self.assertIs(mm3["video"][0], video)
        self.assertEqual(mm3["video"][1]["frames_indices"], list(range(8)))
        self.assertEqual(mm3["video"][1]["fps"], 2.0)
        self.assertFalse(kwargs3["do_sample_frames"])

    def test_only_max_frames_eight_is_accepted(self):
        for count in (0, 4, 16, 64):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "max_frames=8"):
                decode_media(self.sample, max_frames=count)

    def test_clipping_and_claimed_source_fps_are_rejected(self):
        for fields in ({"start": 0.0}, {"end": 4.0}, {"frame_fps": 3.0}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                decode_media(replace(self.sample, **fields), max_frames=8)

    def test_official_marker_and_frame_media_type_are_required(self):
        for fields in (
            {"metadata": {"frame_files": self.names}},
            {"media_type": "video"},
        ):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "official eight-frame"):
                decode_media(replace(self.sample, **fields), max_frames=8)

    def test_exact_eight_explicit_names_are_required(self):
        for names in (None, "frame_30.jpg", [], self.names[:7], self.names + ["extra.jpg"]):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "exactly 8"):
                decode_media(self.fixture_with_names(names), max_frames=8)

    def test_missing_or_corrupt_eighth_frame_fails_without_dropping_it(self):
        last = self.root / self.names[-1]
        last.unlink()
        with self.assertRaises(FileNotFoundError):
            decode_media(self.sample, max_frames=8)
        last.write_text("Corrupted image", encoding="utf-8")
        with self.assertRaises(UnidentifiedImageError):
            decode_media(self.sample, max_frames=8)

    def test_unsafe_relative_paths_are_rejected(self):
        for name in ("../outside.jpg", str(self.root / self.names[0]),
                     "sub/../../outside.jpg", "..\\outside.jpg", "", "bad\x00.jpg", None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe relative"):
                decode_media(self.fixture_with_names([name, *self.names[1:]]), max_frames=8)

    def test_duplicate_paths_and_aliases_are_rejected(self):
        for name in (self.names[0], "./" + self.names[0]):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "duplicate"):
                decode_media(self.fixture_with_names([*self.names[:-1], name]), max_frames=8)

    def test_symlink_escaping_directory_is_rejected(self):
        outside = Path(self.temporary.name) / "outside.jpg"
        Image.new("RGB", (84, 56)).save(outside)
        (self.root / "escape.jpg").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes its directory"):
            decode_media(self.fixture_with_names(["escape.jpg", *self.names[1:]]), max_frames=8)

    def test_symlink_alias_duplicate_is_rejected(self):
        (self.root / "alias.jpg").symlink_to(self.root / self.names[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            decode_media(self.fixture_with_names([*self.names[:-1], "alias.jpg"]), max_frames=8)


if __name__ == "__main__":
    unittest.main()
