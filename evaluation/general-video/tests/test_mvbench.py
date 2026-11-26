"""MVBench layout and clip contracts; no model or dataset download required."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from general_video_eval.loaders.mvbench import TASKS, load_mvbench


class MVBenchLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def write_json(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def row(self, video="example.mp4", **extra):
        return {"video": video, "question": "What happens?", "candidates": ["Walk", "Run"],
                "answer": "Run", **extra}

    def test_all_official_tasks_and_bounded_clip_metadata(self):
        for _task, (stem, _prefix, _kind, bounded) in TASKS.items():
            self.write_json(f"json/{stem}.json", [self.row(start=2, end=5)] if bounded else [self.row()])
        clip = self.root / "video/star/Charades_v1_480/example.mp4"
        clip.parent.mkdir(parents=True)
        clip.touch()
        samples = load_mvbench(self.root)
        self.assertEqual(len(samples), 20)
        sample = samples[0]
        self.assertEqual((sample.start, sample.end), (2.0, 5.0))
        self.assertEqual(sample.media_path, str(clip))
        self.assertEqual(sample.answer, "B")
        self.assertEqual(sample.metadata["task"], "Action Sequence")
        self.assertEqual(sample.metadata["sampling"], "segment_midpoints")
        fa = next(sample for sample in samples if sample.metadata["task"] == "Fine-grained Action")
        self.assertIsNone(fa.start)
        self.assertFalse(fa.metadata["converted_clip"])

    def test_partial_default_fails_but_explicit_file_is_debuggable(self):
        source = self.write_json("json/fine_grained_action.json", [self.row()])
        with self.assertRaisesRegex(FileNotFoundError, "Complete MVBench"):
            load_mvbench(self.root)
        sample, = load_mvbench(self.root, source)
        self.assertEqual(sample.metadata["task"], "Fine-grained Action")

    def test_relative_annotation_override_resolves_against_data_root(self):
        self.write_json("custom/fine_grained_action.json", [self.row()])
        sample, = load_mvbench(self.root, "custom/fine_grained_action.json")
        self.assertEqual(sample.metadata["task"], "Fine-grained Action")
        self.assertEqual(sample.metadata["annotation_file"], str(self.root / "custom/fine_grained_action.json"))

    def test_explicit_annotation_directory_requires_complete_tasks(self):
        for _task, (stem, _prefix, _kind, bounded) in TASKS.items():
            self.write_json(f"custom/{stem}.json", [self.row(start=2, end=5)] if bounded else [self.row()])
        self.assertEqual(len(load_mvbench(self.root, "custom")), 20)
        self.assertEqual(len(load_mvbench(self.root, self.root / "custom")), 20)
        (self.root / "custom/fine_grained_action.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing: fine_grained_action.json"):
            load_mvbench(self.root, "custom")

    def test_tvqa_frame_directory_and_data0613_extraction(self):
        source = self.write_json("json/episodic_reasoning.json", [self.row("episode", start=10, end=13)])
        frames = self.root / "video/data0613/tvqa/frames_fps3_hq/episode"
        frames.mkdir(parents=True)
        (frames / "00001.jpg").touch()
        sample, = load_mvbench(self.root, source)
        self.assertEqual(sample.media_type, "frames")
        self.assertEqual(sample.frame_fps, 3)
        self.assertEqual(sample.media_path, str(frames))
        self.assertEqual((sample.start, sample.end), (10, 13))

    def test_converted_test_json_does_not_crop_again(self):
        source = self.write_json("test.json", {
            "root": {"Action Sequence": "video/Action Sequence", "Episodic Reasoning": "video/TVQA"},
            "meta": {"Action Sequence": [self.row(start=4, end=6)],
                     "Episodic Reasoning": [self.row(start=10, end=13)]},
        })
        first = self.root / "video/Action Sequence/example.mp4"
        first.parent.mkdir(parents=True)
        first.touch()
        samples = load_mvbench(self.root, source)
        self.assertEqual(samples[0].media_path, str(first))
        for sample in samples:
            self.assertEqual(sample.media_type, "video")
            self.assertIsNone(sample.start)
            self.assertIsNone(sample.end)
            self.assertTrue(sample.metadata["converted_clip"])
        with self.assertRaisesRegex(ValueError, "incomplete MVBench"):
            load_mvbench(self.root)

    def test_original_tsv_keeps_bounds_and_literal_candidates(self):
        source = self.root / "MVBench.tsv"
        row = self.row(start="2", end="5", prefix="your_data_path/star/Charades_v1_480/",
                       task_type="Action Sequence", data_type="video", bound="True")
        row["candidates"] = repr(row["candidates"])
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), delimiter="\t")
            writer.writeheader()
            writer.writerow(row)
        sample, = load_mvbench(self.root, source)
        self.assertEqual((sample.start, sample.end), (2, 5))
        self.assertEqual(sample.choices, ("Walk", "Run"))
        self.assertEqual(sample.media_path, str(self.root / "video/star/Charades_v1_480/example.mp4"))

    def test_mp4_tsv_marks_tvqa_as_video_and_ignores_bounds(self):
        source = self.root / "MVBench_MP4.tsv"
        row = self.row(start="2", end="5", prefix="video/TVQA", task_type="Episodic Reasoning")
        row["candidates"] = repr(row["candidates"])
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), delimiter="\t")
            writer.writeheader()
            writer.writerow(row)
        sample, = load_mvbench(self.root, source)
        self.assertEqual(sample.media_type, "video")
        self.assertIsNone(sample.start)
        self.assertIsNone(sample.end)
        self.assertEqual(sample.media_path, str(self.root / "video/TVQA/example.mp4"))

    def test_unbounded_task_ignores_unrelated_time_fields(self):
        source = self.write_json("json/fine_grained_action.json", [self.row(start=20, end=30)])
        sample, = load_mvbench(self.root, source)
        self.assertIsNone(sample.start)
        self.assertIsNone(sample.end)

    def test_original_bound_requires_complete_interval(self):
        source = self.write_json("json/action_sequence.json", [self.row(start=20)])
        with self.assertRaisesRegex(ValueError, "requires start/end"):
            load_mvbench(self.root, source)

    def test_gif_and_direct_video_root(self):
        source = self.write_json("json/fine_grained_action.json", [self.row("example.gif")])
        gif = self.root / "Moments_in_Time_Raw/videos/example.gif"
        gif.parent.mkdir(parents=True)
        gif.touch()
        sample, = load_mvbench(self.root, source)
        self.assertEqual(sample.media_type, "gif")
        self.assertEqual(sample.media_path, str(gif))

    def test_fine_pose_avi_prefers_original_when_mp4_also_exists(self):
        source = self.write_json("json/fine_grained_pose.json", [self.row("S010C002P017R002A005_rgb.avi")])
        original = self.root / "nturgbd/S010C002P017R002A005_rgb.avi"
        original.parent.mkdir()
        original.touch()
        original.with_suffix(".mp4").touch()
        sample, = load_mvbench(self.root, source)
        self.assertEqual(sample.media_path, str(original))
        self.assertEqual(sample.metadata["source_video"], original.name)
        self.assertFalse(sample.metadata["converted_clip"])

    def test_fine_pose_missing_avi_uses_same_prefix_mp4_without_recropping(self):
        source = self.write_json("json/fine_grained_pose.json", [self.row("S010C002P017R002A005_rgb.avi", bound=True, start=2, end=5)])
        mp4 = self.root / "nturgbd/S010C002P017R002A005_rgb.mp4"
        mp4.parent.mkdir()
        mp4.touch()
        sample, = load_mvbench(self.root, source)
        self.assertEqual(sample.media_path, str(mp4))
        self.assertEqual(sample.metadata["source_video"], "S010C002P017R002A005_rgb.avi")
        self.assertEqual(sample.metadata["source_prefix"], "nturgbd")
        self.assertEqual((sample.start, sample.end), (2, 5))
        self.assertFalse(sample.metadata["converted_clip"])

    def test_fine_pose_alternate_convert_prefix_supports_known_layouts(self):
        source = self.write_json("json/fine_grained_pose.json", [self.row("example.avi")])
        for relative in ("nturgbd_convert", "video/nturgbd_convert", "videos/nturgbd_convert", "video/data0613/nturgbd_convert"):
            with self.subTest(relative=relative):
                mp4 = self.root / relative / "example.mp4"
                mp4.parent.mkdir(parents=True, exist_ok=True)
                mp4.touch()
                sample, = load_mvbench(self.root, source)
                self.assertEqual(sample.media_path, str(mp4))
                self.assertFalse(sample.metadata["converted_clip"])
                mp4.unlink()

    def test_fine_pose_missing_both_formats_retains_original_avi_path(self):
        source = self.write_json("json/fine_grained_pose.json", [self.row("example.avi")])
        sample, = load_mvbench(self.root, source)
        self.assertEqual(sample.media_path, str(self.root / "video/nturgbd/example.avi"))

    def test_mp4_extension_fallback_is_not_applied_to_other_tasks(self):
        source = self.write_json("json/fine_grained_action.json", [self.row("example.avi")])
        mp4 = self.root / "Moments_in_Time_Raw/videos/example.mp4"
        mp4.parent.mkdir(parents=True)
        mp4.touch()
        sample, = load_mvbench(self.root, source)
        self.assertEqual(sample.media_path, str(self.root / "video/Moments_in_Time_Raw/videos/example.avi"))


if __name__ == "__main__":
    unittest.main()
