"""Short-prompt / 32-frame variant retains the previous sampling and budgets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from general_video_eval import run
from general_video_eval.media import decode_media, frame_plan
from general_video_eval.protocol import build_prompt
from general_video_eval.schema import Sample


PROJECT = Path(__file__).resolve().parents[1]
BENCHMARKS = ["mvbench", "videomme", "tempcompass"]


class Simple32Tests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def config(self, profile="simple_32", extra=(), benchmarks=None):
        args = run.parser().parse_args(["--eval-profile", profile, *extra])
        return run.make_config(args, benchmarks or BENCHMARKS)

    def test_only_prompt_frame_budget_and_variant_labels_differ(self):
        old = self.config("videor1_64")
        new = self.config()
        changed = {key for key in old if old[key] != new[key]}
        self.assertEqual(changed, {"eval_profile", "prompt_style", "protocol_reference", "protocol",
                                   "max_frames", "max_frames_by_benchmark"})
        self.assertEqual(new["video_sampling"], "uniform_fps_floor")
        self.assertEqual(new["max_frames_by_benchmark"], dict.fromkeys(BENCHMARKS, 32))
        sample = Sample("test", "videomme", "What happens?", ("Walking", "Running"), "B", "unused.mp4")
        prompt = build_prompt(sample, new["think_mode"], new["prompt_style"])
        self.assertEqual(prompt, "Select the best answer based on the video. Reason briefly using only visible video evidence "
                         "inside <think>...</think>, then output the option letter inside <answer>...</answer>.\n"
                         "Question: What happens?\nOptions:\n(A) Walking\n(B) Running")

    def test_conflicting_settings_and_seed_are_rejected(self):
        for key, value in {"MAX_FRAMES": "64", "MVBENCH_MAX_FRAMES": "64", "THINK_MODE": "0",
                           "FPS": "1", "MAX_NEW_TOKENS": "8192", "TOP_P": "1"}.items():
            with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                with self.assertRaisesRegex(ValueError, "simple_32 requires"):
                    self.config()
        with self.assertRaisesRegex(ValueError, "eight pre-extracted frames"):
            self.config(benchmarks=["seedbench"])

    def test_short_clip_keeps_floor_sampling_long_clip_is_capped_not_cropped(self):
        config = self.config()
        for total, fps, bounds, count in ((114, 10, {}, 22), (1000, 10, {"start": 20, "end": 60}, 32)):
            with self.subTest(total=total):
                plan = frame_plan(total, fps, max_frames=config["max_frames"],
                                  target_fps=config["fps"], min_frames=config["min_frames"],
                                  sampling=config["video_sampling"], **bounds)
                self.assertEqual(plan["sampled_frames"], count)
                if bounds:
                    self.assertEqual((plan["sampled_frame_indices"][0], plan["sampled_frame_indices"][-1]), (200, 600))
                else:
                    old = frame_plan(total, fps, max_frames=64, sampling="uniform_fps_floor")
                    self.assertEqual(plan["sampled_frame_indices"], old["sampled_frame_indices"])

    def test_mvbench_frame_directory_keeps_annotated_bounds_with_new_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index in range(1, 121):
                Image.new("RGB", (56, 56), (index, 0, 0)).save(Path(temporary) / f"{index:05d}.png")
            sample = Sample("test", "mvbench", "Which?", ("A", "B"), "B", temporary,
                            media_type="frames", frame_fps=3, start=5, end=30)
            config = self.config()
            video, info = decode_media(sample, max_frames=32, sampling=config["video_sampling"],
                                       min_pixels=3136, max_pixels=3136)
            self.assertEqual(video.shape[0], 32)
            self.assertEqual((info["sampled_frame_indices"][0], info["sampled_frame_indices"][-1]), (14, 89))
            self.assertEqual([int(value) for value in video[:, 0, 0, 0]],
                             [i + 1 for i in info["sampled_frame_indices"]])

    def test_prepared_dataset_metadata_records_actual_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "data" / "clip.mp4").touch()
            (root / "test.json").write_text(json.dumps([{
                "question_id": "one", "videoID": "clip", "question": "What happens?",
                "options": ["A. Walking", "B. Running"], "answer": "B",
                "duration": "short", "task_type": "Action Recognition",
            }]))
            args = run.parser().parse_args(["--eval-profile", "simple_32", "--videomme-data-dir", str(root),
                                           "--allow-nonstandard-counts"])
            run.make_config(args, ["videomme"])
            samples, _ = run.prepare_samples(args, ["videomme"])
            self.assertEqual(samples[0].metadata["sampling"], "uniform_fps_floor")
            self.assertEqual(samples[0].metadata["eval_profile"], "simple_32")

    def test_launcher_resets_inherited_protocol_and_partial_run_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / "capture-python"
            runner.write_text(f"#!{sys.executable}\n" +
                              "import json, os\nfrom general_video_eval import run\n" +
                              "a = run.parser().parse_args([])\n" +
                              "print(json.dumps({'config': run.make_config(a, a.benchmarks.split(',')), " +
                              "'flags': [a.preflight, a.score_only, a.limit, a.allow_nonstandard_counts, a.verify_frame_images, a.no_resume], " +
                              "'exp': a.exp_name, 'gpus': a.num_gpus}))\n")
            runner.chmod(0o755)
            env = {"PATH": os.defpath, "PYTHON_BIN": str(runner), "MODEL_PATH": "test-model",
                   "EXP_NAME": "videor1_simple32_test_full", "EVAL_PROFILE": "seedbench_8",
                   "BENCHMARKS": "seedbench", "THINK_MODE": "0", "TEMPERATURE": "1",
                   "MAX_FRAMES": "8", "VIDEOMME_MAX_FRAMES": "64", "MAX_NEW_TOKENS": "8192",
                   "PREFLIGHT": "1", "SCORE_ONLY": "1", "LIMIT": "5", "RESUME": "0",
                   "ALLOW_NONSTANDARD_COUNTS": "1", "VERIFY_FRAME_IMAGES": "1"}
            result = subprocess.run(["bash", "eval/scripts/eval_simple32.sh"], cwd=PROJECT,
                                    env=env, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = json.loads(result.stdout)
            self.assertEqual(actual["config"]["benchmarks"], BENCHMARKS)
            self.assertEqual(actual["config"]["prompt_style"], "generic")
            self.assertEqual(actual["config"]["video_sampling"], "uniform_fps_floor")
            self.assertEqual(actual["config"]["max_frames_by_benchmark"], dict.fromkeys(BENCHMARKS, 32))
            self.assertEqual(actual["config"]["max_new_tokens"], 1024)
            self.assertEqual(actual["flags"], [False, False, 0, False, False, False])
            self.assertEqual(actual["exp"], "videor1_simple32_test_full")
            self.assertEqual(actual["gpus"], 8)


if __name__ == "__main__":
    unittest.main()
