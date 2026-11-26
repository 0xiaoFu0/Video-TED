"""Same short prompt across source-video 64-frame and official SEED 8-frame runs."""
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
from general_video_eval.media import decode_media
from general_video_eval.protocol import build_prompt
from general_video_eval.schema import Sample


PROJECT = Path(__file__).resolve().parents[1]
THREE_BENCHMARKS = ["mvbench", "videomme", "tempcompass"]
SHORT_INSTRUCTION = (
    "Select the best answer based on the video. Reason briefly using only visible video evidence "
    "inside <think>...</think>, then output the option letter inside <answer>...</answer>."
)


class Simple64Tests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def config(self, profile="simple_64", extra=(), benchmarks=None):
        selected = benchmarks if benchmarks is not None else THREE_BENCHMARKS
        args = run.parser().parse_args(["--eval-profile", profile, *extra])
        return run.make_config(args, selected)

    def test_each_new_profile_changes_only_prompt_and_protocol_labels(self):
        for new_profile, old_profile, benchmarks, frames, sampling in (
            ("simple_64", "videor1_64", THREE_BENCHMARKS, 64, "uniform_fps_floor"),
            ("seedbench_simple_8", "seedbench_8", ["seedbench"], 8, "seedbench_official_8"),
        ):
            with self.subTest(profile=new_profile):
                old = self.config(old_profile, benchmarks=benchmarks)
                new = self.config(new_profile, benchmarks=benchmarks)
                self.assertEqual(set(old), set(new))
                changed = {key for key in old if old[key] != new[key]}
                self.assertEqual(changed, {"eval_profile", "prompt_style", "protocol_reference", "protocol"})
                self.assertEqual(new["prompt_style"], "generic")
                self.assertTrue(new["think_mode"])
                self.assertEqual(new["video_sampling"], sampling)
                self.assertEqual(new["max_frames_by_benchmark"], dict.fromkeys(benchmarks, frames))
        previous = self.config("simple_32")
        current = self.config()
        self.assertEqual({key for key in previous if previous[key] != current[key]},
                         {"eval_profile", "protocol_reference", "max_frames", "max_frames_by_benchmark"})

    def test_all_four_benchmarks_use_the_exact_same_short_prompt(self):
        expected = SHORT_INSTRUCTION + "\nQuestion: What happens?\nOptions:\n(A) Walking\n(B) Running"
        for benchmark in [*THREE_BENCHMARKS, "seedbench"]:
            with self.subTest(benchmark=benchmark):
                profile = "seedbench_simple_8" if benchmark == "seedbench" else "simple_64"
                config = self.config(profile, benchmarks=[benchmark])
                sample = Sample("secret-id", benchmark, "What happens?", ("Walking", "Running"),
                                "B", "/secret/ground-truth.mp4", metadata={"answer": "secret-label"})
                prompt = build_prompt(sample, config["think_mode"], config["prompt_style"])
                self.assertEqual(prompt, expected)
                self.assertEqual(prompt, build_prompt(sample, True, "generic"))
                self.assertNotIn("secret", prompt)

    def test_wrong_frame_budgets_generation_settings_and_benchmark_combinations_fail(self):
        cases = (
            ("simple_64", THREE_BENCHMARKS, {"MAX_FRAMES": "32", "VIDEOMME_MAX_FRAMES": "32"}),
            ("seedbench_simple_8", ["seedbench"], {"MAX_FRAMES": "64", "SEEDBENCH_MAX_FRAMES": "64"}),
        )
        for profile, benchmarks, frame_overrides in cases:
            overrides = {**frame_overrides, "THINK_MODE": "0", "FPS": "1", "TOP_P": "1",
                         "MAX_NEW_TOKENS": "8192", "MAX_MODEL_LEN": "32768"}
            for key, value in overrides.items():
                with self.subTest(profile=profile, key=key), patch.dict(os.environ, {key: value}):
                    with self.assertRaisesRegex(ValueError, profile + " requires"):
                        self.config(profile, benchmarks=benchmarks)
        for profile, benchmarks in (
            ("simple_64", ["seedbench"]),
            ("simple_64", [*THREE_BENCHMARKS, "seedbench"]),
            ("seedbench_simple_8", THREE_BENCHMARKS),
            ("seedbench_simple_8", ["seedbench", "mvbench"]),
        ):
            with self.subTest(profile=profile, benchmarks=benchmarks), self.assertRaises(ValueError):
                self.config(profile, benchmarks=benchmarks)
        with self.assertRaisesRegex(ValueError, "simple_64 requires"):
            self.config(extra=["--mvbench-max-frames", "32"])
        with self.assertRaisesRegex(ValueError, "seedbench_simple_8 requires"):
            self.config("seedbench_simple_8", extra=["--seedbench-max-frames", "32"], benchmarks=["seedbench"])

    def test_actual_frame_decode_preserves_short_clip_and_caps_long_clip_at_64(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as temporary:
            for number in range(1, 211):
                Image.new("RGB", (56, 56), (number, 0, 0)).save(Path(temporary) / f"{number:05d}.png")
            for end, count, last_index in ((10, 10, 29), (60, 64, 179)):
                with self.subTest(end=end):
                    sample = Sample("clip", "mvbench", "What happens?", ("Walking", "Running"),
                                    "B", temporary, media_type="frames", frame_fps=3, start=5, end=end)
                    video, info = decode_media(sample, max_frames=config["max_frames"],
                                               fps=config["fps"], min_frames=config["min_frames"],
                                               sampling=config["video_sampling"], min_pixels=3136, max_pixels=3136)
                    self.assertEqual(video.shape[0], count)
                    self.assertEqual(info["sampling"], "uniform_fps_floor")
                    self.assertEqual((info["sampled_frame_indices"][0], info["sampled_frame_indices"][-1]),
                                     (14, last_index))
                    self.assertEqual(video[:, 0, 0, 0].tolist(), [i + 1 for i in info["sampled_frame_indices"]])
                    self.assertEqual((info["start"], info["end"]), (5, end))

    def test_prepared_seed_pack_keeps_all_eight_frames_and_synthetic_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "v1_video/task10/ssv2_8_frame/v1"
            folder.mkdir(parents=True)
            for number in range(1, 9):
                Image.new("RGB", (56, 56), (20 * number, 0, 0)).save(folder / f"{number}.png")
            (root / "SEED-Bench.json").write_text(json.dumps({"questions": [{
                "question_id": "v1", "question_type_id": 10, "data_type": "video",
                "data_id": "same-source.mp4", "question": "What happens?",
                "choice_a": "Walk", "choice_b": "Run", "choice_c": "Sit", "choice_d": "Jump", "answer": "A",
            }]}), encoding="utf-8")
            args = run.parser().parse_args(["--eval-profile", "seedbench_simple_8", "--benchmarks", "seedbench",
                                           "--seedbench-data-dir", str(root), "--allow-nonstandard-counts"])
            config = run.make_config(args, ["seedbench"])
            samples, _ = run.prepare_samples(args, ["seedbench"])
            self.assertEqual(samples[0].metadata["sampling"], "seedbench_official_8")
            self.assertIsNone(samples[0].start)
            self.assertIsNone(samples[0].end)
            with patch("general_video_eval.media.frame_plan", side_effect=AssertionError("SEED must not resample")):
                video, info = decode_media(samples[0], max_frames=config["max_frames"],
                                           sampling=config["video_sampling"], min_pixels=3136, max_pixels=3136)
            self.assertEqual(video.shape[0], 8)
            self.assertEqual(video[:, 0, 0, 0].tolist(), list(range(20, 161, 20)))
            self.assertEqual(info["sampling"], "seedbench_official_8")
            self.assertEqual(info["timing_source"], "synthetic_ordinal_2fps")
            self.assertFalse(info["source_timing_known"])
            self.assertFalse(config["seedbench_input"]["resample"])
            self.assertFalse(config["seedbench_input"]["source_timing_known"])

    def test_prepared_source_video_metadata_uses_floor_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "data/clip.mp4").touch()
            (root / "test.json").write_text(json.dumps([{
                "question_id": "one", "videoID": "clip", "question": "What happens?",
                "options": ["A. Walking", "B. Running"], "answer": "B",
                "duration": "short", "task_type": "Action Recognition",
            }]), encoding="utf-8")
            args = run.parser().parse_args(["--eval-profile", "simple_64", "--videomme-data-dir", str(root),
                                           "--allow-nonstandard-counts"])
            run.make_config(args, ["videomme"])
            samples, _ = run.prepare_samples(args, ["videomme"])
            self.assertEqual(samples[0].metadata["sampling"], "uniform_fps_floor")
            self.assertEqual(samples[0].metadata["eval_profile"], "simple_64")

    def capture_launcher(self, root, fail_first=False):
        runner = root / "capture-python"
        capture = root / "calls.jsonl"
        runner.write_text(f"#!{sys.executable}\n" +
                          "import json, os, sys\nfrom general_video_eval import run\n" +
                          "a = run.parser().parse_args([])\n" +
                          "record = {'config': run.make_config(a, a.benchmarks.split(',')), " +
                          "'flags': [a.preflight, a.score_only, a.limit, a.allow_nonstandard_counts, a.verify_frame_images, a.no_resume], " +
                          "'exp': a.exp_name, 'gpus': a.num_gpus}\n" +
                          "with open(os.environ['CAPTURE_PATH'], 'a') as f: f.write(json.dumps(record) + '\\n')\n" +
                          "if os.getenv('FAIL_FIRST') == '1': sys.exit(7)\n", encoding="utf-8")
        runner.chmod(0o755)
        env = {"PATH": os.defpath, "PYTHON_BIN": str(runner), "MODEL_PATH": "test-model",
               "EXP_PREFIX": "treesd7b_test_v122", "EXP_NAME": "stale_name", "CAPTURE_PATH": str(capture),
               "OUTPUT_DIR": str(root / "outputs"), "EVAL_PROFILE": "seedbench_8", "BENCHMARKS": "seedbench",
               "THINK_MODE": "0", "TEMPERATURE": "1", "MAX_FRAMES": "32", "MVBENCH_MAX_FRAMES": "32",
               "VIDEOMME_MAX_FRAMES": "32", "TEMPCOMPASS_MAX_FRAMES": "32", "SEEDBENCH_MAX_FRAMES": "64",
               "MAX_NEW_TOKENS": "8192", "PREFLIGHT": "1", "SCORE_ONLY": "1", "LIMIT": "5", "RESUME": "0",
               "ALLOW_NONSTANDARD_COUNTS": "1", "VERIFY_FRAME_IMAGES": "1", "FAIL_FIRST": str(int(fail_first))}
        result = subprocess.run(["bash", "eval/scripts/eval_simple64_all.sh"], cwd=PROJECT,
                                env=env, text=True, capture_output=True, timeout=20)
        records = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        return result, records

    def test_launcher_runs_three_video_benchmarks_then_seed_with_same_short_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, records = self.capture_launcher(Path(temporary))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(records), 2, result.stdout + result.stderr)
        self.assertEqual([r["exp"] for r in records], ["treesd7b_test_v122_simple64_three_full",
                                                      "treesd7b_test_v122_seedbench_simple8_full"])
        for record, profile, benchmarks, frames, sampling in zip(
            records, ("simple_64", "seedbench_simple_8"), (THREE_BENCHMARKS, ["seedbench"]),
            (64, 8), ("uniform_fps_floor", "seedbench_official_8"),
        ):
            config = record["config"]
            self.assertEqual(config["eval_profile"], profile)
            self.assertEqual(config["benchmarks"], benchmarks)
            self.assertEqual(config["max_frames_by_benchmark"], dict.fromkeys(benchmarks, frames))
            self.assertEqual(config["video_sampling"], sampling)
            self.assertEqual(config["prompt_style"], "generic")
            self.assertTrue(config["think_mode"])
            self.assertEqual(config["max_new_tokens"], 1024)
            self.assertEqual(record["flags"], [False, False, 0, False, False, False])
            self.assertEqual(record["gpus"], 8)

    def test_launcher_stops_if_first_evaluation_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, records = self.capture_launcher(Path(temporary), fail_first=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(records), 1, result.stdout + result.stderr)
        self.assertEqual(records[0]["config"]["eval_profile"], "simple_64")


if __name__ == "__main__":
    unittest.main()
