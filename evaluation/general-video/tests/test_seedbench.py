"""Question-to-frame mapping, strict preflight and portable SEED launch checks."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from PIL import Image

from general_video_eval import run
from general_video_eval.loaders.seedbench import load_seedbench, check_frame_coverage
from general_video_eval.media import decode_media

PROJECT = Path(__file__).resolve().parents[1]


class SeedBenchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.rows = []
        # Shared source ID with different segments must not merge questions.
        for qid, task, segment in (("v1", 10, None), ("v1743", 11, [1, 2]),
                                   ("v1744", 11, [3, 4]), ("v3817", 12, [15, 60])):
            self.rows.append(dict(question_id=qid, question_type_id=task,
                data_type="video", data_id="same_source.mp4", segment=segment,
                question="What happens?", choice_a="Walk", choice_b="Run",
                choice_c="Sit", choice_d="Jump", answer="A" if task == 10 else "B"))
        self.rows.append(dict(question_id="image1", question_type_id=1, data_type="image"))
        self.write()
        # Fixtures independently specify official paths, not production TASKS.
        directories = ("task10/ssv2_8_frame/v1", "task11/kitchen_8_frame/v1743",
                       "task11/kitchen_8_frame/v1744", "task12/breakfast_8_frame/v3817")
        for directory in directories:
            folder = self.root / "v1_video" / directory
            folder.mkdir(parents=True)
            for number in range(1, 9):
                Image.new("RGB", (28, 28), (number * 20, 0, 0)).save(folder / f"{number}.png")
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def write(self):
        (self.root / "SEED-Bench.json").write_text(json.dumps({"questions": self.rows}))

    def args(self, *extra):
        return run.parser().parse_args(["--benchmarks", "seedbench", "--eval-profile", "seedbench_8",
            "--seedbench-data-dir", str(self.root), "--allow-nonstandard-counts", *extra])

    def test_video_only_question_keyed_mapping_and_all_eight_pngs(self):
        samples = load_seedbench(self.root)
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0].metadata["task"], "Action Recognition")
        self.assertEqual(Path(samples[1].media_path).relative_to(self.root).as_posix(),
                         "v1_video/task11/kitchen_8_frame/v1743")
        self.assertNotEqual(samples[1].media_path, samples[2].media_path)
        self.assertEqual(samples[1].metadata["data_id"], samples[2].metadata["data_id"])
        self.assertIsNone(samples[1].start)
        self.assertIsNone(samples[1].end)
        with redirect_stdout(io.StringIO()):
            check_frame_coverage(samples, decode=True)
        video, info = decode_media(samples[3], max_frames=8)
        self.assertEqual(video.shape[0], 8)
        self.assertEqual(video[:, 0, 0, 0].tolist(), list(range(20, 161, 20)))
        self.assertEqual(info["sampling"], "seedbench_official_8")

    def test_invalid_annotations_are_rejected(self):
        for fields in ({"question_id": "../v1"}, {"data_type": "image"},
                       {"answer": "Z"}, {"choice_d": ""}, {"question_type_id": 10.5}):
            original = dict(self.rows[0])
            with self.subTest(fields=fields):
                self.rows[0].update(fields)
                self.write()
                with self.assertRaises(ValueError):
                    load_seedbench(self.root)
                self.rows[0] = original
        self.rows.append(dict(self.rows[0]))
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_seedbench(self.root)

    def test_missing_and_corrupt_frame_fail_preflight_before_gpu(self):
        path = self.root / "v1_video/task11/kitchen_8_frame/v1744/8.png"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "before GPU initialization"):
            run.prepare_samples(self.args(), ["seedbench"])
        path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "bad images"), redirect_stdout(io.StringIO()):
            run.prepare_samples(self.args("--verify-frame-images"), ["seedbench"])

    def test_smoke_covers_three_tasks_and_named_profile_is_fixed(self):
        args = self.args("--limit", "3")
        config = run.make_config(args, ["seedbench"])
        with redirect_stdout(io.StringIO()):
            samples, audit = run.prepare_samples(args, ["seedbench"])
        self.assertEqual([s.metadata["question_type_id"] for s in samples], [10, 11, 12])
        self.assertEqual(audit["seedbench"]["scope"], "partial_or_nonstandard")
        self.assertEqual(config["prompt_style"], "videor1")
        self.assertEqual(config["max_frames_by_benchmark"], {"seedbench": 8})
        self.assertEqual(config["max_new_tokens"], 1024)
        self.assertFalse(config["seedbench_input"]["source_timing_known"])
        for overrides in (("--seedbench-max-frames", "64"), ("--max-new-tokens", "8192")):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                run.make_config(self.args(*overrides), ["seedbench"])
        with self.assertRaises(ValueError):
            run.make_config(self.args(), ["seedbench", "mvbench"])
        with self.assertRaises(ValueError):
            run.make_config(self.args("--eval-profile", "videor1_64"), ["seedbench"])

    def test_nonstandard_counts_fail_without_explicit_override(self):
        args = self.args()
        args.allow_nonstandard_counts = False
        with self.assertRaisesRegex(ValueError, "expected 3757"):
            run.prepare_samples(args, ["seedbench"])

    def test_shell_preflight_smoke_full_resume_and_offline_rescore(self):
        env = dict(os.environ, PYTHON_BIN=sys.executable,
            PYTHONPATH=os.pathsep.join(sys.path), MODEL_PATH="mock-model",
            BACKEND="mock", SEEDBENCH_DATA_DIR=str(self.root),
            OUTPUT_DIR=str(self.root / "outputs"), EXP_NAME="seed_test",
            ALLOW_NONSTANDARD_COUNTS="1", CUDA_VISIBLE_DEVICES="0,1", NUM_GPUS="2",
            MAX_FRAMES="64", MAX_NEW_TOKENS="8192", EVAL_PROFILE="videor1_64")
        command = ["bash", "eval/scripts/eval_seedbench.sh", "run"]
        first = subprocess.run(command, cwd=PROJECT, env=env, text=True, capture_output=True, timeout=40)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        directory = self.root / "outputs/seed_test_full"
        report = json.loads((directory / "summary.json").read_text())
        self.assertTrue(report["complete"])
        self.assertTrue(report["is_mock"])
        self.assertEqual(report["benchmarks"]["seedbench"]["overall"], 25.0)
        self.assertEqual(report["benchmarks"]["seedbench"]["action_recognition"], 100.0)
        self.assertIn("SEED-Bench Video", (directory / "summary.md").read_text())
        predictions = {p: p.read_bytes() for p in (directory / "predictions").glob("*.jsonl")}
        second = subprocess.run(command, cwd=PROJECT, env=env, text=True, capture_output=True, timeout=40)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("4 saved, 0 remaining", second.stdout)
        self.assertEqual(predictions, {p: p.read_bytes() for p in predictions})
        (self.root / "v1_video").rename(self.root / "offline_frames")
        env.update(SCORE_ONLY="1", EXP_NAME="seed_test_full")
        rescore = subprocess.run(["bash", "eval/scripts/eval_all.sh"], cwd=PROJECT, env=env,
                                 text=True, capture_output=True, timeout=10)
        self.assertEqual(rescore.returncode, 0, rescore.stdout + rescore.stderr)


if __name__ == "__main__":
    unittest.main()
