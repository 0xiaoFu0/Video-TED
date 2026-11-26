"""End-to-end CPU checks for deployment, durable output and worker failures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from general_video_eval.loaders.mvbench import TASKS
from general_video_eval.run import build_prompt, make_config, parser, read_predictions, run_workers
from general_video_eval.schema import Sample
from general_video_eval.scoring import summarize


PROJECT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "outputs"
        self.data = {b: self.root / b for b in ("mvbench", "videomme", "tempcompass")}
        for directory in self.data.values():
            directory.mkdir()
        for task, (stem, prefix, kind, bounded) in TASKS.items():
            media = self.data["mvbench"] / "video" / prefix / ("example" if kind == "frames" else "example.mp4")
            if kind == "frames":
                media.mkdir(parents=True)
                (media / "00001.jpg").touch()
            else:
                media.parent.mkdir(parents=True, exist_ok=True)
                media.touch()
            annotation = self.data["mvbench"] / "json" / (stem + ".json")
            annotation.parent.mkdir(exist_ok=True)
            row = {"video": media.name, "question": "What happens?", "candidates": ["Walk", "Run"],
                   "answer": "Run" if task == "Action Sequence" else "Walk"}
            if bounded:
                row.update(start=0, end=1)
            annotation.write_text(json.dumps([row]))
        vrows = []
        for index, task in enumerate(("Action Recognition", "Action Reasoning")):
            video = self.data["videomme"] / "data" / f"v{index}.mp4"
            video.parent.mkdir(exist_ok=True)
            video.touch()
            vrows.append({"question_id": str(index), "videoID": f"v{index}", "question": "What happens?",
                          "options": ["A. Walk", "B. Run"], "answer": "A" if index == 0 else "B",
                          "task_type": task, "duration": "short", "domain": "Life Record"})
        (self.data["videomme"] / "test.json").write_text(json.dumps(vrows))
        trows = []
        for index, dimension in enumerate(("action", "direction", "speed", "order", "attribute_change")):
            video = self.data["tempcompass"] / "videos" / f"t{index}.mp4"
            video.parent.mkdir(exist_ok=True)
            video.touch()
            trows.append({"question_id": str(index), "video_id": f"t{index}", "dim": dimension,
                          "question": "What happens?\nA. Walk\nB. Run",
                          "answer": "A. Walk" if index == 0 else "B. Run"})
        (self.data["tempcompass"] / "multi-choice.json").write_text(json.dumps(trows))
        self.env = dict(os.environ)
        self.env.update(BACKEND="mock", MODEL_PATH="", BENCHMARKS="mvbench,videomme,tempcompass",
                        EXP_NAME="integration", OUTPUT_DIR=str(self.output), PYTHON_BIN=sys.executable,
                        ALLOW_NONSTANDARD_COUNTS="1", CUDA_VISIBLE_DEVICES="0,1", NUM_GPUS="2", TP_SIZE="1",
                        THINK_MODE="1", N="1", TEMPERATURE="0", VOTE="majority_voting", LIMIT="0",
                        PREFLIGHT="0", SCORE_ONLY="0", RESUME="1", WORKER_TIMEOUT_S="15")
        for name, directory in self.data.items():
            self.env[name.upper() + "_DATA_DIR"] = str(directory)
            self.env.pop(name.upper() + "_ANNO_FILE", None)

    def invoke(self, changes=None):
        env = dict(self.env)
        env.update(changes or {})
        return subprocess.run(["bash", "eval/scripts/eval_all.sh"], cwd=PROJECT, env=env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)

    @property
    def run_dir(self):
        return self.output / "integration"

    def test_three_benchmark_shell_resume_and_score_only(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout)
        report = json.loads((self.run_dir / "summary.json").read_text())
        self.assertTrue(report["complete"])
        self.assertTrue(report["is_mock"])
        self.assertEqual(report["n_total"], 27)
        stats = report["benchmarks"]
        self.assertEqual(stats["mvbench"]["avg"], 95)
        self.assertEqual(stats["mvbench"]["fa"], 100)
        self.assertEqual(stats["videomme"]["overall"], 50)
        self.assertEqual(stats["videomme"]["action_recognition"], 100)
        self.assertEqual(stats["tempcompass"]["overall"], 20)
        self.assertEqual(stats["tempcompass"]["action_recognition"], 100)
        self.assertTrue(all(scope["scope"] == "partial_or_nonstandard" for scope in report["scope"].values()))
        original = {p.name: p.read_bytes() for p in (self.run_dir / "predictions").glob("*.jsonl")}
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("27 saved, 0 remaining", result.stdout)
        self.assertEqual(original, {p.name: p.read_bytes() for p in (self.run_dir / "predictions").glob("*.jsonl")})
        result = self.invoke({"THINK_MODE": "0"})
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("Resume refused", result.stdout)
        # Re-scoring uses the saved expected set and predictions, with no source
        # datasets or model imports. Moving them proves this separation.
        for path in self.data.values():
            path.rename(path.with_name(path.name + "-offline"))
        result = self.invoke({"SCORE_ONLY": "1", "CUDA_VISIBLE_DEVICES": "-1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("MOCK OUTPUT", result.stdout)

    def test_interrupted_jsonl_tail_is_repaired_but_complete_corruption_fails(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout)
        path = next((self.run_dir / "predictions").glob("*.jsonl"))
        original = path.read_bytes()
        path.write_bytes(original + b'{"uid":')
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Recovered interrupted final JSONL line", result.stdout)
        self.assertEqual(path.read_bytes(), original)
        path.write_bytes(original + b'bad complete line\n')
        result = self.invoke()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("Corrupt prediction JSONL", result.stdout)

    def test_missing_media_fails_preflight_without_creating_results(self):
        (self.data["videomme"] / "data" / "v0.mp4").unlink()
        result = self.invoke({"PREFLIGHT": "1", "BACKEND": "vllm", "CUDA_VISIBLE_DEVICES": "-1"})
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("Media preflight failed before GPU initialization", result.stdout)
        self.assertFalse(self.output.exists())

    def test_custom_counts_cannot_silently_become_full_results(self):
        result = self.invoke({"ALLOW_NONSTANDARD_COUNTS": "0", "PREFLIGHT": "1"})
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("expected 4000", result.stdout)

    def test_prompt_contains_neither_label_metadata_nor_source_path(self):
        sample = Sample("test", "tempcompass", "What happens?", ("Walk", "Run"), "B",
                        "/sensitive/path/answer-B.mp4", metadata={"answer": "B", "secret": "leak"})
        prompt = build_prompt(sample, True)
        self.assertIn("(B) Run", prompt)
        self.assertNotIn(sample.media_path, prompt)
        self.assertNotIn("secret", prompt)
        self.assertNotIn("The answer is B", prompt)


class WorkerFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "logs").mkdir()
        (self.root / "predictions").mkdir()
        self.old_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.addCleanup(self.restore_devices)
        self.args = argparse.Namespace(num_gpus=1, tensor_parallel_size=1, worker_timeout=10)
        self.config = {"backend": "mock", "think_mode": True, "n": 1,
                       "max_frames_by_benchmark": {"tempcompass": 32}}

    def restore_devices(self):
        if self.old_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.old_devices

    def sample(self, uid, **metadata):
        return Sample(uid, "tempcompass", "What happens?", ("Walk", "Run"), "A", "mock.mp4",
                      metadata={"dim": "action", **metadata})

    def test_worker_exception_preserves_prior_predictions_and_can_resume(self):
        first = self.sample("good")
        second = self.sample("failed", mock_error="intentional decoder error")
        errors = run_workers(self.args, self.config, [first, second], self.root)
        self.assertTrue(errors)
        records = read_predictions(self.root)
        self.assertEqual([row["uid"] for row in records], ["good"])
        partial = summarize([first, second], records, errors)
        self.assertEqual((partial["n_total"], partial["n_done"], partial["overall"]), (2, 1, 50))
        self.assertFalse(partial["complete"])
        # Resume appends only the remaining ID; infrastructure failures never
        # produce an answer record and cannot remove it from the denominator.
        errors = run_workers(self.args, self.config, [self.sample("failed")], self.root)
        self.assertFalse(errors)
        final = summarize([first, second], read_predictions(self.root), errors)
        self.assertTrue(final["complete"])
        self.assertEqual(final["overall"], 100)

    def test_worker_timeout_terminates_promptly(self):
        self.args.worker_timeout = 1
        started = time.monotonic()
        errors = run_workers(self.args, self.config, [self.sample("slow", mock_sleep_s=30)], self.root)
        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue(errors)
        self.assertIn("No worker progress", errors[0]["error"])
        self.assertEqual(read_predictions(self.root), [])


if __name__ == "__main__":
    unittest.main()
