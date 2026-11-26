"""Run the real paired launcher against isolated CPU-only entrypoint stubs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "eval/scripts/eval_pair_videor1_64.sh"
CAPTURE_KEYS = (
    "MODEL_PATH", "EXP_NAME", "LIMIT", "BACKEND", "BENCHMARKS", "OUTPUT_DIR",
    "EVAL_PROFILE", "THINK_MODE", "N", "TEMPERATURE", "TOP_P", "REPETITION_PENALTY",
    "MAX_NEW_TOKENS", "MAX_MODEL_LEN", "FPS", "MIN_FRAMES", "MIN_PIXELS", "MAX_PIXELS",
    "MAX_FRAMES", "MVBENCH_MAX_FRAMES", "VIDEOMME_MAX_FRAMES", "TEMPCOMPASS_MAX_FRAMES",
    "VOTE", "SEED", "DTYPE", "DECODER", "PREFLIGHT", "SCORE_ONLY", "RESUME",
)


@unittest.skipUnless(shutil.which("bash"), "bash is required for the server launcher")
class PairLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "isolated project"
        scripts = self.project / "eval/scripts"
        scripts.mkdir(parents=True)
        self.launcher = scripts / "eval_pair_videor1_64.sh"
        shutil.copyfile(LAUNCHER, self.launcher)
        self.trace = self.root / "trace.jsonl"
        # Capture the environment seen by each actual child shell invocation.
        # Fail only after capture so tests can verify the exact stopping point.
        stub = '''\
#!/usr/bin/env bash
set -Eeuo pipefail
"${PYTHON_BIN:-python3}" - <<'PY'
import json
import os
from pathlib import Path
keys = CAPTURE_KEYS_PLACEHOLDER
record = {key: os.environ.get(key) for key in keys}
record["event"] = "eval"
with Path(os.environ["STUB_CAPTURE_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\\n")
suffix = os.environ.get("STUB_FAIL_SUFFIX")
if suffix and os.environ["EXP_NAME"].endswith(suffix):
    raise SystemExit(17)
total = int(os.environ["LIMIT"]) or 10
names = [name.strip() for name in os.environ["BENCHMARKS"].split(",") if name.strip()]
invalid_suffix = os.environ.get("STUB_ALL_INVALID_SUFFIX")
all_invalid = bool(invalid_suffix and os.environ["EXP_NAME"].endswith(invalid_suffix))
benchmarks = {name: {"n_total": total, "n_done": total,
                     "invalid_answers": total if all_invalid and name == names[-1] else 0,
                     "overall": 0, "n_correct": 0} for name in names}
destination = Path(os.environ["OUTPUT_DIR"]) / os.environ["EXP_NAME"]
destination.mkdir(parents=True, exist_ok=True)
(destination / "summary.json").write_text(json.dumps({"complete": True, "benchmarks": benchmarks}), encoding="utf-8")
PY
'''.replace("CAPTURE_KEYS_PLACEHOLDER", repr(CAPTURE_KEYS))
        (scripts / "eval_all.sh").write_text(stub, encoding="utf-8")
        package = self.project / "general_video_eval"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "compare.py").write_text(textwrap.dedent('''
            import json
            import os
            from pathlib import Path
            import sys
            with Path(os.environ["STUB_CAPTURE_PATH"]).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "compare", "argv": sys.argv[1:]}) + "\\n")
        '''), encoding="utf-8")
        self.models = {}
        for name in ("base", "ours"):
            directory = self.root / f"{name} checkpoint"
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}), encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"nonempty fixture weights")
            self.models[name] = directory
        self.output = self.root / "results"
        self.environment = dict(os.environ)
        self.environment.update({
            "PYTHON_BIN": sys.executable, "PYTHONPATH": str(self.project),
            "BASE_MODEL_PATH": str(self.models["base"]), "OURS_MODEL_PATH": str(self.models["ours"]),
            "EXP_PREFIX": "paired", "OUTPUT_DIR": str(self.output), "BACKEND": "vllm",
            "BENCHMARKS": "mvbench,videomme", "CUDA_VISIBLE_DEVICES": "0", "NUM_GPUS": "1", "TP_SIZE": "1",
            "STUB_CAPTURE_PATH": str(self.trace),
            # Simulate inherited settings from a previous generic/scoring run.
            "MODEL_PATH": "/stale/model", "EVAL_PROFILE": "generic", "THINK_MODE": "0", "N": "8",
            "TEMPERATURE": "0", "TOP_P": "1", "REPETITION_PENALTY": "1.9",
            "MAX_NEW_TOKENS": "2048", "MAX_MODEL_LEN": "32768", "FPS": "1", "MIN_FRAMES": "2",
            "MIN_PIXELS": "3136", "MAX_PIXELS": "262144", "MAX_FRAMES": "32",
            "MVBENCH_MAX_FRAMES": "16", "VIDEOMME_MAX_FRAMES": "32", "TEMPCOMPASS_MAX_FRAMES": "16",
            "VOTE": "confidence_weighting", "SEED": "999", "DTYPE": "float16", "DECODER": "opencv",
            "LIMIT": "99", "PREFLIGHT": "1", "SCORE_ONLY": "1", "RESUME": "0",
        })
        self.environment.pop("STUB_FAIL_SUFFIX", None)
        self.environment.pop("STUB_ALL_INVALID_SUFFIX", None)

    def invoke(self, changes=None):
        environment = dict(self.environment)
        environment.update(changes or {})
        # Invoke from outside the project to verify the launcher's own cwd logic.
        return subprocess.run(["bash", str(self.launcher)], cwd=self.root, env=environment,
                              capture_output=True, text=True, timeout=15)

    def records(self):
        if not self.trace.exists():
            return []
        return [json.loads(line) for line in self.trace.read_text().splitlines() if line.strip()]

    def test_run_order_fixed_settings_and_full_comparison_paths(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.records()
        self.assertEqual([record["event"] for record in records], ["eval"] * 4 + ["compare"])
        runs = records[:4]
        self.assertEqual([record["EXP_NAME"] for record in runs],
                         ["paired_base_smoke", "paired_ours_smoke", "paired_base_full", "paired_ours_full"])
        self.assertEqual([record["LIMIT"] for record in runs], ["5", "5", "0", "0"])
        self.assertEqual([record["MODEL_PATH"] for record in runs],
                         [str(self.models[name]) for name in ("base", "ours", "base", "ours")])
        expected = {
            "EVAL_PROFILE": "videor1_64", "THINK_MODE": "1", "N": "1", "TEMPERATURE": "0.01", "TOP_P": "0.001",
            "REPETITION_PENALTY": "1.0", "MAX_NEW_TOKENS": "1024", "MAX_MODEL_LEN": "16384",
            "FPS": "2", "MIN_FRAMES": "4", "MIN_PIXELS": "100352", "MAX_PIXELS": "200704",
            "MAX_FRAMES": "64", "MVBENCH_MAX_FRAMES": "64", "VIDEOMME_MAX_FRAMES": "64", "TEMPCOMPASS_MAX_FRAMES": "64",
            "VOTE": "majority_voting", "SEED": "42", "DTYPE": "bfloat16", "DECODER": "decord",
            "PREFLIGHT": "0", "SCORE_ONLY": "0", "RESUME": "1",
        }
        for index, record in enumerate(runs):
            for key, value in expected.items():
                with self.subTest(run=index, setting=key):
                    self.assertEqual(record[key], value)
        common = [{key: value for key, value in record.items() if key not in {"MODEL_PATH", "EXP_NAME", "LIMIT"}}
                  for record in runs]
        self.assertTrue(all(record == common[0] for record in common))
        self.assertEqual(records[-1]["argv"], [str(self.output / "paired_base_full"), str(self.output / "paired_ours_full")])
        # Zero accuracy is allowed when outputs are parseable; smoke validates
        # the inference/answer contract, not whether this tiny sample is correct.
        for name in ("paired_base_smoke", "paired_ours_smoke"):
            summary = json.loads((self.output / name / "summary.json").read_text())
            self.assertTrue(all(bucket["overall"] == 0 and bucket["invalid_answers"] == 0
                                for bucket in summary["benchmarks"].values()))

    def test_either_smoke_failure_stops_before_any_full_run_or_comparison(self):
        for suffix, expected_names in (("base_smoke", ["paired_base_smoke"]),
                                       ("ours_smoke", ["paired_base_smoke", "paired_ours_smoke"])):
            with self.subTest(failed=suffix):
                if self.trace.exists():
                    self.trace.unlink()
                result = self.invoke({"STUB_FAIL_SUFFIX": suffix})
                self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
                records = self.records()
                self.assertEqual([record["EXP_NAME"] for record in records], expected_names)
                self.assertTrue(all(record["event"] == "eval" and record["LIMIT"] == "5" for record in records))

    def test_full_failure_prevents_comparison(self):
        result = self.invoke({"STUB_FAIL_SUFFIX": "base_full"})
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertEqual([record["EXP_NAME"] for record in self.records()],
                         ["paired_base_smoke", "paired_ours_smoke", "paired_base_full"])

    def test_all_invalid_answers_in_one_benchmark_stop_before_full_or_comparison(self):
        for suffix, expected_names in (("base_smoke", ["paired_base_smoke"]),
                                       ("ours_smoke", ["paired_base_smoke", "paired_ours_smoke"])):
            with self.subTest(unparseable=suffix):
                if self.trace.exists():
                    self.trace.unlink()
                result = self.invoke({"STUB_ALL_INVALID_SUFFIX": suffix})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("entirely unparseable", result.stdout + result.stderr)
                self.assertEqual([record["EXP_NAME"] for record in self.records()], expected_names)
                report = json.loads((self.output / ("paired_" + suffix) / "summary.json").read_text())
                self.assertTrue(report["complete"])
                self.assertEqual(report["benchmarks"]["mvbench"]["invalid_answers"], 0)
                self.assertEqual(report["benchmarks"]["videomme"]["invalid_answers"], 5)

    def test_mock_backend_explicitly_enables_mock_comparison(self):
        result = self.invoke({"BACKEND": "mock"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.records()[-1]["argv"],
                         [str(self.output / "paired_base_full"), str(self.output / "paired_ours_full"), "--allow-mock"])

    def test_missing_base_weights_fail_before_any_evaluation(self):
        (self.models["base"] / "model.safetensors").unlink()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing exported base", result.stdout + result.stderr)
        self.assertEqual(self.records(), [])

    def test_missing_ours_weights_fail_before_any_evaluation(self):
        (self.models["ours"] / "model.safetensors").unlink()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing exported ours", result.stdout + result.stderr)
        self.assertEqual(self.records(), [])

    def test_empty_weights_and_missing_config_are_rejected(self):
        weights = self.models["ours"] / "model.safetensors"
        weights.write_bytes(b"")
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.records(), [])
        weights.write_bytes(b"nonempty fixture weights")
        (self.models["ours"] / "config.json").unlink()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.records(), [])

    def test_non_qwen25_checkpoint_is_rejected_before_any_evaluation(self):
        (self.models["ours"] / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8")
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Qwen2.5-VL", result.stdout + result.stderr)
        self.assertEqual(self.records(), [])


if __name__ == "__main__":
    unittest.main()
