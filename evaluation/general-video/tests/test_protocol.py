"""Named protocol, prompt fidelity, CLI and resume isolation checks without CUDA."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from general_video_eval import run
from general_video_eval.protocol import build_prompt
from general_video_eval.schema import Sample


PROJECT = Path(__file__).resolve().parents[1]
# Independently retain the reference instruction text, rather than importing
# production constants and comparing the implementation with itself.
REFERENCE_INSTRUCTION = (
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        # A developer's inherited evaluation knobs must not influence the
        # test's distinction between unspecified and explicitly supplied values.
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def config(self, argv=(), benchmarks=None):
        selected = benchmarks or ["mvbench", "videomme", "tempcompass"]
        return run.make_config(run.parser().parse_args(list(argv)), selected)

    def sample(self, answer="B", media_path="/secret/classes/GT-B.mp4", metadata=None):
        return Sample("private-question-id", "videomme", "Which activity happens?",
                      ("Walking", "Running"), answer, media_path,
                      metadata=metadata or {"answer": "CLASS_LABEL_SECRET", "secret": "METADATA_SECRET"})

    def test_videor1_prompt_preserves_instruction_and_normalizes_question_options_layout(self):
        actual = build_prompt(self.sample(), True, "videor1")
        expected = "Which activity happens?\nOptions:\nA. Walking\nB. Running\n\n" + REFERENCE_INSTRUCTION
        self.assertEqual(actual, expected)
        self.assertEqual(run.build_prompt(self.sample(), True, "videor1"), expected)

    def test_prompt_cannot_change_when_ground_truth_paths_or_metadata_change(self):
        left = self.sample(answer="A", media_path="/secret/path/ANSWER_A.mp4",
                           metadata={"secret": "LEFT_PRIVATE_METADATA", "answer": "A"})
        right = self.sample(answer="B", media_path="/different/path/ANSWER_B.mp4",
                            metadata={"secret": "RIGHT_PRIVATE_METADATA", "answer": "B"})
        for style in ("generic", "videor1"):
            with self.subTest(style=style):
                prompt = build_prompt(left, True, style)
                self.assertEqual(prompt, build_prompt(right, True, style))
                for private in (left.uid, left.media_path, right.media_path,
                                "LEFT_PRIVATE_METADATA", "RIGHT_PRIVATE_METADATA"):
                    self.assertNotIn(private, prompt)

    def test_generic_prompt_remains_compatible_with_existing_callers(self):
        sample = self.sample()
        expected_direct = (
            "Select the best answer based on the video. Respond with only the option letter.\n"
            "Question: Which activity happens?\nOptions:\n(A) Walking\n(B) Running"
        )
        self.assertEqual(build_prompt(sample, False), expected_direct)
        self.assertEqual(run.build_prompt(sample, False), expected_direct)
        self.assertIn("Reason briefly using only visible video evidence", build_prompt(sample, True))

    def test_invalid_prompt_styles_and_non_thinking_reference_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "THINK_MODE=1"):
            build_prompt(self.sample(), False, "videor1")
        with self.assertRaisesRegex(ValueError, "Unknown prompt style"):
            build_prompt(self.sample(), True, "misspelled")

    def test_profile_defaults_are_complete_and_self_describing(self):
        config = self.config(["--eval-profile", "videor1_64"])
        expected = {
            "eval_profile": "videor1_64", "prompt_style": "videor1",
            "video_sampling": "uniform_fps_floor", "think_mode": True, "n": 1,
            "temperature": 0.01, "top_p": 0.001, "max_new_tokens": 1024,
            "max_model_len": 16384, "repetition_penalty": 1.0, "fps": 2.0,
            "min_frames": 4, "min_pixels": 100352, "max_pixels": 200704,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(config[key], value)
        self.assertEqual(config["max_frames_by_benchmark"], {"mvbench": 64, "videomme": 64, "tempcompass": 64})
        self.assertTrue(config["protocol_reference"])
        self.assertIn("videor1", config["protocol"])

    def test_environment_can_select_profile_without_any_knob_overrides(self):
        with patch.dict(os.environ, {"EVAL_PROFILE": "videor1_64"}):
            config = self.config(benchmarks=["videomme"])
        self.assertEqual(config["eval_profile"], "videor1_64")
        self.assertEqual(config["max_frames_by_benchmark"], {"videomme": 64})

    def test_each_conflicting_inherited_environment_setting_is_rejected(self):
        conflicting = {
            "THINK_MODE": "0", "N": "2", "TEMPERATURE": "0", "TOP_P": "1",
            "MAX_NEW_TOKENS": "2048", "MAX_MODEL_LEN": "32768",
            "REPETITION_PENALTY": "1.05", "FPS": "1", "MIN_FRAMES": "2",
            "MIN_PIXELS": "3136", "MAX_PIXELS": "262144", "MAX_FRAMES": "32",
            "MVBENCH_MAX_FRAMES": "32", "VIDEOMME_MAX_FRAMES": "32", "TEMPCOMPASS_MAX_FRAMES": "32",
        }
        for name, value in conflicting.items():
            with self.subTest(name=name):
                with patch.dict(os.environ, {"EVAL_PROFILE": "videor1_64", name: value}):
                    with self.assertRaisesRegex(ValueError, "videor1_64 requires"):
                        self.config()

    def test_conflicting_explicit_cli_setting_is_rejected(self):
        for flag, value in (("--temperature", "0"), ("--think-mode", "0"),
                            ("--n", "2"), ("--videomme-max-frames", "32")):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "videor1_64 requires"):
                    self.config(["--eval-profile", "videor1_64", flag, value])

    def test_matching_explicit_cli_and_environment_settings_are_accepted(self):
        with patch.dict(os.environ, {"EVAL_PROFILE": "videor1_64", "MAX_FRAMES": "64",
                                     "N": "1", "THINK_MODE": "1", "TEMPERATURE": "0.010",
                                     "REPETITION_PENALTY": "1"}):
            config = self.config(["--top-p", "0.001", "--max-new-tokens", "1024",
                                  "--max-model-len", "16384", "--fps", "2",
                                  "--min-pixels", "100352", "--max-pixels", "200704",
                                  "--min-frames", "4"])
        self.assertEqual(config["temperature"], 0.01)
        self.assertEqual(config["max_frames"], 64)

    def test_generic_defaults_and_explicit_customization_are_preserved(self):
        config = self.config()
        self.assertEqual((config["eval_profile"], config["prompt_style"], config["video_sampling"]),
                         ("generic", "generic", "benchmark_default"))
        self.assertEqual((config["temperature"], config["top_p"], config["repetition_penalty"]), (0.0, 1.0, 1.05))
        self.assertEqual((config["max_new_tokens"], config["max_model_len"]), (2048, 32768))
        self.assertEqual(config["max_frames_by_benchmark"], {"mvbench": 32, "videomme": 64, "tempcompass": 32})
        direct = self.config(["--think-mode", "0"])
        self.assertEqual(direct["max_new_tokens"], 128)
        repeated = self.config(["--n", "3"])
        self.assertEqual(repeated["temperature"], 0.7)
        customized = self.config(["--temperature", "0.2", "--repetition-penalty", "1.2", "--mvbench-max-frames", "16"])
        self.assertEqual(customized["temperature"], 0.2)
        self.assertEqual(customized["repetition_penalty"], 1.2)
        self.assertEqual(customized["max_frames_by_benchmark"]["mvbench"], 16)

    def fixture(self):
        dataset = self.root / "data"
        (dataset / "data").mkdir(parents=True)
        (dataset / "data" / "clip.mp4").touch()
        (dataset / "test.json").write_text(json.dumps([{
            "question_id": "one", "videoID": "clip", "question": "Which activity happens?",
            "options": ["A. Walking", "B. Running"], "answer": "B",
            "duration": "short", "task_type": "Action Recognition",
        }]), encoding="utf-8")
        return ["--backend", "mock", "--benchmarks", "videomme", "--videomme-data-dir", str(dataset),
                "--output-dir", str(self.root / "outputs"), "--exp-name", "isolation",
                "--allow-nonstandard-counts"]

    def test_profile_cli_preflight_runs_without_loading_any_model(self):
        command = [sys.executable, "-m", "general_video_eval.run", *self.fixture(),
                   "--eval-profile", "videor1_64", "--preflight"]
        result = subprocess.run(command, cwd=PROJECT, env={"PYTHONPATH": str(PROJECT)},
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Data preflight passed", result.stdout)
        self.assertFalse((self.root / "outputs").exists())

    def test_switching_profile_cannot_overwrite_or_resume_existing_experiment(self):
        argv = self.fixture()
        run_dir = self.root / "outputs" / "isolation"

        def fixed_worker_output(args, config, pending, destination):
            records = [{"uid": sample.uid, "benchmark": sample.benchmark,
                        "prediction": "A", "responses": ["A"], "parsed_answers": ["A"]}
                       for sample in pending]
            if records:
                (destination / "predictions" / "worker-0.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            return []

        with patch.object(run, "run_workers", side_effect=fixed_worker_output), redirect_stdout(io.StringIO()):
            self.assertEqual(run.main([*argv, "--eval-profile", "videor1_64"]), 0)
            paths = [run_dir / name for name in ("manifest.json", "dataset.jsonl", "summary.json")]
            paths.append(run_dir / "predictions" / "worker-0.jsonl")
            originals = {path: path.read_bytes() for path in paths}
            manifest = json.loads(originals[run_dir / "manifest.json"])
            self.assertEqual(manifest["config"]["eval_profile"], "videor1_64")
            self.assertEqual(manifest["config"]["prompt_style"], "videor1")
            self.assertEqual(manifest["config"]["video_sampling"], "uniform_fps_floor")
            for profile in ("generic", "simple_32"):
                with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "Resume refused"):
                    run.main([*argv, "--eval-profile", profile])
                self.assertEqual(originals, {path: path.read_bytes() for path in paths})
            self.assertEqual(run.main([*argv, "--eval-profile", "videor1_64"]), 0)
            self.assertEqual((run_dir / "predictions" / "worker-0.jsonl").read_bytes(), originals[paths[-1]])

    def test_paired_models_share_every_protocol_setting(self):
        base = self.config(["--eval-profile", "videor1_64", "--model-path", "base-model-id"])
        trained = self.config(["--eval-profile", "videor1_64", "--model-path", "trained-model-id"])
        for config in (base, trained):
            config.pop("model_path")
            config.pop("model_identity")
        self.assertEqual(base, trained)


if __name__ == "__main__":
    unittest.main()
