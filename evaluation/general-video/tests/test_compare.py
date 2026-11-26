"""Comparison must not silently mix incompatible benchmark protocols."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from general_video_eval.compare import ComparisonError, compare_runs, main, render_csv, render_markdown


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.ours = self.root / "ours"
        self.base.mkdir()
        self.ours.mkdir()
        self.base_manifest, self.base_summary = self.fixture("base")
        self.ours_manifest, self.ours_summary = self.fixture("ours", improvement=True)
        self.save()

    def fixture(self, name, improvement=False, mock=False, benchmarks=None):
        all_expected = {"mvbench": 4000, "videomme": 2700, "tempcompass": 1580, "seedbench": 3757}
        expected = {name: all_expected[name] for name in
                    (benchmarks if benchmarks is not None else ("mvbench", "videomme", "tempcompass"))}
        action_counts = {"mvbench": 200, "videomme": 100, "tempcompass": 338, "seedbench": 1509}
        config = {"model_path": f"/models/{name}", "model_identity": {"name": name}, "backend": "mock" if mock else "vllm",
                  "benchmarks": list(expected), "eval_profile": "videor1_64", "video_sampling": "uniform_fps_floor",
                  "max_frames_by_benchmark": {key: 64 for key in expected}, "fps": 2, "min_frames": 4,
                  "max_pixels": 262144, "min_pixels": 3136, "think_mode": True, "n": 1, "temperature": 0,
                  "max_new_tokens": 2048, "protocol": "video_frames_only_no_subtitles", "source_sha256": "c" * 64,
                  "dependencies": {"vllm": "0.11.0", "transformers": "4.57.3"}, "gpu_memory_utilization": 0.8}
        datasets, buckets = {}, {}
        for benchmark, full in expected.items():
            total = 10 if mock else full
            action = 2 if mock else action_counts[benchmark]
            correct = total // 2 + (1 if improvement else 0)
            action_correct = action // 2 + (1 if improvement else 0)
            datasets[benchmark] = {"scope": "partial_or_nonstandard" if mock else "full",
                                   "selected_questions": total, "annotation_questions": total,
                                   "official_expected_questions": full, "data_dir": f"/data/{benchmark}"}
            bucket = {"n_total": total, "n_done": total, "n_correct": correct, "complete": True,
                      "overall": correct / total * 100, "n_errors": 0, "invalid_answers": 0,
                      "action_recognition_n_total": action, "action_recognition_n_done": action,
                      "action_recognition_n_correct": action_correct, "action_recognition": action_correct / action * 100}
            if benchmark == "mvbench":
                bucket.update(avg=bucket["overall"], fa=bucket["action_recognition"])
            buckets[benchmark] = bucket
        manifest = {"exp_name": name, "config": config, "datasets": datasets, "dataset_sha256": "d" * 64}
        summary = {"experiment": name, "backend": config["backend"], "is_mock": mock, "complete": True,
                   "scope": copy.deepcopy(datasets), "benchmarks": buckets, "errors": [], "n_errors": 0,
                   "n_total": sum(bucket["n_total"] for bucket in buckets.values()),
                   "n_done": sum(bucket["n_done"] for bucket in buckets.values()),
                   "n_correct": sum(bucket["n_correct"] for bucket in buckets.values())}
        return manifest, summary

    def save(self):
        for directory, manifest, summary in ((self.base, self.base_manifest, self.base_summary),
                                              (self.ours, self.ours_manifest, self.ours_summary)):
            (directory / "manifest.json").write_text(json.dumps(manifest))
            (directory / "summary.json").write_text(json.dumps(summary))

    def test_scores_and_percentage_point_delta_are_rendered(self):
        comparison = compare_runs(self.base, self.ours)
        self.assertEqual(len(comparison["rows"]), 6)
        first, action = comparison["rows"][:2]
        self.assertEqual((first["benchmark"], first["metric"], first["questions"]), ("MVBench", "Avg", 4000))
        self.assertAlmostEqual(first["delta_pp"], 0.025)
        self.assertEqual(action["metric"], "FA")
        self.assertAlmostEqual(action["delta_pp"], 0.5)
        self.assertIn("uniform_fps_floor", render_markdown(comparison))
        self.assertIn("| MVBench | FA | 200 | full | 50.00 | 50.50 | +0.50 |", render_markdown(comparison))
        self.assertIn("MVBench,FA,200,full,50.000000,50.500000,0.500000,False", render_csv(comparison))
        self.assertIn("videor1_64,uniform_fps_floor,2,4,64,3136,262144,/models/base,/models/ours", render_csv(comparison))

    def test_profile_source_dependencies_and_gpu_settings_must_match(self):
        for key, changed in (("eval_profile", "generic"), ("source_sha256", "e" * 64),
                             ("dependencies", {"vllm": "0.10.0"}), ("gpu_memory_utilization", 0.7)):
            with self.subTest(key=key):
                original = self.ours_manifest["config"][key]
                self.ours_manifest["config"][key] = changed
                self.save()
                with self.assertRaisesRegex(ComparisonError, "config differs"):
                    compare_runs(self.base, self.ours)
                self.ours_manifest["config"][key] = original

    def test_seedbench_video_only_comparison_reports_official_counts(self):
        self.base_manifest, self.base_summary = self.fixture("base", benchmarks=("seedbench",))
        self.ours_manifest, self.ours_summary = self.fixture("ours", improvement=True, benchmarks=("seedbench",))
        self.save()
        comparison = compare_runs(self.base, self.ours)
        self.assertEqual(len(comparison["rows"]), 2)
        overall, action = comparison["rows"]
        self.assertEqual((overall["benchmark"], overall["metric"], overall["questions"]),
                         ("SEED-Bench Video", "Overall", 3757))
        self.assertEqual((action["benchmark"], action["metric"], action["questions"]),
                         ("SEED-Bench Video", "Action", 1509))
        self.assertAlmostEqual(overall["delta_pp"], 100 / 3757)
        self.assertAlmostEqual(action["delta_pp"], 100 / 1509)
        self.assertIn("SEED-Bench Video Action is the Action Recognition task (question type 10).",
                      render_markdown(comparison))
        self.assertIn("SEED-Bench Video,Action,1509,full", render_csv(comparison))

    def test_seedbench_real_action_count_must_be_official(self):
        self.base_manifest, self.base_summary = self.fixture("base", benchmarks=("seedbench",))
        self.ours_manifest, self.ours_summary = self.fixture("ours", improvement=True, benchmarks=("seedbench",))
        self.ours_summary["benchmarks"]["seedbench"]["action_recognition_n_total"] = 1508
        self.save()
        with self.assertRaisesRegex(ComparisonError, "official 1509 questions"):
            compare_runs(self.base, self.ours)

    def test_seedbench_mock_allows_small_pipeline_fixture(self):
        self.base_manifest, self.base_summary = self.fixture("base", mock=True, benchmarks=("seedbench",))
        self.ours_manifest, self.ours_summary = self.fixture("ours", improvement=True, mock=True,
                                                             benchmarks=("seedbench",))
        self.save()
        comparison = compare_runs(self.base, self.ours, allow_mock=True)
        self.assertEqual([row["questions"] for row in comparison["rows"]], [10, 2])
        self.assertTrue(all(row["is_mock"] for row in comparison["rows"]))

    def test_dataset_hash_and_count_mismatches_are_rejected(self):
        self.ours_manifest["dataset_sha256"] = "e" * 64
        self.save()
        with self.assertRaisesRegex(ComparisonError, "dataset_sha256 differs"):
            compare_runs(self.base, self.ours)
        self.ours_manifest["dataset_sha256"] = "d" * 64
        self.ours_summary["benchmarks"]["mvbench"]["n_total"] -= 1
        self.save()
        with self.assertRaisesRegex(ComparisonError, "counts"):
            compare_runs(self.base, self.ours)

    def test_partial_real_results_and_incomplete_results_are_rejected(self):
        self.ours_manifest["datasets"]["mvbench"]["scope"] = "partial_or_nonstandard"
        self.ours_summary["scope"] = copy.deepcopy(self.ours_manifest["datasets"])
        self.save()
        with self.assertRaisesRegex(ComparisonError, "scope=full"):
            compare_runs(self.base, self.ours)
        self.ours_summary["complete"] = False
        self.save()
        with self.assertRaisesRegex(ComparisonError, "incomplete"):
            compare_runs(self.base, self.ours)

    def test_missing_provenance_cannot_pass_as_equal_empty_settings(self):
        for manifest in (self.base_manifest, self.ours_manifest):
            manifest["config"].pop("source_sha256")
        self.save()
        with self.assertRaisesRegex(ComparisonError, "source_sha256"):
            compare_runs(self.base, self.ours)

    def test_mock_requires_explicit_flag_and_remains_marked(self):
        self.base_manifest, self.base_summary = self.fixture("base", mock=True)
        self.ours_manifest, self.ours_summary = self.fixture("ours", improvement=True, mock=True)
        self.save()
        with self.assertRaisesRegex(ComparisonError, "--allow-mock"):
            compare_runs(self.base, self.ours)
        comparison = compare_runs(self.base, self.ours, allow_mock=True)
        self.assertIn("MOCK COMPARISON", render_markdown(comparison))
        self.assertTrue(all(row["is_mock"] for row in comparison["rows"]))
        self.ours_manifest, self.ours_summary = self.fixture("ours")
        self.save()
        with self.assertRaisesRegex(ComparisonError, "MOCK and real"):
            compare_runs(self.base, self.ours, allow_mock=True)

    def test_partial_mock_without_action_questions_shows_no_fabricated_score(self):
        self.base_manifest, self.base_summary = self.fixture("base", mock=True)
        self.ours_manifest, self.ours_summary = self.fixture("ours", improvement=True, mock=True)
        for summary in (self.base_summary, self.ours_summary):
            bucket = summary["benchmarks"]["mvbench"]
            for key in ("action_recognition_n_total", "action_recognition_n_done", "action_recognition_n_correct"):
                bucket[key] = 0
            bucket.update(action_recognition=None, fa=None)
        self.save()
        comparison = compare_runs(self.base, self.ours, allow_mock=True)
        self.assertIn("| MVBench | FA | 0 | partial_or_nonstandard | — | — | — |", render_markdown(comparison))

    def test_cli_writes_new_artifacts_and_preserves_inputs(self):
        before = {path: path.read_bytes() for directory in (self.base, self.ours)
                  for path in (directory / "summary.json", directory / "manifest.json")}
        output = io.StringIO()
        with redirect_stdout(output):
            result = main([str(self.base), str(self.ours)])
        self.assertEqual(result, 0)
        self.assertIn("TempCompass-MCQ", output.getvalue())
        self.assertTrue((self.ours / "comparison.md").is_file())
        self.assertTrue((self.ours / "comparison.csv").is_file())
        self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))
        self.ours_manifest["config"]["eval_profile"] = "generic"
        self.save()
        errors = io.StringIO()
        with redirect_stderr(errors):
            result = main([str(self.base), str(self.ours)])
        self.assertEqual(result, 2)
        self.assertIn("comparison refused", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
