"""Keep MVBench's five official single-option questions without inventing answers."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from general_video_eval.loaders.mvbench import load_mvbench
from general_video_eval.schema import Sample
from general_video_eval.scoring import extract_answer, majority_vote, summarize


# The original OpenGVLab/MVBench counterfactual_inference.json contains these
# single-option cases at zero-based indices 81, 98, 111, 137, and 162. Preserve
# the source questions, videos, and answers here so tests need no download.
OFFICIAL_SINGLETON_ROWS = [
    {"video": "video_10978.mp4", "question": "Without the purple object, which event will happen?",
     "candidates": ["The cube collides with the gray cylinder"],
     "answer": "The cube collides with the gray cylinder"},
    {"video": "video_11854.mp4", "question": "What will happen without the sphere?",
     "candidates": ["The rubber cylinder collides with the cube"],
     "answer": "The rubber cylinder collides with the cube"},
    {"video": "video_10427.mp4", "question": "Without the cylinder, which of the following will happen?",
     "candidates": ["The brown cube and the sphere collide"],
     "answer": "The brown cube and the sphere collide"},
    {"video": "video_10462.mp4", "question": "Which of the following will happen without the cylinder?",
     "candidates": ["The cyan object and the brown cube collide"],
     "answer": "The cyan object and the brown cube collide"},
    {"video": "video_13455.mp4", "question": "If the sphere is removed, which of the following will happen?",
     "candidates": ["The cyan cylinder collides with the yellow cylinder"],
     "answer": "The cyan cylinder collides with the yellow cylinder"},
]


class SingleChoiceRegressionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.annotation = self.root / "counterfactual_inference.json"
        self.annotation.write_text(json.dumps(OFFICIAL_SINGLETON_ROWS), encoding="utf-8")

    def test_mvbench_loader_preserves_all_official_single_option_cases(self):
        samples = load_mvbench(self.root, self.annotation)
        self.assertEqual(len(samples), 5)
        self.assertEqual(len({sample.uid for sample in samples}), 5)
        for sample, row in zip(samples, OFFICIAL_SINGLETON_ROWS):
            with self.subTest(video=row["video"]):
                self.assertEqual(sample.choices, tuple(row["candidates"]))
                self.assertEqual(sample.answer, "A")
                self.assertEqual(sample.labels, ("A",))
                self.assertEqual(sample.question, row["question"])
                self.assertEqual(sample.metadata["task"], "Counterfactual Inference")
                self.assertEqual(Path(sample.media_path).name, row["video"])

    def test_single_option_parser_accepts_only_an_explicit_valid_answer(self):
        choice = ("The cube collides with the gray cylinder",)
        for response in ("A", "<answer>A</answer>", "<think>Visible collision.</think><answer>A</answer>",
                         choice[0], "A. " + choice[0]):
            with self.subTest(response=response):
                self.assertEqual(extract_answer(response, choice), "A")

    def test_single_option_does_not_turn_invalid_outputs_into_a(self):
        choice = ("The cube collides with the gray cylinder",)
        for response in ("", "I cannot answer", "<think>I might choose A", "B", "<answer>B</answer>",
                         "<think><answer>A</answer></think>", "<answer>Unknown</answer>"):
            with self.subTest(response=response):
                self.assertIsNone(extract_answer(response, choice))

    def test_all_invalid_single_option_completions_have_no_majority(self):
        choice = ("The cube collides with the gray cylinder",)
        parsed = [extract_answer(response, choice) for response in ("", "B", "<think>A")]
        self.assertEqual(parsed, [None, None, None])
        self.assertIsNone(majority_vote(parsed))

    def test_invalid_single_option_predictions_score_zero_without_dropping_questions(self):
        samples = load_mvbench(self.root, self.annotation)
        responses = ("", "B", "<think>A", "I cannot answer", "<answer>Unknown</answer>")
        records = [{
            "uid": sample.uid, "benchmark": sample.benchmark,
            "prediction": extract_answer(response, sample.choices),
            "responses": [response], "parsed_answers": [extract_answer(response, sample.choices)],
        } for sample, response in zip(samples, responses)]
        report = summarize(samples, records)
        self.assertEqual((report["n_total"], report["n_done"], report["n_correct"]), (5, 5, 0))
        self.assertEqual(report["overall"], 0)
        self.assertEqual(report["invalid_answers"], 5)
        task = report["benchmarks"]["mvbench"]["by_task"]["Counterfactual Inference"]
        self.assertEqual(task["n_total"], 5)
        self.assertEqual(task["accuracy"], 0)

    def test_missing_single_option_predictions_remain_in_denominator(self):
        samples = load_mvbench(self.root, self.annotation)
        report = summarize(samples, [])
        self.assertEqual((report["n_total"], report["n_missing"], report["n_correct"]), (5, 5, 0))
        self.assertEqual(report["overall"], 0)
        self.assertFalse(report["complete"])

    def test_single_choice_exception_is_limited_to_mvbench(self):
        for benchmark in ("videomme", "tempcompass"):
            with self.subTest(benchmark=benchmark):
                with self.assertRaises(ValueError):
                    Sample("one", benchmark, "What happens?", ("collision",), "A", "/missing.mp4")
        sample = Sample("one", "MVBENCH", "What happens?", ("collision",), "A", "/missing.mp4")
        self.assertEqual(sample.labels, ("A",))

    def test_empty_choice_list_is_still_rejected(self):
        for benchmark in ("mvbench", "videomme", "tempcompass"):
            with self.subTest(benchmark=benchmark):
                with self.assertRaises(ValueError):
                    Sample("empty", benchmark, "What happens?", (), "A", "/missing.mp4")
        with self.assertRaises(ValueError):
            extract_answer("A", ())


if __name__ == "__main__":
    unittest.main()
