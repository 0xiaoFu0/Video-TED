"""Regressions for answer parsing, invalid votes, and expected-set scoring."""
from __future__ import annotations

import unittest

from general_video_eval.schema import Sample
from general_video_eval.scoring import extract_answer, majority_vote, summarize


class AnswerParsingTests(unittest.TestCase):
    choices = ("walking", "running", "jumping", "sitting")

    def test_invalid_output_does_not_become_a(self):
        for text in ("", "I cannot answer", "The A and B options differ", "I think B", "<think>Considering B",
                     "The answer is a person walking", "The answer is A person walking"):
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text, self.choices))
        self.assertIsNone(majority_vote([None, None, "invalid", ""]))

    def test_answer_tag_without_complete_think(self):
        self.assertEqual(extract_answer("<answer>B</answer>", self.choices), "B")
        self.assertEqual(extract_answer("<think>Truncated reasoning <answer>C</answer>", self.choices), "C")
        self.assertIsNone(extract_answer("<think><answer>A</answer></think>", self.choices))
        self.assertIsNone(extract_answer("<answer>B</answer><answer>Unknown</answer>", self.choices))

    def test_completed_think_and_chat_template_continuation(self):
        self.assertEqual(extract_answer("<think>The answer is A.</think>\nB", self.choices), "B")
        self.assertEqual(extract_answer("I considered A. </think>\n(C)", self.choices), "C")

    def test_supported_plain_formats(self):
        for text in ("B", "(B)", "B.", "Final answer: B", "Answer: (B)",
                     "Some reasoning.\nFinal answer: B", "The answer is B because the person runs.",
                     "running", "B. running", "**B**"):
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text, self.choices), "B")
        self.assertIsNone(extract_answer("Final answer: B or C", self.choices))
        self.assertIsNone(extract_answer("B. jumping", self.choices))

    def test_variable_option_count_and_unique_exact_text(self):
        self.assertIsNone(extract_answer("C", ("walk", "run")))
        self.assertEqual(extract_answer("E", ("one", "two", "three", "four", "five")), "E")
        self.assertEqual(extract_answer("Z", tuple(str(index) for index in range(26))), "Z")
        self.assertIsNone(extract_answer("same", ("same", "same")))
        self.assertIsNone(extract_answer("The person is running", ("walking", "running")))

    def test_vote_ties_follow_completion_order(self):
        self.assertEqual(majority_vote([None, "C", "B", "B", "C"]), "C")
        self.assertEqual(majority_vote([None, "D", "A"]), "D")
        self.assertEqual(majority_vote([None, "D", "B", "B"]), "B")


class ScoringTests(unittest.TestCase):
    def sample(self, uid, benchmark="mvbench", answer="A", **metadata):
        return Sample(uid, benchmark, "Which action?", ("walk", "run"), answer, "/missing.mp4", metadata=metadata)

    def record(self, sample, prediction):
        return {"uid": sample.uid, "benchmark": sample.benchmark, "prediction": prediction,
                "responses": [], "parsed_answers": []}

    def test_partial_coverage_keeps_all_expected_denominators(self):
        samples = [self.sample("one", task="Fine-grained Action"),
                   self.sample("two", task="Fine-grained Action"),
                   self.sample("three", task="Action Sequence")]
        report = summarize(samples, [self.record(samples[0], "A"), self.record(samples[2], None)],
                           [{"uid": "two", "benchmark": "mvbench", "error": "decode failed"}])
        mvbench = report["benchmarks"]["mvbench"]
        self.assertFalse(report["complete"])
        self.assertEqual((report["n_total"], report["n_done"], report["n_correct"]), (3, 2, 1))
        self.assertAlmostEqual(mvbench["overall"], 100 / 3)
        self.assertEqual(mvbench["action_recognition"], 50)
        self.assertEqual(mvbench["action_recognition_n_total"], 2)
        self.assertEqual(mvbench["avg"], 25)
        self.assertEqual(mvbench["fa"], 50)
        self.assertEqual(mvbench["invalid_answers"], 1)
        self.assertEqual(mvbench["n_missing"], 1)
        self.assertEqual(mvbench["by_task"]["Fine-grained Action"]["n_total"], 2)

    def test_action_definitions_and_other_dimension_counts(self):
        samples = [self.sample("m", task="Fine-grained Action"),
                   self.sample("v1", "videomme", task_type="Action Recognition", duration="short", domain="Sports"),
                   self.sample("v2", "videomme", task_type="Action Reasoning", duration="long", domain="Sports"),
                   self.sample("t1", "tempcompass", dim="action"),
                   self.sample("t2", "tempcompass", dim="order")]
        report = summarize(samples, [self.record(sample, "A" if sample.uid in {"m", "v2", "t1"} else "B")
                                     for sample in samples])
        self.assertTrue(report["complete"])
        video = report["benchmarks"]["videomme"]
        self.assertEqual(video["overall"], 50)
        self.assertEqual(video["action_recognition"], 0)
        self.assertEqual(video["action_recognition_n_total"], 1)
        self.assertEqual(video["by_duration"]["short"]["accuracy"], 0)
        self.assertEqual(video["by_domain"]["Sports"]["accuracy"], 50)
        temp = report["benchmarks"]["tempcompass"]
        self.assertEqual(temp["action_recognition"], 100)
        self.assertEqual(temp["by_dimension"]["order"]["accuracy"], 0)

    def test_all_recorded_but_invalid_is_complete_and_incorrect(self):
        sample = self.sample("one", task="Fine-grained Action")
        report = summarize([sample], [self.record(sample, None)])
        self.assertTrue(report["complete"])
        self.assertEqual(report["overall"], 0)
        self.assertEqual(report["invalid_answers"], 1)

    def test_seedbench_video_uses_pooled_accuracy_and_recognition_only(self):
        samples = [self.sample("r1", "seedbench", task="Action Recognition", question_type_id=10),
                   self.sample("r2", "seedbench", task="Action Recognition", question_type_id=10),
                   self.sample("r3", "seedbench", task="Action Recognition", question_type_id=10),
                   self.sample("p1", "seedbench", task="Action Prediction", question_type_id=11),
                   self.sample("p2", "seedbench", task="Action Prediction", question_type_id=11),
                   self.sample("u1", "seedbench", task="Procedure Understanding", question_type_id=12)]
        records = [self.record(sample, None if sample.uid == "r2" else "B" if sample.uid == "u1" else "A")
                   for sample in samples if sample.uid != "r3"]
        errors = [{"uid": "r3", "benchmark": "seedbench", "error": "decode failed"}]
        report = summarize(samples, records, errors)
        seed = report["benchmarks"]["seedbench"]
        self.assertEqual((seed["n_total"], seed["n_done"], seed["n_correct"]), (6, 5, 3))
        self.assertEqual(seed["overall"], 50)
        self.assertEqual((seed["action_recognition_n_total"], seed["action_recognition_n_done"],
                          seed["action_recognition_n_correct"]), (3, 2, 1))
        self.assertAlmostEqual(seed["action_recognition"], 100 / 3)
        self.assertEqual((seed["invalid_answers"], seed["n_missing"]), (1, 1))
        self.assertFalse(seed["complete"])
        self.assertEqual(set(seed["by_task"]), {"Action Recognition", "Action Prediction", "Procedure Understanding"})
        self.assertEqual(seed["by_task"]["Action Prediction"]["accuracy"], 100)
        self.assertEqual(seed["by_task"]["Procedure Understanding"]["accuracy"], 0)
        self.assertFalse(seed["by_task"]["Action Recognition"]["complete"])
        self.assertTrue(seed["by_task"]["Action Prediction"]["complete"])
        self.assertNotIn("avg", seed)

    def test_errors_prevent_complete_even_if_record_exists(self):
        sample = self.sample("one", task="Fine-grained Action")
        report = summarize([sample], [self.record(sample, "A")], [{"benchmark": "mvbench", "error": "worker failed"}])
        self.assertFalse(report["complete"])
        self.assertFalse(report["benchmarks"]["mvbench"]["complete"])

    def test_duplicates_unknown_ids_and_benchmark_mismatches_rejected(self):
        sample = self.sample("one", task="Fine-grained Action")
        record = self.record(sample, "A")
        with self.assertRaisesRegex(ValueError, "Duplicate expected"):
            summarize([sample, sample], [])
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            summarize([sample], [record, record])
        with self.assertRaisesRegex(ValueError, "Unknown prediction"):
            summarize([sample], [{"uid": "other", "prediction": "A"}])
        with self.assertRaisesRegex(ValueError, "benchmark does not match"):
            summarize([sample], [{**record, "benchmark": "tempcompass"}])
        with self.assertRaisesRegex(ValueError, "Unknown failed"):
            summarize([sample], [], [{"uid": "other", "error": "failed"}])

    def test_empty_action_category_returns_none_not_zero(self):
        sample = self.sample("one", "videomme", task_type="Action Reasoning")
        report = summarize([sample], [self.record(sample, "A")])
        self.assertIsNone(report["benchmarks"]["videomme"]["action_recognition"])
        self.assertEqual(report["benchmarks"]["videomme"]["action_recognition_n_total"], 0)
        self.assertIsNone(summarize([], [])["overall"])


if __name__ == "__main__":
    unittest.main()
