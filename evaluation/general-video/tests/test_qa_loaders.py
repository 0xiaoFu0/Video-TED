"""CPU-only checks for local annotation and media layouts."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from general_video_eval.loaders.tempcompass import load_tempcompass
from general_video_eval.loaders.videomme import load_videomme


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def write_json(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def touch(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path.resolve())

    def video_row(self, **changes):
        row = dict(question_id="001-1", videoID="youtube_id", question="What is happening?",
                   options=["A. running", "B. jumping", "C. sitting", "D. walking"],
                   answer="B", duration="short", domain="Life Record", sub_category="Daily Life",
                   task_type="Action Recognition")
        row.update(changes)
        return row

    def temp_row(self, **changes):
        row = dict(video_id="clip_01", dim="action", question="What action?\nA. running\nB. jumping\nC. sitting",
                   answer="B. jumping")
        row.update(changes)
        return row

    def test_videomme_mesd_media_layout_and_official_metadata(self):
        self.write_json("videomme/test.json", [self.video_row()])
        expected = self.touch("data/youtube_id.mp4")
        sample = load_videomme(self.root)[0]
        self.assertEqual(sample.media_path, expected)
        self.assertEqual(sample.answer, "B")
        self.assertEqual(sample.choices, ("running", "jumping", "sitting", "walking"))
        self.assertEqual(sample.metadata["task_type"], "Action Recognition")
        self.assertEqual(sample.metadata["duration"], "short")
        self.assertNotIn("answer", sample.metadata)

    def test_videomme_official_nested_questions_and_video_data_layout(self):
        row = self.video_row()
        video = {key: row.pop(key) for key in ("videoID", "duration", "domain", "sub_category")}
        video["questions"] = [row]
        self.write_json("videomme.json", [video])
        expected = self.touch("video/data/youtube_id.mp4")
        sample = load_videomme(self.root)[0]
        self.assertEqual(sample.media_path, expected)
        self.assertEqual(sample.uid, "videomme:001-1")
        self.assertEqual(sample.metadata["source_question_id"], "001-1")
        self.assertEqual(sample.metadata["domain"], "Life Record")

    def test_videomme_vlmeval_tsv_and_explicit_media_path(self):
        row = self.video_row()
        row["candidates"] = repr(row.pop("options"))
        row["video"] = row.pop("videoID")
        row["video_path"] = "./custom/youtube_id.mp4"
        path = self.root / "Video-MME.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), delimiter="\t")
            writer.writeheader()
            writer.writerow(row)
        expected = self.touch("custom/youtube_id.mp4")
        self.assertEqual(load_videomme(self.root)[0].media_path, expected)

    def test_videomme_missing_video_kept_for_preflight(self):
        self.write_json("test.json", [self.video_row()])
        sample = load_videomme(self.root)[0]
        self.assertEqual(sample.media_path, str(self.root / "data/youtube_id.mp4"))

    def test_videomme_jsonl_override_and_text_answer(self):
        path = self.root / "custom.jsonl"
        path.write_text(json.dumps(self.video_row(answer="jumping")) + "\n", encoding="utf-8")
        self.assertEqual(load_videomme(self.root, path)[0].answer, "B")

    def test_videomme_relative_override(self):
        self.write_json("annotations/custom.json", [self.video_row()])
        sample = load_videomme(self.root, "annotations/custom.json")[0]
        self.assertEqual(sample.uid, "videomme:001-1")
        self.assertEqual(sample.metadata["annotation_file"], str(self.root / "annotations/custom.json"))

    def test_videomme_rejects_duplicate_ids_and_bad_answer_with_id(self):
        path = self.write_json("test.json", [self.video_row(), self.video_row()])
        with self.assertRaisesRegex(ValueError, "001-1.*duplicate"):
            load_videomme(self.root)
        self.write_json("test.json", [self.video_row(answer="Z")])
        with self.assertRaisesRegex(ValueError, "001-1.*ground-truth"):
            load_videomme(self.root, path)

    def test_tempcompass_hf_shape_variable_choices_and_dimensions(self):
        rows = [self.temp_row(), self.temp_row(dim="direction", question="Which direction?\nA. left\nB. right", answer="A")]
        self.write_json("multi-choice/test.json", rows)
        expected = self.touch("videos/clip_01.mp4")
        samples = load_tempcompass(self.root)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].question, "What action?")
        self.assertEqual(samples[0].choices, ("running", "jumping", "sitting"))
        self.assertEqual(samples[0].answer, "B")
        self.assertEqual(samples[0].media_path, expected)
        self.assertEqual(samples[1].metadata["dim"], "direction")
        self.assertEqual(len(samples[1].choices), 2)
        self.assertEqual(len({sample.uid for sample in samples}), 2)

    def test_tempcompass_original_nested_release(self):
        row = self.temp_row()
        row.pop("video_id")
        row.pop("dim")
        self.write_json("questions/multi-choice.json", {"clip_01": {"action": [row]}})
        sample = load_tempcompass(self.root)[0]
        self.assertEqual(sample.metadata["video_id"], "clip_01")
        self.assertEqual(sample.metadata["dim"], "action")
        self.assertEqual(sample.media_path, str(self.root / "videos/clip_01.mp4"))

    def test_tempcompass_explicit_options_and_jsonl_override(self):
        path = self.root / "custom.jsonl"
        row = self.temp_row(question="What action?", options=["A. running", "B. jumping"], answer=1)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.assertEqual(load_tempcompass(self.root, path)[0].answer, "B")

    def test_tempcompass_relative_override(self):
        self.write_json("annotations/custom.json", [self.temp_row(question_id="relative")])
        sample = load_tempcompass(self.root, "annotations/custom.json")[0]
        self.assertEqual(sample.uid, "tempcompass:relative")
        self.assertEqual(sample.metadata["annotation_file"], str(self.root / "annotations/custom.json"))

    def test_tempcompass_official_option_without_space_after_letter(self):
        self.write_json("test.json", [self.temp_row(question="Which event first?\nA. dancing\nB. waving\nC.Both events", answer="B. waving")])
        sample = load_tempcompass(self.root)[0]
        self.assertEqual(sample.choices, ("dancing", "waving", "Both events"))

    def test_benchmark_namespace_prevents_cross_benchmark_id_collision(self):
        video_path = self.write_json("videomme.json", [self.video_row(question_id="0")])
        temp_path = self.write_json("multi-choice.json", [self.temp_row(question_id="0")])
        video = load_videomme(self.root, video_path)[0]
        temp = load_tempcompass(self.root, temp_path)[0]
        self.assertEqual(video.uid, "videomme:0")
        self.assertEqual(temp.uid, "tempcompass:0")
        self.assertEqual(temp.metadata["source_question_id"], "0")

    def test_tempcompass_never_merges_caption_matching(self):
        self.write_json("multi-choice/test.json", [self.temp_row()])
        self.write_json("caption_matching/test.json", [self.temp_row()])
        self.assertEqual(len(load_tempcompass(self.root)), 1)
        with self.assertRaisesRegex(ValueError, "requires multi-choice"):
            load_tempcompass(self.root, self.root / "caption_matching/test.json")

    def test_tempcompass_rejects_wrong_task_and_invalid_dimension(self):
        path = self.write_json("test.json", [self.temp_row(task_type="caption_matching")])
        with self.assertRaisesRegex(ValueError, "clip_01.*multi-choice QA only"):
            load_tempcompass(self.root)
        self.write_json("test.json", [self.temp_row(dim="garbage")])
        with self.assertRaisesRegex(ValueError, "clip_01.*unknown or missing dim"):
            load_tempcompass(self.root, path)

    def test_tempcompass_rejects_malformed_choices_and_gt(self):
        path = self.write_json("test.json", [self.temp_row(question="What action?\nA. running\nC. jumping")])
        with self.assertRaisesRegex(ValueError, "clip_01.*consecutive"):
            load_tempcompass(self.root)
        self.write_json("test.json", [self.temp_row(answer="Z")])
        with self.assertRaisesRegex(ValueError, "clip_01.*ground-truth"):
            load_tempcompass(self.root, path)

    def test_empty_annotations_are_errors(self):
        path = self.write_json("test.json", [])
        for loader in (load_tempcompass, load_videomme):
            with self.subTest(loader=loader.__name__):
                with self.assertRaisesRegex(ValueError, "no questions"):
                    loader(self.root, path)


if __name__ == "__main__":
    unittest.main()
