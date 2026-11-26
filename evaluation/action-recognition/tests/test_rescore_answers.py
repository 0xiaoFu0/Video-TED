"""Regression tests for missing-tag answers; no GPUs or model weights."""
import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import (EVAL_SPLITS, PROMPT_PROTOCOL, ROOT, atomic_json, digest,
                      load_dataset, official, read_json, registry, summarize)
from rescore_answers import CandidateParser, SCORING_PROTOCOL, recover_records, rescore


@pytest.fixture
def parser():
    return CandidateParser({"0": "Brush hair: grooming", "1": "Walk: locomotion", "2": "Run: locomotion"})


@pytest.mark.parametrize("text,label,method", [
    ("The woman moves the brush through her hair.\n\nBrush hair", "Brush hair", "exact_final_line"),
    ("The motion continues throughout the video frames.\n\nBrush hair", "Brush hair", "exact_final_line"),
    ("<think>Visible grooming.</think>\nBrush hair", "Brush hair", "exact_final_line"),
    ("**brush_hair**", "Brush hair", "exact_final_line"),
    ("<answer>Brush hair</answer>", "Brush hair", "answer_tag"),
    ("<Answer>Brush hair</Answer>", "Brush hair", "answer_tag"),
    ("<ANSWER>\nbrush_hair\n</ANSWER>", "Brush hair", "answer_tag"),
    ("<answer>Walk</answer>\n<answer>walk</answer>", "Walk", "answer_tag"),
    ("Brush hair or walk is discussed here.\n\nRun", "Run", "exact_final_line"),
])
def test_valid_final_answers(parser, text, label, method):
    assert parser.parse(text, "stop") == (label, method)


@pytest.mark.parametrize("text", [
    "The action is brush hair.",
    "Brush hair is observed in the video.",
    "Brush hair\nThis is my conclusion.",
    "Brush hair or Walk",
    "Answer: Brush hair",  # Not part of v1's deliberately narrow fallback.
    "<think>Brush hair</think>",
    "<think>Reasoning continues.\nBrush hair",
    "<answer>Walk</answer>\n<answer>Brush hair</answer>",
    "<answer>Unknown</answer>\nBrush hair",
    "<answer></answer>\nBrush hair",
    "<answer>Brush hair",
    "<answer>Walk</answer>\n<answer>Brush hair",
    "Walk\nBrush hair",
    "\n\n",
])
def test_no_prose_matching_or_ambiguous_recovery(parser, text):
    assert parser.parse(text, "stop")[0] == ""


def test_truncated_plain_answer_is_not_recovered(parser):
    assert parser.parse("Evidence\nBrush hair", "length")[0] == ""
    # A complete explicit answer before truncation can still be scored.
    assert parser.parse("<answer>Walk</answer>", "length")[0] == "Walk"
    assert parser.parse("<answer>Walk</answer>", "error", "CUDA error")[0] == ""


def test_wrong_final_label_stays_wrong_and_all_rows_count():
    actions = {"0": "Brush hair: a", "1": "Walk: b"}
    records = [
        {"solution": "<answer>brush hair</answer>", "predict": "", "inference_error": None,
         "finish_reason": "stop", "inference_output": "A woman is brushing her hair.\nBrush hair"},
        {"solution": "<answer>brush hair</answer>", "predict": "", "inference_error": None,
         "finish_reason": "stop", "inference_output": "Walk"},
        {"solution": "<answer>brush hair</answer>", "predict": "", "inference_error": None,
         "finish_reason": "length", "inference_output": "Brush hair"},
    ]
    fixed, audit = recover_records(records, actions)
    assert summarize(fixed, 3, 3)["accuracy"] == 1/3
    assert audit["newly_correct"] == 1
    assert fixed[1]["predict"] == "<answer>Walk</answer>"
    altered, _ = recover_records([{**row, "solution": "<answer>Walk</answer>", "path": "walk/leak.avi"}
                                   for row in records], actions)
    assert [r["predict"] for r in fixed] == [r["predict"] for r in altered]
    assert all(row["predict"] == "" for row in records)  # Original records were not mutated.


@pytest.fixture
def source_run(tmp_path):
    run = tmp_path / "original"
    metrics = {}
    for key in EVAL_SPLITS:
        spec = registry()["datasets"][key]
        rows, _, count = load_dataset(spec, 3)
        manifest = {"benchmark": key, "prompt_protocol": PROMPT_PROTOCOL, "inference_passes": 1,
                    "model": {"path": "fixture-not-real-model"}, "selected": 3, "official_split_size": count,
                    "annotations_sha256": digest(ROOT / spec["annotations"]),
                    "classes_sha256": digest(ROOT / spec["classes"])}
        api = official(spec["entry"])
        records = []
        for i, row in enumerate(rows):
            label = row["solution"].replace("<answer>", "").replace("</answer>", "")
            raw = row["solution"] if i == 0 else "Visible evidence.\n\n" + label
            records.append({**row, "inference_output": raw, "predict": api["extract_answer"](raw),
                            "finish_reason": "stop", "inference_error": None})
        atomic_json(run / key / "run.json", manifest)
        atomic_json(run / key / "results.json", records)
        metrics[key] = summarize(records, 3, count)
    atomic_json(run / "summary.json", {"model_name": "CPU-FIXTURE", "metrics": metrics,
                                       "training_audit": {"model_role": "base"}})
    return run


def test_offline_rescore_all_splits_preserves_sources_and_exports(source_run):
    before = {str(p): digest(p) for p in source_run.rglob("*.json")}
    result = rescore(source_run, allow_subset=True)
    assert all(m["accuracy"] == 1 for m in result["metrics"].values())
    assert all(m["accuracy"] == 1/3 for m in result["strict_metrics"].values())
    assert all(a["newly_correct"] == 2 for a in result["audit"].values())
    assert all(digest(Path(p)) == sha for p, sha in before.items())
    dest = source_run / "answer_rescore_v1"
    assert read_json(dest / "summary.json")["scoring_protocol"] == SCORING_PROTOCOL
    for name in ("models.csv", "models_strict.csv"):
        with (dest / name).open() as handle:
            table = list(csv.DictReader(handle))
        assert len(table) == 1 and table[0]["Subset only"] == "True"
        assert table[0]["HMDB51 Base (%)"] == ("100.00" if name == "models.csv" else "33.33")
    for key in EVAL_SPLITS:
        assert len((dest / (key + ".answers.jsonl")).read_text().splitlines()) == 3
    with pytest.raises(ValueError, match="already exists"):
        rescore(source_run, allow_subset=True)


@pytest.mark.parametrize("damage", ["missing", "ground_truth", "hash", "summary", "inference_error", "raw_mismatch", "subset"])
def test_invalid_input_never_publishes_report(source_run, damage):
    result_path = source_run / "hmdb51-split1/results.json"
    rows = read_json(result_path)
    if damage == "missing":
        atomic_json(result_path, rows[:-1])
    elif damage == "ground_truth":
        rows[0]["solution"] = "<answer>wrong</answer>"
        atomic_json(result_path, rows)
    elif damage == "hash":
        path = source_run / "hmdb51-split1/run.json"
        atomic_json(path, {**read_json(path), "classes_sha256": "wrong"})
    elif damage == "summary":
        path = source_run / "summary.json"
        value = read_json(path)
        value["metrics"]["hmdb51-split1"]["invalid_answers"] = 99
        atomic_json(path, value)
    elif damage == "inference_error":
        rows[0]["inference_error"] = "test runtime error"
        atomic_json(result_path, rows)
    elif damage == "raw_mismatch":
        rows[0]["inference_output"] = "different output"
        atomic_json(result_path, rows)
    with pytest.raises(ValueError):
        rescore(source_run, allow_subset=(damage != "subset"))
    assert not (source_run / "answer_rescore_v1").exists()
    assert not list(source_run.glob(".answer-rescore-*"))


def test_no_gpu_dependency_imported():
    import subprocess
    result = subprocess.run([sys.executable, "-c",
        "import sys; import rescore_answers; assert not ({'torch','vllm','transformers'} & sys.modules.keys())"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
