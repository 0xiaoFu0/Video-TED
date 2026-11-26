import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate
from compare_results import compare
from protocol import (ROOT, atomic_json, load_dataset, load_progress, messages,
                      multimodal_inputs, official, official_scorer, read_json,
                      registry, resolve_video_root, summarize)


def test_upstream_sources_match_pinned_git_blobs():
    manifest = read_json(ROOT / "SOURCE_MANIFEST.json")
    for item in manifest["files"]:
        content = (ROOT / item["package_path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
        assert hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() == item["git_blob"]


@pytest.mark.parametrize("key", sorted(registry()["datasets"]))
def test_all_official_splits_load_with_original_labels(key):
    spec = registry()["datasets"][key]
    rows, classes, full_count = load_dataset(spec)
    assert len(rows) == full_count == spec["rows"]
    score = official_scorer()
    labels = {score["normalize"](s.split(":", 1)[0]) for s in classes.values()}
    targets = {score["normalize"](score["extract_answer"](r["solution"])) for r in rows}
    assert targets <= labels


@pytest.mark.parametrize("kind", ["ceval1oss", "ceval2oss", "ceval3oss"])
def test_shared_recognition_prompt_uses_official_candidates_without_tool_instructions(kind):
    api = official(kind, {"0": "Brush hair: grooming hair"}, "/video", "/pose")
    row = {"path": "brush_hair/a.avi"}
    msg = messages(row, api)
    prompt = msg[0]["content"][1]["text"]
    assert "candidate action labels below:\nBrush hair\n" in prompt
    assert "object position or state" in prompt
    assert "<think>" in prompt and "<answer>" in prompt
    for legacy_instruction in ("Stage 1", "Stage 2", "action definitions", "from its description", "Score each candidate", "annotated version"):
        assert legacy_instruction not in prompt
    assert msg[0]["content"][0]["video"] == "/video/brush_hair/a.avi"
    assert "grooming hair" not in msg[0]["content"][1]["text"]
    assert "three independent tools" not in msg[0]["content"][1]["text"]


def test_media_root_uses_relative_path_not_filename(tmp_path):
    (tmp_path / "wrong_class").mkdir()
    (tmp_path / "wrong_class/a.avi").touch()
    (tmp_path / "UCF-101/CorrectClass").mkdir(parents=True)
    (tmp_path / "UCF-101/CorrectClass/a.avi").touch()
    rows = [{"path": "CorrectClass/a.avi"}, {"path": "CorrectClass/missing.avi"}]
    found, missing = resolve_video_root(tmp_path, rows)
    assert found == tmp_path / "UCF-101"
    assert missing == ["CorrectClass/missing.avi"]


def test_fps_is_kept_as_per_video_list():
    processor = SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt")
    vision = lambda *a, **k: (None, ["video_a", "video_b"], {"fps": [0.586797, 2.0]})
    inputs = multimodal_inputs([[], []], processor, vision)
    assert inputs[0]["mm_processor_kwargs"] == {"fps": [0.586797]}
    assert inputs[1]["mm_processor_kwargs"] == {"fps": [2.0]}
    assert inputs[1]["multi_modal_data"] == {"video": "video_b"}


def test_errors_and_parse_failures_stay_in_denominator():
    results = [
        {"solution": "<answer>Brush hair</answer>", "predict": "<answer>brush_hair</answer>"},
        {"solution": "<answer>wave</answer>", "predict": "<answer>error</answer>", "inference_error": "decode error"},
        {"solution": "<answer>walk</answer>", "predict": ""},
    ]
    stats = summarize(results, 4, 1530)
    assert stats["accuracy"] == 0.25
    assert stats["completed"] == 3 and stats["expected"] == 4 and not stats["complete"]
    assert stats["invalid_answers"] == 1 and stats["inference_errors"] == 1
    assert stats["subset_only"]


def test_resume_requires_order_and_ground_truth(tmp_path):
    file = tmp_path / "results.json"
    expected = [{"path": "a.avi", "solution": "<answer>wave</answer>"}]
    atomic_json(file, [{**expected[0], "predict": "<answer>wave</answer>", "inference_error": None, "finish_reason": "stop", "inference_output": "output"}])
    assert len(load_progress(file, expected)) == 1
    with pytest.raises(ValueError, match="order"):
        load_progress(file, [{**expected[0], "path": "b.avi"}])
    with pytest.raises(ValueError, match="ground truth"):
        load_progress(file, [{**expected[0], "solution": "<answer>walk</answer>"}])


def test_single_pass_execution_and_resume(tmp_path):
    spec = registry()["datasets"]["hmdb51-split1"]
    rows, actions, full_count = load_dataset(spec, 3)
    selected = ("hmdb51-split1", spec, rows, actions, full_count, tmp_path, tmp_path / "pose")
    args = SimpleNamespace(batch_size=2)
    class FakeBackend:
        def __init__(self):
            self.stages = []
            self.calls = 0
        def generate(self, batch):
            first = "three independent tools" in batch[0][0]["content"][1]["text"]
            self.stages.append("first" if first else "second")
            self.calls += 1
            text = "<pose_estimation>no</pose_estimation><person_detection>no</person_detection><noun_explanation>no</noun_explanation>" if first else "<think>Observed grooming.</think><answer>brush_hair</answer>"
            return [(text, "stop") for _ in batch]
    backend = FakeBackend()
    result = evaluate.run_split(backend, args, selected, tmp_path, {})
    assert backend.stages == ["second", "second"]
    assert not (tmp_path / "first_turn.json").exists()
    assert read_json(tmp_path / "progress.json")["completed"] == 3
    assert result["complete"] and result["accuracy"] == 1.0
    before = backend.calls
    evaluate.run_split(backend, args, selected, tmp_path, {})
    assert backend.calls == before


def test_runtime_error_preserves_every_sample(tmp_path):
    spec = registry()["datasets"]["hmdb51-split1"]
    rows, actions, count = load_dataset(spec, 3)
    class Broken:
        def generate(self, batch):
            raise RuntimeError("synthetic decode failure")
    result = evaluate.run_split(Broken(), SimpleNamespace(batch_size=2),
                                ("hmdb51-split1", spec, rows, actions, count, tmp_path, tmp_path), tmp_path, {})
    assert result["completed"] == 3
    assert result["inference_errors"] == 3
    assert result["accuracy"] == 0


def test_missing_media_aborts_before_gpu_initialization(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("GPU backend must not be initialized")
    monkeypatch.setattr(evaluate, "Backend", forbidden)
    assert evaluate.main(["run", "--model", str(tmp_path), "--run-name", "test", "--limit", "2",
                          "--benchmarks", "hmdb51", "--hmdb51-root", str(tmp_path)]) == 2


def test_comparison_rejects_changed_settings(tmp_path):
    spec = registry()["datasets"]["hmdb51-split1"]
    rows, _, _ = load_dataset(spec, 1)
    from protocol import digest
    signature = {"benchmark": "hmdb51-split1", "selected": 1,
                 "annotations_sha256": digest(ROOT / spec["annotations"]), "temperature": 0.8}
    for name in ("base", "trained"):
        directory = tmp_path / ("pair_" + name) / "hmdb51-split1"
        atomic_json(directory / "run.json", {**signature, "model": name})
        atomic_json(directory / "results.json", [{**rows[0], "predict": rows[0]["solution"], "inference_error": None, "finish_reason": "stop", "inference_output": rows[0]["solution"]}])
    assert compare("pair", tmp_path) == 0
    path = tmp_path / "pair_trained/hmdb51-split1/run.json"
    value = read_json(path)
    value["temperature"] = 0
    atomic_json(path, value)
    with pytest.raises(ValueError, match="different"):
        compare("pair", tmp_path)
