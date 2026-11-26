"""Regression tests for label-bearing paths in model-visible text (CPU only)."""
import copy
import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protocol
from protocol import (EVAL_SPLITS, load_dataset, messages, multimodal_inputs, official,
                      path_free_template, registry)


def text_of(msg):
    return msg[0]["content"][1]["text"]


@pytest.mark.parametrize("kind", ["ceval1oss", "ceval2oss", "ceval3oss"])
def test_row_targets_and_media_paths_cannot_change_text(kind):
    classes = {"0": "Walk: movement", "1": "Run: fast movement"}
    first_row = {"path": "TRUE_CLASS_ONE/v_TRUE_CLASS_ONE_001.avi",
                 "solution": "<answer>GOLD_ONE</answer>", "explain": "PRIVATE_ONE",
                 "problem": "PRIVATE_QUESTION_ONE"}
    other_row = {"path": "TRUE_CLASS_TWO/v_TRUE_CLASS_TWO_002.avi",
                 "solution": "<answer>GOLD_TWO</answer>", "explain": "PRIVATE_TWO",
                 "problem": "PRIVATE_QUESTION_TWO"}
    api_a = official(kind, classes, "/SOURCE_ONE", "/ANNOTATED_ONE")
    api_b = official(kind, classes, "/SOURCE_TWO", "/ANNOTATED_TWO")
    a = messages(first_row, api_a)
    b = messages(other_row, api_b)
    assert text_of(a) == text_of(b)
    for marker in ("TRUE_CLASS", "GOLD_", "PRIVATE_", "SOURCE_", "ANNOTATED_", ".avi"):
        assert marker not in text_of(a)
    assert "annotated version" not in text_of(a)
    assert "Walk" in text_of(a) and "Run" in text_of(a)
    assert "fast movement" not in text_of(a)
    assert a[0]["content"][0]["video"] == "/SOURCE_ONE/" + first_row["path"]


@pytest.mark.parametrize("key", EVAL_SPLITS)
def test_every_requested_video_has_path_free_prompts(key):
    spec = registry()["datasets"][key]
    rows, classes, count = load_dataset(spec)
    api = official(spec["entry"], classes, "/ACTUAL_MEDIA", "/ANNOTATED_MEDIA")
    expected = text_of(messages(rows[0], api))
    for row in rows:
        msg = messages(row, api)
        assert text_of(msg) == expected
        assert row["path"] not in text_of(msg)
        assert Path(row["path"]).name not in text_of(msg)
        assert "/ACTUAL_MEDIA" not in text_of(msg)
        assert "/ANNOTATED_MEDIA" not in text_of(msg)
    assert count == spec["rows"]


def test_chat_template_gets_markers_but_decoder_gets_original_paths():
    # Serialize *everything* received by the chat processor. A path must not leak
    # even if the chat template starts including extra metadata in the future.
    batch = [[{"role": "user", "content": [
        {"type": "video", "video": "/ApplyEyeMakeup/v_ApplyEyeMakeup_001.avi", "fps": 2},
        {"type": "text", "text": "Choose from Walk and Run."}], "filename": "HIDDEN"}]]
    before = copy.deepcopy(batch)
    class Processor:
        def apply_chat_template(self, msg, **kwargs):
            assert msg[0]["content"][0] == {"type": "video"}
            return json.dumps(msg)
    def vision(msgs, **kwargs):
        assert msgs == before
        return None, ["DECODED_FRAMES"], {"fps": [2.0]}
    result = multimodal_inputs(batch, Processor(), vision)
    assert result[0]["multi_modal_data"] == {"video": "DECODED_FRAMES"}
    assert result[0]["mm_processor_kwargs"] == {"fps": [2.0]}
    assert "ApplyEyeMakeup" not in result[0]["prompt"]
    assert "HIDDEN" not in result[0]["prompt"]
    assert "Choose from Walk and Run." in result[0]["prompt"]
    assert batch == before


def test_changed_upstream_notice_fails_instead_of_reintroducing_path():
    with pytest.raises(ValueError, match="Unexpected upstream"):
        path_free_template("New prompt includes {ANNOTATED_VIDEO_PATH}")
