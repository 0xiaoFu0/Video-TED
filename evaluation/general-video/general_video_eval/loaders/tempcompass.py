"""TempCompass multiple-choice QA only (not VLMEvalKit's merged MCQ task)."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

from .common import find_file, read_rows
from ..schema import Sample, normalize_answer, normalize_choices

DIMENSIONS = ("action", "direction", "speed", "order", "attribute_change")
_OPTION_LINE = re.compile(r"^\s*(?:\(([A-Z])\)|([A-Z])[.)])\s*(.+?)\s*$")


def _annotation_paths(root, override):
    if override:
        return [find_file(root, [], "TempCompass multi-choice annotations", override)]
    shards = sorted((root / "multi-choice").glob("test*.parquet"))
    if shards:
        return shards
    return [find_file(root, [
        "questions/multi-choice.json", "multi-choice.json", "multi-choice/test.json",
        "multi-choice/test.jsonl", "multi-choice.jsonl", "test.json", "test.jsonl",
    ], "TempCompass multi-choice annotations (multi-choice/test*.parquet or questions/multi-choice.json)")]


def _rows(payload):
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("TempCompass rows must be JSON objects")
            yield row
        return
    if not isinstance(payload, dict):
        raise ValueError("TempCompass annotations must be a row list or video_id -> dimension -> questions")
    if "question" in payload:
        yield payload
        return
    # Original release: video_id -> dimension -> list of question/answer objects.
    for video_id, dimensions in payload.items():
        if not isinstance(dimensions, dict):
            raise ValueError(f"TempCompass {video_id}: expected dimension -> questions mapping")
        for dim, questions in dimensions.items():
            if not isinstance(questions, list):
                raise ValueError(f"TempCompass {video_id}/{dim}: expected a question list")
            for row in questions:
                if not isinstance(row, dict):
                    raise ValueError(f"TempCompass {video_id}/{dim}: question must be an object")
                yield dict(row, video_id=str(video_id), dim=dim)


def _question_choices(row):
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing or empty question")
    raw = row.get("options", row.get("choices", row.get("candidates")))
    if raw is not None:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = ast.literal_eval(raw)
        choices = normalize_choices(raw)
    else:
        question_lines, option_lines, labels = [], [], []
        reached_options = False
        for line in question.splitlines():
            if not line.strip():
                continue
            match = _OPTION_LINE.match(line)
            if match:
                reached_options = True
                labels.append(match.group(1) or match.group(2))
                option_lines.append(match.group(3))
            elif reached_options:
                raise ValueError("option lines must each begin with a letter such as 'A. '")
            else:
                question_lines.append(line.strip())
        expected = [chr(65 + index) for index in range(len(labels))]
        if labels != expected:
            raise ValueError(f"answer options must be consecutive A, B, ...; found {labels}")
        question = "\n".join(question_lines)
        choices = tuple(option_lines)
    if not question.strip() or not 2 <= len(choices) <= 26 or any(not choice for choice in choices):
        raise ValueError("expected a nonempty question and 2..26 nonempty answer options")
    return question.strip(), choices


def _media_path(root, row, video_id):
    candidates = []
    if row.get("video_path"):
        explicit = Path(str(row["video_path"])).expanduser()
        candidates.append(explicit if explicit.is_absolute() else root / explicit)
    filename = str(video_id)
    if not Path(filename).suffix:
        filename += ".mp4"
    candidates.extend(root / directory / filename for directory in ("videos", "video", ""))
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    return str(candidates[0].absolute())


def load_tempcompass(data_dir, annotation_path=None) -> list[Sample]:
    """Load only multi-choice QA; preserve official dimensions for scoring.

    The primary layout is the user's HF download: multi-choice/test*.parquet
    and videos/{video_id}.mp4. Caption matching and other tasks are excluded.
    """
    root = Path(data_dir).expanduser().resolve()
    samples, seen = [], set()
    for path in _annotation_paths(root, annotation_path):
        if any(task in str(path.parent.name).lower() or task in path.stem.lower()
               for task in ("caption", "yes_no", "yes-no")):
            raise ValueError(f"TempCompass-MCQ requires multi-choice annotations, not {path}")
        for index, row in enumerate(_rows(read_rows(path))):
            video_id = row.get("video_id", row.get("video"))
            dim = str(row.get("dim", row.get("dimension", ""))).strip().lower().replace(" ", "_")
            source_uid = str(row.get("question_id", row.get("id", f"{path.stem}:{index}:{video_id}:{dim}")))
            uid = f"tempcompass:{source_uid}"
            try:
                if uid in seen:
                    raise ValueError("duplicate question ID")
                task = row.get("task_type", "multi-choice")
                if task not in ("multi-choice", "multi_choice", "mcq", "MCQ"):
                    raise ValueError(f"task_type={task!r}; this runner accepts multi-choice QA only")
                if video_id is None or not str(video_id).strip():
                    raise ValueError("missing video_id")
                if dim not in DIMENSIONS:
                    raise ValueError(f"unknown or missing dim {dim!r}; expected {', '.join(DIMENSIONS)}")
                question, choices = _question_choices(row)
                if "answer" not in row:
                    raise ValueError("missing ground-truth answer")
                answer = normalize_answer(row["answer"], choices)
                samples.append(Sample(
                    uid=uid, benchmark="tempcompass", question=question, choices=choices,
                    answer=answer, media_path=_media_path(root, row, video_id), metadata={
                        "video_id": str(video_id), "dim": dim, "task_type": "multi-choice",
                        "source_question_id": source_uid,
                        "protocol": "TempCompass-MCQ/multi-choice-only", "annotation_file": str(path),
                    },
                ))
                seen.add(uid)
            except (TypeError, ValueError, SyntaxError) as exc:
                raise ValueError(f"TempCompass-MCQ {uid} in {path}: {exc}") from exc
    if not samples:
        raise ValueError(f"TempCompass multi-choice annotation files contain no questions in {root}")
    return samples
