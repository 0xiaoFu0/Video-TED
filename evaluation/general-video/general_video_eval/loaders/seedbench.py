"""SEED-Bench-1 video MCQ, using the official question-keyed eight-frame pack.

The upstream ZIP central directory was checked against SEED-Bench.json.
Do not resolve frames by data_id: different questions can refer to different
segments of the same source video. The pack is keyed by question_id instead.
"""
from __future__ import annotations

from pathlib import Path
import re

from .common import find_file, read_rows
from ..schema import Sample


TASKS = {
    10: ("Action Recognition", "task10/ssv2_8_frame"),
    11: ("Action Prediction", "task11/kitchen_8_frame"),
    12: ("Procedure Understanding", "task12/breakfast_8_frame"),
}
TASK_COUNTS = {"Action Recognition": 1509, "Action Prediction": 1225,
               "Procedure Understanding": 1023}
FRAME_NAMES = tuple(f"{i}.png" for i in range(1, 9))


def _frame_root(root: Path) -> Path:
    candidates = [root / "v1_video", root]
    found = [p for p in candidates if any((p / task_path).is_dir() for _, task_path in TASKS.values())]
    if len(found) > 1:
        raise ValueError(f"Ambiguous SEED-Bench frame roots: {found}; keep one extracted layout")
    return found[0] if found else candidates[0]


def load_seedbench(data_dir, annotation_path=None) -> list[Sample]:
    root = Path(data_dir).expanduser().resolve()
    path = find_file(root, ["SEED-Bench.json"], "SEED-Bench annotations", annotation_path)
    payload = read_rows(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        raise ValueError(f"SEED-Bench requires an object with a questions list: {path}")
    frames_root = _frame_root(root)
    samples, seen = [], set()
    for index, row in enumerate(payload["questions"]):
        if not isinstance(row, dict):
            raise ValueError(f"SEED-Bench row {index}: expected a question object")
        data_type = str(row.get("data_type", "")).casefold()
        raw_type = row.get("question_type_id")
        # Accept integer strings, but never truncate fractional IDs or booleans.
        if isinstance(raw_type, bool) or not re.fullmatch(r"\d+", str(raw_type)):
            raise ValueError(f"SEED-Bench row {index}: invalid question_type_id {raw_type!r}")
        type_id = int(raw_type)
        if (data_type == "video") != (type_id in TASKS):
            raise ValueError(f"SEED-Bench row {index}: inconsistent data_type/question_type_id")
        if data_type != "video":
            continue
        question_id = str(row.get("question_id", ""))
        try:
            if not re.fullmatch(r"v\d+", question_id):
                raise ValueError("expected official question_id such as v1738")
            if question_id in seen:
                raise ValueError("duplicate video question_id")
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("missing or empty question")
            raw_choices = [row.get(f"choice_{label}") for label in "abcd"]
            if any(not isinstance(choice, str) or not choice.strip() for choice in raw_choices):
                raise ValueError("expected four nonempty choice_a..choice_d strings")
            answer = row.get("answer")
            if answer not in ("A", "B", "C", "D"):
                raise ValueError("expected ground-truth answer A, B, C, or D")
            data_id = row.get("data_id")
            if not isinstance(data_id, str) or not data_id.strip():
                raise ValueError("missing data_id")
            task, relative = TASKS[type_id]
            samples.append(Sample(
                uid=f"seedbench:{question_id}", benchmark="seedbench",
                question=question.strip(), choices=tuple(s.strip() for s in raw_choices),
                answer=answer, media_path=str(frames_root / relative / question_id),
                media_type="frames", frame_fps=2.0,
                metadata={
                    "source_question_id": question_id, "question_type_id": type_id,
                    "task": task, "data_id": data_id,
                    "source_segment": row.get("segment"),
                    "annotation_file": str(path), "frame_files": list(FRAME_NAMES),
                    "sampling": "seedbench_official_8", "protocol": "SEED-Bench-1/video/official-eight-frames",
                    "timing_source": "synthetic_ordinal_2fps", "source_timing_known": False,
                },
            ))
            seen.add(question_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SEED-Bench {question_id or index} in {path}: {exc}") from exc
    if not samples:
        raise ValueError(f"No SEED-Bench video questions in {path}")
    return samples


def check_frame_coverage(samples: list[Sample], *, decode=False) -> None:
    """Check each required image. Optional full decoding needs Pillow, no GPU.

    Additional archive entries are allowed; the annotation determines coverage.
    Do not silently accept an existing question directory with missing frames.
    """
    errors, error_count, checked = [], 0, 0
    if decode:
        from PIL import Image
        print(f"SEED-Bench: decoding all {len(samples) * 8} required images before GPU initialization...", flush=True)
    for index, sample in enumerate(samples, 1):
        for name in sample.metadata["frame_files"]:
            path = Path(sample.media_path) / name
            try:
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError("missing or empty image")
                if decode:
                    with Image.open(path) as image:
                        image.load()
                        image.convert("RGB").load()
                checked += 1
            except (OSError, ValueError) as exc:
                error_count += 1
                if len(errors) < 12:
                    errors.append(f"{path}: {exc}")
        if decode and index % 200 == 0:
            print(f"SEED-Bench image check: {index}/{len(samples)} questions", flush=True)
    if errors:
        raise ValueError(f"SEED-Bench frame preflight failed before GPU initialization: {error_count} bad images; examples:\n  " + "\n  ".join(errors))
    print(f"SEED-Bench frame coverage passed: {len(samples)} questions, {checked} images" +
          ("; all images decoded." if decode else "; existence and nonzero size checked."), flush=True)
