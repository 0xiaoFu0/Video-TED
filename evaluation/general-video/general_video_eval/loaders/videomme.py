"""Load local Video-MME annotations with the video-only, no-subtitle protocol."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from .common import find_file, read_rows
from ..schema import Sample, normalize_answer, normalize_choices


def _annotation_paths(root: Path, override):
    if override:
        return [find_file(root, [], "Video-MME annotations", override)]
    # These globs are bounded to annotation directories, never the video tree.
    for directory in (root / "videomme", root):
        shards = sorted(directory.glob("test*.parquet"))
        if shards:
            return shards
    return [find_file(root, [
        "Video-MME.tsv", "videomme.json", "Video-MME.json", "test.json",
        "annotations.json", "videomme/test.json", "videomme/test.jsonl", "test.jsonl",
    ], "Video-MME annotations (videomme/test*.parquet, test*.parquet, or JSON/TSV)")]


def _rows(payload):
    if isinstance(payload, dict):
        if "questions" in payload or "question" in payload:
            payload = [payload]
        elif isinstance(payload.get("data"), list):
            payload = payload["data"]
        else:
            raise ValueError("Video-MME JSON must contain a list of rows or a 'data' list")
    if not isinstance(payload, list):
        raise ValueError("Video-MME annotations must contain a list of rows")
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Video-MME annotation rows must be objects")
        if "questions" not in row:
            yield row
            continue
        questions = row["questions"]
        if not isinstance(questions, list):
            raise ValueError(f"Video-MME {row.get('videoID', row.get('video_id'))}: questions must be a list")
        base = {key: value for key, value in row.items() if key != "questions"}
        for question in questions:
            if not isinstance(question, dict):
                raise ValueError(f"Video-MME {base.get('videoID')}: question entry must be an object")
            yield dict(base, **question)


def _choices(row):
    raw = row.get("options", row.get("choices", row.get("candidates")))
    if raw is None:
        raw = {letter: row[letter] for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter in row}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            # VLMEvalKit TSV writes Python list literals, not JSON arrays.
            raw = ast.literal_eval(raw)
    choices = normalize_choices(raw)
    if not 2 <= len(choices) <= 26 or any(not choice for choice in choices):
        raise ValueError("expected 2..26 nonempty answer options")
    return choices


def _media_path(root, row, video_id):
    candidates = []
    explicit = row.get("video_path")
    if explicit:
        path = Path(str(explicit)).expanduser()
        candidates.append(path if path.is_absolute() else root / path)
    filename = str(video_id)
    if not Path(filename).suffix:
        filenames = [filename + extension for extension in (".mp4", ".MP4", ".mkv")]
    else:
        filenames = [filename]
    for directory in ("data", "video/data", "video", "videos", ""):
        for name in filenames:
            candidates.append(root / directory / name)
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    return str(candidates[0].absolute())


def load_videomme(data_dir, annotation_path=None) -> list[Sample]:
    """Read all questions; do not download, add subtitles, or silently skip videos.

    Supports the MeSD Hugging Face layout, VLMEvalKit's Video-MME.tsv,
    flat JSON/JSONL fixtures, and the official per-video JSON question template.
    Missing media is retained for the runner's complete preflight report.
    """
    root = Path(data_dir).expanduser().resolve()
    samples = []
    seen = set()
    for path in _annotation_paths(root, annotation_path):
        for index, row in enumerate(_rows(read_rows(path))):
            source_uid = str(row.get("question_id", row.get("id", f"{path.stem}:{index}")))
            uid = f"videomme:{source_uid}"
            try:
                if uid in seen:
                    raise ValueError("duplicate question_id; each question must have a unique ID")
                video_id = row.get("videoID", row.get("video", row.get("video_id")))
                if video_id is None or not str(video_id).strip():
                    raise ValueError("missing videoID/video/video_id")
                question = row.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("missing or empty question")
                choices = _choices(row)
                if "answer" not in row:
                    raise ValueError("missing ground-truth answer")
                answer = normalize_answer(row["answer"], choices)
                metadata = {
                    "video_id": str(video_id), "source_question_id": source_uid, "annotation_file": str(path),
                    "protocol": "video-only/no-subtitles",
                }
                for field in ("duration", "domain", "sub_category", "task_type"):
                    if row.get(field) is not None:
                        metadata[field] = row[field]
                samples.append(Sample(
                    uid=uid, benchmark="videomme", question=question.strip(), choices=choices,
                    answer=answer, media_path=_media_path(root, row, video_id), metadata=metadata,
                ))
                seen.add(uid)
            except (TypeError, ValueError, SyntaxError) as exc:
                raise ValueError(f"Video-MME {uid} in {path}: {exc}") from exc
    if not samples:
        raise ValueError(f"Video-MME annotation files contain no questions in {root}")
    return samples
