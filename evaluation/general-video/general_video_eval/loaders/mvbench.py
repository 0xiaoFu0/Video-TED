"""Local MVBench annotations, including original clips and preconverted MP4s.

Original MVBench has 20 JSON files and includes TVQA frame directories. The
separate OpenGVLab ``video`` revision contains clips that are already cropped;
its ``test.json``/``MVBench_MP4.tsv`` must not inherit original clip bounds.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

from ..schema import Sample, normalize_answer, normalize_choices
from .common import read_rows


# Official OpenGVLab/MVBench task definitions, in its published table order.
TASKS = {
    "Action Sequence": ("action_sequence", "star/Charades_v1_480", "video", True),
    "Action Prediction": ("action_prediction", "star/Charades_v1_480", "video", True),
    "Action Antonym": ("action_antonym", "ssv2_video", "video", False),
    "Fine-grained Action": ("fine_grained_action", "Moments_in_Time_Raw/videos", "video", False),
    "Unexpected Action": ("unexpected_action", "FunQA_test/test", "video", False),
    "Object Existence": ("object_existence", "clevrer/video_validation", "video", False),
    "Object Interaction": ("object_interaction", "star/Charades_v1_480", "video", True),
    "Object Shuffle": ("object_shuffle", "perception/videos", "video", False),
    "Moving Direction": ("moving_direction", "clevrer/video_validation", "video", False),
    "Action Localization": ("action_localization", "sta/sta_video", "video", True),
    "Scene Transition": ("scene_transition", "scene_qa/video", "video", False),
    "Action Count": ("action_count", "perception/videos", "video", False),
    "Moving Count": ("moving_count", "clevrer/video_validation", "video", False),
    "Moving Attribute": ("moving_attribute", "clevrer/video_validation", "video", False),
    "State Change": ("state_change", "perception/videos", "video", False),
    "Fine-grained Pose": ("fine_grained_pose", "nturgbd", "video", False),
    "Character Order": ("character_order", "perception/videos", "video", False),
    "Egocentric Navigation": ("egocentric_navigation", "vlnqa", "video", False),
    "Episodic Reasoning": ("episodic_reasoning", "tvqa/frames_fps3_hq", "frames", True),
    "Counterfactual Inference": ("counterfactual_inference", "clevrer/video_validation", "video", False),
}

def _task_name(value: Any) -> str:
    key = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    # Human task names include Fine-grained, whose filename omits the hyphen.
    for name, spec in TASKS.items():
        if key in {spec[0], name.casefold().replace("-", "_").replace(" ", "_")}:
            return name
    raise ValueError(f"Unknown MVBench task {value!r}; expected an official task name")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _seconds(value: Any) -> float | None:
    if value is None or str(value).strip().casefold() in {"", "none", "nan", "null"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Invalid MVBench clip timestamp {value!r}")
    return result


def _choices(value: Any) -> tuple[str, ...]:
    # VLMEvalKit TSV files store Python list literals, not always valid JSON.
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("MVBench candidates must contain a serialized list") from exc
    return normalize_choices(value)


def _clean_prefix(value: Any) -> Path:
    text = str(value or "").replace("\\", "/")
    if text == "your_data_path":
        text = ""
    if text.startswith("your_data_path/"):
        text = text[len("your_data_path/"):]
    return Path(text)


def _media_path(root: Path, prefix: Any, video: Any) -> str:
    """Resolve a bounded list of known extraction layouts, without tree scans."""
    video_path = Path(str(video).replace("\\", "/"))
    if video_path.is_absolute():
        return str(video_path)
    prefix_path = _clean_prefix(prefix)
    if prefix_path.is_absolute():
        absolute = prefix_path / video_path
        if absolute.exists():
            return str(absolute)
        # Keep absolute prefixes useful for intentional external dataset roots.
        return str(absolute)
    relative = prefix_path / video_path
    candidates: list[Path] = []
    for base in (root, root / "MVBench"):
        if relative.parts and relative.parts[0] in {"video", "videos", "data0613"}:
            candidates.append(base / relative)
        candidates.extend((
            base / "video" / relative,
            base / relative,
            base / "videos" / relative,
            base / "video" / "data0613" / relative,
            base / "data0613" / relative,
            base / "videos" / "data0613" / relative,
        ))
        # Some converted annotations already include their directory in video.
        candidates.extend((base / "video" / video_path, base / video_path, base / "videos" / video_path))
    unique = list(dict.fromkeys(candidates))
    return str(next((path for path in unique if path.exists()), unique[0]))


def _make_sample(root: Path, row: dict[str, Any], task: str, index: int,
                 source: Path, layout: str, converted: bool = False,
                 prefix_override: Any = None) -> Sample:
    stem, default_prefix, default_type, default_bound = TASKS[task]
    choices = _choices(row.get("candidates", row.get("choices", row.get("options"))))
    video = row.get("video", row.get("video_path"))
    if not video:
        raise ValueError(f"{source}, row {index}: MVBench video path is missing")
    prefix = row.get("prefix") or (prefix_override if prefix_override is not None else default_prefix)
    # Converted MP4 variants contain the original clip already, including TVQA.
    media_type = "video" if converted else str(row.get("data_type", default_type)).casefold()
    if media_type == "frame":
        media_type = "frames"
    if Path(str(video)).suffix.casefold() == ".gif":
        media_type = "gif"
    bounded = False if converted else default_bound or _bool(row.get("bound", False))
    start = _seconds(row.get("start")) if bounded else None
    end = _seconds(row.get("end")) if bounded else None
    if bounded and (start is None or end is None):
        raise ValueError(f"{source}, {task}, row {index}: original bounded clip requires start/end seconds")
    source_fps = float(row.get("frame_fps", row.get("source_fps", 3)) or 3)
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"{source}, row {index}: invalid frame-directory FPS {source_fps}")
    media_path = _media_path(root, prefix, video)
    # NTU RGB files may be full-length MP4 transcodes of official AVI filenames.
    if task == "Fine-grained Pose" and Path(str(video)).suffix.casefold() == ".avi" and not Path(media_path).is_file():
        mp4_video = str(Path(str(video)).with_suffix(".mp4"))
        for candidate_prefix in (prefix, "nturgbd_convert"):
            candidate = _media_path(root, candidate_prefix, mp4_video)
            if Path(candidate).is_file():
                media_path = candidate
                break
    return Sample(
        uid=f"mvbench:{stem}:{index}",
        benchmark="mvbench",
        question=str(row["question"]),
        choices=choices,
        answer=normalize_answer(row["answer"], choices),
        media_path=media_path,
        media_type=media_type,
        start=start,
        end=end,
        frame_fps=source_fps,
        metadata={
            "task": task,
            "source_layout": layout,
            "annotation_file": str(source),
            "annotation_row": index,
            "source_video": str(video),
            "source_prefix": str(prefix),
            "converted_clip": converted,
            "sampling": "segment_midpoints",
        },
    )


def _load_container(root: Path, source: Path) -> list[Sample]:
    data = read_rows(source)
    if isinstance(data, dict) and isinstance(data.get("meta"), dict):
        roots = data.get("root", {})
        samples = []
        for raw_task, rows in data["meta"].items():
            task = _task_name(raw_task)
            if not isinstance(rows, list):
                raise ValueError(f"{source}: meta[{raw_task!r}] must be a list")
            prefix = roots.get(raw_task, roots.get(task, "")) if isinstance(roots, dict) else roots
            for index, row in enumerate(rows):
                samples.append(_make_sample(root, row, task, index, source,
                                            "converted_mp4_json", True, prefix))
        return samples
    if not isinstance(data, list):
        raise ValueError(f"{source}: expected MVBench annotation rows or converted meta/root object")
    converted = "mp4" in source.stem.casefold()
    default_task = None
    if source.stem.casefold() in {spec[0] for spec in TASKS.values()}:
        default_task = _task_name(source.stem)
    samples = []
    task_indices: dict[str, int] = {}
    for row in data:
        if not isinstance(row, dict):
            raise ValueError(f"{source}: each annotation row must be an object")
        task = _task_name(row.get("task_type", row.get("task", default_task)))
        index = task_indices.get(task, 0)
        task_indices[task] = index + 1
        layout = "converted_mp4_tsv" if converted else "original_tsv" if source.suffix.lower() == ".tsv" else "original_json"
        samples.append(_make_sample(root, row, task, index, source, layout, converted))
    return samples


def _complete(samples: list[Sample], description: str) -> list[Sample]:
    present = {sample.metadata["task"] for sample in samples}
    missing = [task for task in TASKS if task not in present]
    if missing:
        raise ValueError(f"{description}: incomplete MVBench annotation set; missing/empty tasks: {', '.join(missing)}. "
                         "All 20 tasks are required for the default benchmark. Use an explicit annotation_path only for debugging.")
    return samples


def _load_directory(root: Path, directory: Path) -> list[Sample]:
    sources = [directory / f"{spec[0]}.json" for spec in TASKS.values()]
    missing = [source.name for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Explicit MVBench annotation directory requires all 20 task JSON files: {directory}. "
                                f"Missing: {', '.join(missing)}")
    samples = [sample for source in sources for sample in _load_container(root, source)]
    return _complete(samples, str(directory))


def load_mvbench(data_dir, annotation_path=None) -> list[Sample]:
    """Read local data. Default mode requires all 20 official tasks.

    Relative annotation paths resolve against data_dir. An explicit directory
    requires all 20 task JSONs; an explicit file can contain fewer tasks for debugging;
    the runner must identify such limited evaluations in its manifest/results.
    Missing media are intentionally returned to the common preflight report.
    """
    root = Path(data_dir).expanduser().resolve()
    if annotation_path is not None:
        source = Path(annotation_path).expanduser()
        if not source.is_absolute():
            source = root / source
        if source.is_dir():
            return _load_directory(root, source)
        if not source.is_file():
            raise FileNotFoundError(f"Explicit MVBench annotation path does not exist: {source}")
        return _load_container(root, source)
    annotation_dirs = (root / "json", root / "annotations", root / "annotation", root,
                       root / "MVBench" / "json", root / "MVBench" / "annotations")
    for directory in annotation_dirs:
        found = [(task, directory / f"{spec[0]}.json") for task, spec in TASKS.items()
                 if (directory / f"{spec[0]}.json").is_file()]
        if len(found) == len(TASKS):
            return _load_directory(root, directory)
    for relative in ("MVBench.tsv", "MVBench_MP4.tsv", "test.json", "annotations/test.json",
                     "MVBench/MVBench.tsv", "MVBench/MVBench_MP4.tsv", "MVBench/test.json"):
        source = root / relative
        if source.is_file():
            return _complete(_load_container(root, source), str(source))
    best_dir = max(annotation_dirs, key=lambda directory: sum((directory / f"{spec[0]}.json").is_file()
                                                            for spec in TASKS.values()))
    missing_files = [f"{spec[0]}.json" for spec in TASKS.values()
                     if not (best_dir / f"{spec[0]}.json").is_file()]
    raise FileNotFoundError(f"Complete MVBench annotations not found under {root}. Expected json/ with all 20 task files, "
                            "MVBench.tsv, or converted test.json/MVBench_MP4.tsv. "
                            f"Missing in {best_dir}: {', '.join(missing_files)}")
