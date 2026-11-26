"""Pinned Video-STAR data/scoring, with single-pass, tool-free inference."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPT_PROTOCOL = "videostar-v6-neutral-single-no-tools-no-path-3split"
RECOGNITION_TEMPLATE = """Identify the action shown in the provided video.

Choose exactly one label from the candidate action labels below:
{CANDIDATE_LABELS}

Base your decision on the visible video evidence. Consider motion,
temporal order, interactions between people and objects, and changes
in object position or state when relevant. Distinguish what actually
happens from what is merely attempted or suggested by the scene.

Provide a concise summary of the relevant visual evidence inside
<think>...</think>. Then copy exactly one candidate label inside
<answer>...</answer>.

Output format:
<think>Concise visual evidence summary.</think>
<answer>exact candidate label</answer>
"""
EVAL_SPLITS = ("hmdb51-base", "hmdb51-novel", "ucf101-base", "ucf101-novel",
               "ssv2-base", "ssv2-novel",
               "hmdb51-split1", "hmdb51-split2", "hmdb51-split3",
               "ucf101-split1", "ucf101-split2", "ucf101-split3")

# Keep the vendored source intact, but never show this upstream paragraph to
# the model: HMDB/UCF directory and file names disclose the target action.
# The upstream selector also returns raw video for every tool decision, so
# claiming that annotated media is available would be misleading.
_ANNOTATED_VIDEO_NOTICE = (
    "Note: There is also an annotated version of this video available at: {ANNOTATED_VIDEO_PATH}\n"
    "The annotated video contains the same content but with additional annotations:\n"
    "- Blue bounding boxes around detected persons\n"
    "- Pose estimation points and skeleton lines showing body keypoints\n"
)


def path_free_template(template):
    """Remove the known upstream notice; fail if its shape ever changes."""
    if template.count(_ANNOTATED_VIDEO_NOTICE) != 1:
        raise ValueError("Unexpected upstream annotated-video notice; review prompt before inference")
    clean = template.replace(_ANNOTATED_VIDEO_NOTICE, "", 1)
    if "{ANNOTATED_VIDEO_PATH}" in clean:
        raise ValueError("Unremoved video path placeholder")
    return clean


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def official(kind="ceval2oss", actions=None, video_root="", annotated_root=""):
    """Load upstream helpers on CPU and apply the documented path-removal fix."""
    source = ROOT / "upstream/AScripts" / kind / "4subtol.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"get_allowed_answers_str", "extract_answer", "extract_think",
             "extract_first_turn_decisions", "get_video_path"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"os": os, "re": re, "actions": actions or {},
             "original_video_path": str(video_root), "annotated_video_path": str(annotated_root)}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"FIRST_TURN_TEMPLATE", "SECOND_TURN_TEMPLATE"}:
                    scope[target.id] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
    scope["SECOND_TURN_TEMPLATE"] = path_free_template(scope["SECOND_TURN_TEMPLATE"])
    return scope


def official_scorer():
    source = ROOT / "upstream/AScripts/cresult/1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"extract_answer", "normalize"}]
    scope = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
    return scope


def registry():
    return read_json(ROOT / "benchmarks.json")


def expand_benchmarks(names):
    reg = registry()
    expanded = []
    for name in names:
        keys = reg["groups"].get(name, [name])
        for key in keys:
            if key not in reg["datasets"]:
                raise ValueError("Unknown benchmark: " + key)
            if key not in expanded:
                expanded.append(key)
    return [(key, reg["datasets"][key]) for key in expanded]


def load_dataset(spec, limit=0):
    rows = read_json(ROOT / spec["annotations"])
    actions = read_json(ROOT / spec["classes"])
    if not isinstance(rows, list) or not rows or not isinstance(actions, dict) or not actions:
        raise ValueError("Invalid official annotation/class file")
    seen = set()
    for row in rows:
        path = row.get("path", "")
        if not path or Path(path).is_absolute() or ".." in Path(path).parts or row.get("data_type") != "video":
            raise ValueError("Invalid video entry: " + repr(row))
        if path in seen:
            raise ValueError("Duplicate video path: " + path)
        if not row.get("solution"):
            raise ValueError("Missing ground truth: " + path)
        seen.add(path)
    return (rows[:limit] if limit else rows), actions, len(rows)


def partition_rows(rows, index, count):
    """Assign each annotation to exactly one worker, including uneven tails."""
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid shard index/count")
    return rows[index::count]


def manifest_rows(spec, manifest):
    shard = manifest.get("sharding")
    limit = shard["limit_per_split"] if shard else manifest["selected"]
    rows, actions, count = load_dataset(spec, limit)
    if shard:
        if shard["strategy"] != "strided-v1":
            raise ValueError("Unsupported sharding strategy")
        rows = partition_rows(rows, shard["index"], shard["count"])
    if len(rows) != manifest["selected"]:
        raise ValueError("Manifest sample count differs from annotations")
    return rows, actions, count


def resolve_video_root(base, rows):
    """Try common extraction layouts, never match a video by basename alone."""
    base = Path(base).expanduser().resolve()
    candidates = [(base, 0)]
    # Only inspect directory names, bounded to three levels. File identity remains
    # the official relative path, avoiding ambiguous same-named videos/classes.
    if base.is_dir():
        for folder, dirs, files in os.walk(base):
            rel = Path(folder).relative_to(base)
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            if len(rel.parts) >= 3:
                dirs[:] = []
            video_count = sum(Path(f).suffix.lower() in {".avi", ".webm", ".mp4", ".mkv", ".mov"} for f in files)
            if folder == str(base):
                candidates[0] = (base, video_count)
            else:
                candidates.append((Path(folder), video_count))
    probes = [rows[i]["path"] for i in sorted({i * (len(rows) - 1) // 9 for i in range(10)})]
    def rank(item):
        index, (folder, video_count) = item
        files = sum((folder / p).is_file() for p in probes)
        nested_dirs = sum((folder / Path(p).parent).is_dir() for p in probes if len(Path(p).parts) > 1)
        flat_videos = video_count if all(len(Path(p).parts) == 1 for p in probes) else 0
        return (-files, -nested_dirs, -flat_videos, index)
    chosen = sorted(enumerate(candidates), key=rank)[0][1][0]
    missing = [r["path"] for r in rows if not (chosen / r["path"]).is_file()]
    return chosen, missing


def messages(row, api):
    """Use the shared tool-free recognition prompt, with official class names.

    No tool-selection pass, external tools, class explanations, per-row
    question/solution/explanation, or text media paths reach the model.
    """
    prompt = RECOGNITION_TEMPLATE.format(
        CANDIDATE_LABELS=api["get_allowed_answers_str"](False))
    path = str(Path(api["original_video_path"]) / row["path"])
    return [{"role": "user", "content": [{"type": "video", "video": path}, {"type": "text", "text": prompt}]}]


def template_messages(messages):
    """Expose text and media type markers, never media paths or row metadata."""
    clean = []
    for message in messages:
        content = []
        for item in message["content"]:
            kind = item["type"]
            if kind == "text":
                content.append({"type": "text", "text": item["text"]})
            elif kind in ("video", "image"):
                content.append({"type": kind})
            else:
                raise ValueError("Unsupported chat content type: " + kind)
        clean.append({"role": message["role"], "content": content})
    return clean


def multimodal_inputs(batch, processor, vision_fn):
    """Preserve a one-video list for each sample's FPS metadata (vLLM 0.11)."""
    prompts = [processor.apply_chat_template(template_messages(msg), tokenize=False, add_generation_prompt=True)
               for msg in batch]
    images, videos, kwargs = vision_fn(batch, return_video_kwargs=True)
    if images or videos is None or len(videos) != len(batch):
        raise ValueError("Expected exactly one video per evaluation sample")
    result = []
    for i, prompt in enumerate(prompts):
        mm_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                if len(value) != len(batch):
                    raise ValueError("Unexpected video metadata count: " + key)
                mm_kwargs[key] = [value[i]]
            else:
                mm_kwargs[key] = value
        result.append({"prompt": prompt, "multi_modal_data": {"video": videos[i]},
                       "mm_processor_kwargs": mm_kwargs})
    return result


def summarize(results, expected, full_count):
    score = official_scorer()
    correct = invalid = errors = truncated = 0
    for row in results:
        target = score["normalize"](score["extract_answer"](row["solution"]))
        pred = score["normalize"](score["extract_answer"](row.get("predict", "")))
        correct += int(bool(target) and pred == target)
        invalid += int(not re.search(r"<answer>\s*[^<>]+?\s*</answer>", row.get("predict", ""), re.I | re.S))
        errors += int(bool(row.get("inference_error")))
        truncated += int(row.get("finish_reason") == "length")
    return {"correct": correct, "expected": expected, "completed": len(results),
            "accuracy": correct / expected if expected else 0.0,
            "complete": len(results) == expected, "official_split_size": full_count,
            "subset_only": expected != full_count, "invalid_answers": invalid,
            "inference_errors": errors, "truncated": truncated,
            "prompt_protocol": PROMPT_PROTOCOL, "inference_passes": 1}


def load_progress(path, expected_rows):
    if not Path(path).exists():
        return []
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) > len(expected_rows):
        raise ValueError("Invalid progress file: " + str(path))
    for i, row in enumerate(rows):
        if row.get("path") != expected_rows[i]["path"]:
            raise ValueError("Progress does not match annotation order at index " + str(i))
        required = ("solution", "predict", "inference_output", "inference_error", "finish_reason")
        if not all(k in row for k in required):
            raise ValueError("Incomplete progress record at index " + str(i))
        if row["solution"] != expected_rows[i]["solution"]:
            raise ValueError("Progress ground truth differs at index " + str(i))
    return rows
