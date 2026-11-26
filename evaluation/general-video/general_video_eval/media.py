"""Bounded indexed video decoding and auditable temporal sampling.

Only selected frames are decoded. In particular, failures never fall back to
torchvision's whole-file decoder. All returned arrays are RGB uint8 THWC.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from .schema import Sample


def smart_resize(height: int, width: int, *, factor: int = 28,
                 min_pixels: int = 3136, max_pixels: int = 262144) -> tuple[int, int]:
    """Qwen's aspect-preserving spatial budget, applied once before vLLM."""
    if min(height, width, factor, min_pixels, max_pixels) <= 0:
        raise ValueError("Image dimensions, factor, and pixel limits must be positive")
    if max_pixels < min_pixels or max_pixels < factor * factor:
        raise ValueError("max_pixels must cover min_pixels and one merged vision patch")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Unsupported frame aspect ratio: {width}x{height}")
    resized_h = max(factor, round(height / factor) * factor)
    resized_w = max(factor, round(width / factor) * factor)
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        resized_h = max(factor, math.floor(height / beta / factor) * factor)
        resized_w = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = max(factor, math.ceil(height * beta / factor) * factor)
        resized_w = max(factor, math.ceil(width * beta / factor) * factor)
    # The minimum factor constraint can otherwise push a very thin image over
    # the budget. Fail explicitly rather than quietly changing the pixel cap.
    if resized_h * resized_w > max_pixels:
        raise ValueError(f"Frame aspect ratio cannot meet the pixel budget: {width}x{height}")
    return resized_h, resized_w


def frame_plan(total_frames: int, source_fps: float, *, max_frames: int = 64,
               target_fps: float = 2.0, min_frames: int = 4,
               start: float | None = None, end: float | None = None,
               sampling: str = "uniform", first_index: int = 0) -> dict[str, Any]:
    """Return zero-based decoder indices, including intentional repeats.

    MVBench follows its released segment-midpoint formula, including the
    one-based frame-directory convention. ``uniform`` keeps the original
    nearest-even FPS budget. ``uniform_fps_floor`` follows Video-R1's floor-even
    FPS budget and caps normal clips at their available source frame count.
    Both uniform modes include clip endpoints. ``max_frames`` is a ceiling,
    not a fixed frame count; very short clips repeat frames to meet min_frames.
    """
    if total_frames < 1 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid source video metadata: frames={total_frames}, fps={source_fps}")
    if max_frames < 2 or max_frames % 2 or min_frames < 2 or min_frames % 2:
        raise ValueError("max_frames and min_frames must be positive even integers >= 2")
    if min_frames > max_frames or not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("Require min_frames <= max_frames and target_fps > 0")
    if start is not None and (not math.isfinite(start) or start < 0):
        raise ValueError(f"Invalid clip start: {start}")
    if end is not None and (not math.isfinite(end) or end < (start or 0)):
        raise ValueError(f"Invalid clip end: {end}")
    last_index = first_index + total_frames - 1
    lower = max(first_index, round((start or 0) * source_fps))
    upper = last_index if end is None else min(last_index, round(end * source_fps))
    if lower > upper:
        raise ValueError(f"Clip [{start}, {end}] contains no frames in {total_frames/source_fps:.3f}s video")
    span = upper - lower + 1
    if sampling == "mvbench":
        segment_size = (upper - lower) / max_frames
        selected = [int(lower + segment_size / 2 + np.round(segment_size * i))
                    for i in range(max_frames)]
    elif sampling == "uniform":
        requested = round((span / source_fps * target_fps) / 2) * 2
        count = min(max_frames, max(min_frames, requested))
        selected = np.linspace(lower, upper, count).round().astype(int).tolist()
    elif sampling == "uniform_fps_floor":
        # Video-R1 smart_nframes clamps its FPS budget, then floors to the
        # temporal patch factor. Preserve this evaluator's short-clip support
        # by repeating available frames when there are fewer than min_frames.
        requested = span / source_fps * target_fps
        capped = min(max(requested, min_frames), max_frames, span)
        count = max(min_frames, math.floor(capped / 2) * 2)
        selected = np.linspace(lower, upper, count).round().astype(int).tolist()
    else:
        raise ValueError(f"Unknown frame sampling protocol: {sampling}")
    indices = [i - first_index for i in selected]
    if any(i < 0 or i >= total_frames for i in indices):
        raise AssertionError(f"Sampler produced an out-of-bounds frame index: {indices}")
    return {
        "source_frames": int(total_frames), "source_fps": float(source_fps),
        "source_duration": float(total_frames / source_fps),
        "sampled_frame_indices": indices,
        "sampled_timestamps": [float(i / source_fps) for i in indices],
        "sampled_frames": len(indices), "sampled_fps": float(len(indices) / span * source_fps),
        "input_start_frame": lower - first_index,
        "input_end_frame": upper - first_index,
        "start": start, "end": end, "sampling": sampling,
    }


def frame_files(directory: str | Path) -> list[Path]:
    paths = [p for p in Path(directory).iterdir()
             if p.is_file() and p.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not paths:
        raise ValueError(f"Frame directory has no image files: {directory}")
    numbers = [re.search(r"(\d+)$", p.stem) for p in paths]
    if all(numbers):
        ordered = sorted((int(m.group(1)), p) for m, p in zip(numbers, paths))
        actual = [number for number, _ in ordered]
        expected = list(range(actual[0], actual[0] + len(actual)))
        if actual != expected:
            raise ValueError(f"Frame directory has duplicate/missing numeric frames: {directory}")
        return [p for _, p in ordered]
    return sorted(paths)


def _resize_frame(frame: np.ndarray | Image.Image, size: tuple[int, int]) -> np.ndarray:
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    image = image.convert("RGB")
    height, width = size
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8).copy()


def _seedbench_frame_files(sample: Sample, *, max_frames: int) -> list[Path]:
    """Validate the explicit, ordered eight-frame pack without inferring timing."""
    if sample.media_type != "frames" or sample.metadata.get("sampling") != "seedbench_official_8":
        raise ValueError(f"{sample.uid}: SEED-Bench requires the official eight-frame directory protocol")
    if max_frames != 8:
        raise ValueError(f"{sample.uid}: SEED-Bench official frames require max_frames=8")
    if sample.start is not None or sample.end is not None:
        raise ValueError(f"{sample.uid}: SEED-Bench official frames must not be clipped again")
    if sample.frame_fps != 2.0:
        raise ValueError(f"{sample.uid}: SEED-Bench requires the synthetic ordinal frame_fps=2.0")
    names = sample.metadata.get("frame_files")
    if not isinstance(names, (list, tuple)) or len(names) != 8:
        raise ValueError(f"{sample.uid}: SEED-Bench frame_files must list exactly 8 images")
    root = Path(sample.media_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{sample.uid}: SEED-Bench media_path must be a frame directory")
    files, seen = [], set()
    for name in names:
        if (not isinstance(name, str) or not name or "\x00" in name or "\\" in name
                or Path(name).is_absolute() or ".." in Path(name).parts):
            raise ValueError(f"{sample.uid}: unsafe relative SEED-Bench frame path: {name!r}")
        resolved = (root / name).resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"{sample.uid}: SEED-Bench frame path escapes its directory: {name!r}")
        if resolved in seen:
            raise ValueError(f"{sample.uid}: duplicate SEED-Bench frame path: {name!r}")
        if not resolved.is_file():
            raise ValueError(f"{sample.uid}: SEED-Bench frame is not a file: {name!r}")
        seen.add(resolved)
        files.append(resolved)
    return files


def decode_media(sample: Sample, *, max_frames: int, fps: float = 2.0,
                 min_frames: int = 4, max_pixels: int = 262144,
                 min_pixels: int = 3136, resize_factor: int = 28,
                 decoder: str = "decord", sampling: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode at most one native-resolution frame at a time, then resize.

    ``decoder='opencv'`` is an explicit compatibility option, never a silent
    fallback after a Decord error. GIFs use Pillow; frame directories use the
    source FPS recorded in the benchmark loader (3 FPS for MVBench TVQA).
    ``sampling=None`` or ``'benchmark_default'`` keeps the existing benchmark
    protocol; an explicit sampler changes frame selection within the same
    annotated clip bounds and preserves source timing metadata.

    SEED-Bench's official pack is already sampled. Its explicit eight files are
    decoded in order, regardless of ``sampling``, ``fps`` or ``min_frames``.
    The pack does not provide source timestamps: its 2 FPS clock and 4 second
    duration are synthetic ordinal positions for the model interface, not
    measurements of the source video. Returned metadata labels this explicitly.
    """
    path = Path(sample.media_path)
    if not path.exists():
        raise FileNotFoundError(f"{sample.uid}: missing media: {path}")
    if sampling is None or sampling == "benchmark_default":
        sampling = "mvbench" if sample.benchmark == "mvbench" else "uniform"
    common = dict(max_frames=max_frames, target_fps=fps, min_frames=min_frames,
                  start=sample.start, end=sample.end, sampling=sampling)

    def dimensions(frame):
        array = np.asarray(frame)
        return smart_resize(array.shape[0], array.shape[1], factor=resize_factor,
                            min_pixels=min_pixels, max_pixels=max_pixels)

    if sample.benchmark == "seedbench" or sample.metadata.get("sampling") == "seedbench_official_8":
        files = _seedbench_frame_files(sample, max_frames=max_frames)
        frames, shape = [], None
        for file in files:
            with Image.open(file) as image:
                image = image.convert("RGB")
                if shape is None:
                    shape = dimensions(image)
                frames.append(_resize_frame(image, shape))
        # These interface fields describe the ordinal frame sequence only.
        # They must never be treated as recovered source FPS or clip duration.
        info = {
            "source_frames": 8, "source_fps": 2.0, "source_duration": 4.0,
            "sampled_frame_indices": list(range(8)),
            "sampled_timestamps": [index / 2.0 for index in range(8)],
            "sampled_frames": 8, "sampled_fps": 2.0, "frames": 8,
            "input_start_frame": 0, "input_end_frame": 7,
            "start": None, "end": None, "sampling": "seedbench_official_8",
            "timing_source": "synthetic_ordinal_2fps", "source_timing_known": False,
            "timing_is_synthetic": True,
            "frame_files": list(sample.metadata["frame_files"]),
        }
        used_decoder = "pillow_frames"
    elif sample.media_type == "frames":
        files = frame_files(path)
        # The tele/MVBench reference uses first_idx=1 for frame directories.
        first_index = int(sample.metadata.get("first_index", 1))
        info = frame_plan(len(files), sample.frame_fps, first_index=first_index, **common)
        with Image.open(files[0]) as image:
            shape = dimensions(image.convert("RGB"))
        frames = []
        for index in info["sampled_frame_indices"]:
            with Image.open(files[index]) as image:
                frames.append(_resize_frame(image, shape))
        used_decoder = "pillow_frames"
    elif sample.media_type == "gif":
        with Image.open(path) as image:
            count = image.n_frames
            durations = []
            for index in range(count):
                image.seek(index)
                durations.append(float(image.info.get("duration", 100)) / 1000)
            if any(duration <= 0 for duration in durations):
                raise ValueError(f"{sample.uid}: GIF has nonpositive frame duration")
            source_fps = count / sum(durations)
            info = frame_plan(count, source_fps, **common)
            image.seek(0)
            shape = dimensions(image.convert("RGB"))
            frames = []
            for index in info["sampled_frame_indices"]:
                image.seek(index)
                frames.append(_resize_frame(image, shape))
            # GIF indices follow MVBench's fixed frame-index protocol. The
            # timestamps preserve actual variable frame durations for Qwen3.
            frame_times = np.concatenate(([0.0], np.cumsum(durations[:-1])))
            info["sampled_timestamps"] = [float(frame_times[i]) for i in info["sampled_frame_indices"]]
            # VideoMetadata accepts frame indices rather than arbitrary
            # timestamps. A millisecond timebase represents GIF timing without
            # replacing variable durations with their average FPS.
            info["timestamp_fps"] = 1000.0
            info["timestamp_total_frames"] = int(round(sum(durations) * 1000))
            info["timestamp_frame_indices"] = [int(round(t * 1000)) for t in info["sampled_timestamps"]]
        used_decoder = "pillow_gif"
    elif decoder == "decord":
        try:
            from decord import VideoReader, cpu
        except ImportError as exc:
            raise ImportError("Install decord==0.6.0, or explicitly select --decoder opencv") from exc
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1, fault_tol=0)
        try:
            info = frame_plan(len(reader), float(reader.get_avg_fps()), **common)
            first_frame = reader[info["sampled_frame_indices"][0]].asnumpy()
            shape = dimensions(first_frame)
            del first_frame
            frames = []
            for index in info["sampled_frame_indices"]:
                frames.append(_resize_frame(reader[index].asnumpy(), shape))
        finally:
            del reader
        used_decoder = "decord"
    elif decoder == "opencv":
        import cv2
        reader = cv2.VideoCapture(str(path))
        try:
            if not reader.isOpened():
                raise ValueError(f"OpenCV cannot open video: {path}")
            info = frame_plan(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)),
                              float(reader.get(cv2.CAP_PROP_FPS)), **common)
            frames, shape = [], None
            for index in info["sampled_frame_indices"]:
                reader.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, bgr = reader.read()
                if not success:
                    raise ValueError(f"OpenCV failed at requested frame {index}: {path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if shape is None:
                    shape = dimensions(rgb)
                frames.append(_resize_frame(rgb, shape))
        finally:
            reader.release()
        used_decoder = "opencv"
    else:
        raise ValueError(f"Unknown decoder {decoder!r}; choose decord or opencv")
    video = np.stack(frames)
    info.update(decoder=used_decoder, frame_height=int(video.shape[1]),
                frame_width=int(video.shape[2]), resize="pillow_bicubic",
                max_pixels=max_pixels, min_pixels=min_pixels,
                resize_factor=resize_factor, media_path=str(path))
    return video, info
