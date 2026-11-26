"""Qwen3-VL inputs using the pinned v6 sampler and model-native spatial grid.

Keep upstream files unchanged. The Qwen3/vLLM interface is documented at
https://docs.vllm.ai/en/v0.11.0/api/vllm/model_executor/models/qwen3_vl.html
Metadata contains ORIGINAL fps and the actual sampled frame indices.
"""
from __future__ import annotations

import math
import warnings

from protocol import template_messages

ADAPTER_VERSION = "qwen3-v6-same-frames-native-grid-v1"


def check_runtime():
    """Check installed versions; do not install/upgrade the running environment."""
    from importlib.metadata import version
    from packaging.version import Version
    for package, minimum in (("vllm", "0.11.0"), ("transformers", "4.57.0")):
        installed = version(package)
        if Version(installed) < Version(minimum):
            raise RuntimeError(f"Qwen3-VL requires {package}>={minimum}; found {installed}. "
                               "Use EVAL_PYTHON to select a compatible environment.")


def frame_indices(legacy, total_frames, fps):
    import torch
    if total_frames < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video: total_frames={total_frames}, fps={fps}")
    count = legacy.smart_nframes({}, total_frames=total_frames, video_fps=fps)
    # Exactly the same rounding, endpoints, min/max frame count as v6.
    return torch.linspace(0, total_frames - 1, count).round().long().tolist()


def decode_video(path, legacy):
    """Preserve v6 decord -> torchvision fallback, retaining real timing data."""
    import torch
    backend = legacy.get_video_reader_backend()
    if backend not in ("decord", "torchvision"):
        raise ValueError("This v6 adapter supports decord or torchvision, got " + backend)
    if backend == "decord":
        try:
            import decord
            reader = decord.VideoReader(path)
            total, fps = len(reader), float(reader.get_avg_fps())
            indices = frame_indices(legacy, total, fps)
            frames = torch.tensor(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
        except Exception as exc:
            warnings.warn(f"decord failed ({exc}); trying torchvision, as in v6")
            backend = "torchvision"
    if backend == "torchvision":
        frames, _, info = legacy.io.read_video(
            path, start_pts=0.0, end_pts=None, pts_unit="sec", output_format="TCHW")
        total, fps = len(frames), float(info["video_fps"])
        indices = frame_indices(legacy, total, fps)
        frames = frames[indices]
    return frames, {"fps": fps, "frames_indices": indices,
                    "total_num_frames": total, "video_backend": backend}


def video_geometry(processor):
    component = processor.video_processor
    patch, merge = int(component.patch_size), int(component.merge_size)
    if (patch, merge) != (16, 2):
        raise ValueError(f"Expected Qwen3-VL patch=16, merge=2; got {patch}, {merge}")
    return patch * merge


def resize_video(frames, processor, legacy):
    """Same v6 pixel bounds and bicubic resize, rounded to the Qwen3 grid."""
    nframes, channels, height, width = frames.shape
    if channels != 3:
        raise ValueError("Expected TCHW RGB video")
    min_pixels = legacy.VIDEO_MIN_PIXELS
    max_pixels = max(min(legacy.VIDEO_MAX_PIXELS,
                         legacy.VIDEO_TOTAL_PIXELS / nframes * legacy.FRAME_FACTOR),
                     int(min_pixels * 1.05))
    size = legacy.smart_resize(height, width, factor=video_geometry(processor),
                               min_pixels=min_pixels, max_pixels=max_pixels)
    return legacy.transforms.functional.resize(
        frames, list(size), interpolation=legacy.InterpolationMode.BICUBIC,
        antialias=True).float()


def multimodal_inputs(batch, processor, legacy):
    """One (TCHW video, metadata) pair per sample; no processor resampling."""
    inputs = []
    for msg in batch:
        content = [item for turn in msg for item in turn["content"]]
        videos = [item for item in content if item["type"] == "video"]
        if len(videos) != 1 or any(item["type"] == "image" for item in content):
            raise ValueError("Expected exactly one video and no images")
        if set(videos[0]) != {"type", "video"} or not isinstance(videos[0]["video"], str):
            raise ValueError("v6 uses a full local video with fixed sampling settings")
        frames, metadata = decode_video(videos[0]["video"], legacy)
        frames = resize_video(frames, processor, legacy)
        prompt = processor.apply_chat_template(template_messages(msg), tokenize=False,
                                                add_generation_prompt=True)
        inputs.append({"prompt": prompt,
                       "multi_modal_data": {"video": [(frames, metadata)]},
                       "mm_processor_kwargs": {"do_sample_frames": False, "do_resize": False}})
    return inputs


def input_profile():
    return {"adapter": ADAPTER_VERSION, "family": "qwen3_vl",
            "fps_target": 2.0, "min_frames": 4, "max_frames": 16,
            "frame_selection": "pinned_v6_smart_nframes_and_torch_linspace_round",
            "pixel_bounds": "pinned_v6_VIDEO_MIN_MAX_TOTAL_PIXELS",
            "patch_size": 16, "spatial_merge_size": 2, "resize_factor": 32,
            "timing": "original_fps_and_sampled_frame_indices",
            "do_sample_frames": False, "do_resize": False,
            "spatial_difference_from_qwen25": "native factor 32 instead of 28; rounded size and visual token count can differ"}
