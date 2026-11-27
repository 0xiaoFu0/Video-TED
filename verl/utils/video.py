"""Shared video frames, sampling rate and processor arguments for Qwen VL models."""
from __future__ import annotations

import math


def is_qwen3_video(processor):
    return "Qwen3VL" in type(processor).__name__


def video_patch_size(processor):
    component = getattr(processor, "video_processor", None) or processor.image_processor
    return int(component.patch_size)


def prepare_video_data(data, processor, *, min_pixels=None, max_pixels=None,
                       max_frames=128, video_fps=2.0):
    """Normalize paths/caches to TCHW frames, retaining the *sampled* FPS."""
    if "preprocessed_video_path" in data:
        import torch
        cache = torch.load(data["preprocessed_video_path"], map_location="cpu", weights_only=False)
        result = {"video": [cache["frames"]],
                  "video_metadatas": [cache.get("metadata")],
                  "video_fps": [cache.get("sample_fps")]}
    elif "video" in data:
        result = data
    else:
        from qwen_vl_utils.vision_process import fetch_video
        result = {"video": [], "video_metadatas": [], "video_fps": []}
        patch = video_patch_size(processor)
        for path in data["videos"]:
            info = {"video": path, "fps": video_fps or 2.0, "max_frames": max_frames or 128,
                    "min_pixels": min_pixels or 4 * (patch * 2) ** 2,
                    "max_pixels": max_pixels or 64 * (patch * 2) ** 2}
            (frames, metadata), sample_fps = fetch_video(
                info, image_patch_size=patch, return_video_sample_fps=True, return_video_metadata=True)
            result["video"].append(frames)
            result["video_metadatas"].append(metadata)
            result["video_fps"].append(float(sample_fps))

    component = getattr(processor, "video_processor", None) or processor.image_processor
    factor = video_patch_size(processor) * int(component.merge_size)
    for frames in result["video"]:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("Expected decoded video frames in TCHW format")
        if frames.shape[-2] % factor or frames.shape[-1] % factor:
            raise ValueError(f"Video size {tuple(frames.shape[-2:])} must be divisible by {factor}; regenerate the video cache for this model")
    # Validate the timing information before any of the consumers sees it.
    video_processor_kwargs(processor, result)
    return result


def video_processor_kwargs(processor, data):
    kwargs = {"do_resize": False, "do_sample_frames": False}
    count = len(data["video"])
    if is_qwen3_video(processor):
        metadata = data.get("video_metadatas")
        if metadata is None or len(metadata) != count or any(m is None for m in metadata):
            raise ValueError("Qwen3-VL requires metadata for every sampled video")
        kwargs["video_metadata"] = metadata
    elif "Qwen2_5" in type(processor).__name__:
        fps = data.get("video_fps")
        if fps is None or len(fps) != count or any(f is None or not math.isfinite(float(f)) or f <= 0 for f in fps):
            raise ValueError("Qwen2.5-VL requires the actual sampled FPS for every video")
        kwargs["fps"] = [float(f) for f in fps]
    return kwargs


def tokenize_video(processor, prompt, data):
    return processor(text=[prompt], videos=data["video"], add_special_tokens=False,
                     return_tensors="pt", **video_processor_kwargs(processor, data))


def video_model_inputs(processor, data):
    """Pixel tensors for actor/teacher; temporal positions come from tokenization."""
    kwargs = video_processor_kwargs(processor, data)
    kwargs.pop("fps", None)  # Sampling is disabled; FPS is used by the full processor for mRoPE.
    component = getattr(processor, "video_processor", None)
    if component is not None:
        return dict(component(videos=data["video"], return_tensors="pt", **kwargs))
    return dict(processor.image_processor(images=None, videos=data["video"], return_tensors="pt", **kwargs))


def vllm_video_inputs(processor, data):
    kwargs = video_processor_kwargs(processor, data)
    videos = data["video"]
    if is_qwen3_video(processor):
        from transformers.video_utils import VideoMetadata
        metadata = kwargs.pop("video_metadata")
        allowed = {"total_num_frames", "fps", "frames_indices", "video_backend", "width", "height", "duration"}
        videos = [(v, VideoMetadata(**{k: val for k, val in m.items() if k in allowed})
                   if isinstance(m, dict) else m) for v, m in zip(videos, metadata)]
    # Qwen2.5-VL takes plain frames plus fps, Qwen3-VL takes (frames, metadata).
    return {"video": videos}, kwargs
