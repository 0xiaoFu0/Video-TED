"""Per-worker offline vLLM inference for Qwen2.5-VL and Qwen3-VL.

CUDA_VISIBLE_DEVICES must be assigned before constructing this backend. Heavy
CUDA imports are deliberately lazy so dataset checks and mock runs need no GPU.
"""
from __future__ import annotations

from typing import Any

from .media import decode_media
from .schema import Sample


def multimodal_input(video, info: dict[str, Any], *, model_type: str) -> tuple[dict, dict]:
    """Construct vLLM 0.11's model-specific pre-sampled video contract.

    Qwen3 requires a (video, metadata) tuple. Qwen2.5 requires sampled FPS for
    temporal positions. Both disable processor resampling and duplicate resize.
    See vLLM v0.11.0 model_executor/models/{qwen3_vl,qwen2_5_vl}.py.
    """
    kwargs: dict[str, Any] = {"do_sample_frames": False, "do_resize": False}
    if model_type in {"qwen3_vl", "qwen3_vl_moe"}:
        metadata = {
            "fps": info.get("timestamp_fps", info["source_fps"]),
            "total_num_frames": info.get("timestamp_total_frames", info["source_frames"]),
            "duration": info["source_duration"],
            "frames_indices": info.get("timestamp_frame_indices", info["sampled_frame_indices"]),
            "video_backend": info["decoder"], "do_sample_frames": False,
        }
        return {"video": (video, metadata)}, kwargs
    if model_type == "qwen2_5_vl":
        kwargs["fps"] = info["sampled_fps"]
        return {"video": video}, kwargs
    raise ValueError(f"Unsupported model_type {model_type!r}; expected Qwen2.5-VL or Qwen3-VL")


class VLLMBackend:
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.n = int(config.get("n", 1))
        if self.n < 1:
            raise ValueError("n must be at least 1")
        temperature = float(config.get("temperature", 0.0))
        if self.n > 1 and temperature == 0:
            raise ValueError("n > 1 requires temperature > 0; greedy replicas are identical")
        from transformers import AutoConfig, AutoProcessor
        from vllm import LLM, SamplingParams
        model_path = config["model_path"]
        trust_remote_code = bool(config.get("trust_remote_code", False))
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model_type = hf_config.model_type
        if self.model_type not in {"qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}:
            raise ValueError(f"Unsupported checkpoint architecture {self.model_type!r}")
        vision_config = getattr(hf_config, "vision_config", None)
        patch_size = int(getattr(vision_config, "patch_size", 16 if "qwen3" in self.model_type else 14))
        merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
        self.resize_factor = patch_size * merge_size
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.processor.tokenizer.padding_side = "left"
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": int(config.get("tensor_parallel_size", 1)),
            "dtype": config.get("dtype", "bfloat16"),
            "gpu_memory_utilization": float(config.get("gpu_memory_utilization", 0.8)),
            "trust_remote_code": trust_remote_code,
            "max_num_seqs": int(config.get("max_num_seqs", 1)),
            "limit_mm_per_prompt": {"image": 0, "video": 1},
            "seed": int(config.get("seed", 42)),
            # Match explicit evaluation sampling instead of inheriting a model's
            # potentially different generation_config.json defaults.
            "generation_config": "vllm",
        }
        if config.get("max_model_len"):
            llm_kwargs["max_model_len"] = int(config["max_model_len"])
        if "enforce_eager" in config:
            llm_kwargs["enforce_eager"] = bool(config["enforce_eager"])
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            n=self.n, temperature=temperature, top_p=float(config.get("top_p", 1.0)),
            repetition_penalty=float(config.get("repetition_penalty", 1.05)),
            max_tokens=int(config.get("max_new_tokens", 8192)),
            seed=int(config.get("seed", 42)),
        )

    def generate(self, sample: Sample, prompt: str, max_frames: int) -> dict[str, Any]:
        video, video_info = decode_media(
            sample, max_frames=max_frames, fps=float(self.config.get("fps", 2.0)),
            min_frames=int(self.config.get("min_frames", 4)),
            max_pixels=int(self.config.get("max_pixels", 262144)),
            min_pixels=int(self.config.get("min_pixels", 3136)),
            resize_factor=self.resize_factor, decoder=self.config.get("decoder", "decord"),
            sampling=self.config.get("video_sampling"),
        )
        messages = [{"role": "user", "content": [
            # Rendering needs only the video placeholder. A custom template
            # must not expose a filename that may encode an action or label.
            {"type": "video", "video": "video.mp4"}, {"type": "text", "text": prompt},
        ]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        multimodal_data, processor_kwargs = multimodal_input(video, video_info, model_type=self.model_type)
        outputs = self.llm.generate([{
            "prompt": rendered, "multi_modal_data": multimodal_data,
            "mm_processor_kwargs": processor_kwargs,
        }], sampling_params=self.sampling_params, use_tqdm=False)
        if len(outputs) != 1 or len(outputs[0].outputs) != self.n:
            raise RuntimeError(f"vLLM returned an unexpected completion count for {sample.uid}")
        return {
            "responses": [completion.text for completion in outputs[0].outputs],
            "finish_reasons": [completion.finish_reason for completion in outputs[0].outputs],
            "video_info": {**video_info, "model_type": self.model_type},
        }


class MockBackend:
    """Deterministic CPU-only plumbing check; never a source of model results.

    Test-only ``metadata.mock_error`` and ``metadata.mock_sleep_s`` deliberately
    exercise runner failure and timeout handling. Ground truth is never read.
    """
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)

    def generate(self, sample: Sample, prompt: str, max_frames: int) -> dict[str, Any]:
        if sample.metadata.get("mock_error"):
            raise RuntimeError(str(sample.metadata["mock_error"]))
        if sample.metadata.get("mock_sleep_s"):
            import time
            time.sleep(float(sample.metadata["mock_sleep_s"]))
        responses = sample.metadata.get("mock_responses")
        if responses is None:
            response = "<think>Mock output.</think><answer>A</answer>" if self.config.get("think_mode") else "A"
            responses = [response] * int(self.config.get("n", 1))
        if isinstance(responses, str):
            responses = [responses]
        return {"responses": list(responses), "finish_reasons": ["mock"] * len(responses),
                "video_info": {"mock": True, "media_path": sample.media_path,
                               "max_frames": max_frames}}
