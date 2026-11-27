"""Position IDs shared by AR student and teacher prompt preprocessing."""

from typing import Any

import torch


def compute_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_inputs: dict[str, Any],
    processor: Any,
) -> torch.Tensor:
    """Return veRL's text row plus three multimodal RoPE rows for Qwen-VL.

    Qwen3-VL currently reuses Qwen2's image processor. Detect the main
    processor first so wrappers/subclasses do not silently get 1D text RoPE
    when that image-processor implementation changes.
    """
    processor_names = () if processor is None else tuple(cls.__name__ for cls in type(processor).__mro__)
    qwen3 = any("Qwen3VLProcessor" in name for name in processor_names)
    qwen2 = any(name in ("Qwen2VLProcessor", "Qwen2_5_VLProcessor") for name in processor_names)
    image_processor = getattr(processor, "image_processor", None)
    qwen_image_processor = image_processor is not None and any(
        "Qwen2VLImageProcessor" in cls.__name__ for cls in type(image_processor).__mro__
    )
    if qwen3 or qwen2 or qwen_image_processor:
        if qwen3:
            from ..models.transformers.qwen3_vl import get_rope_index
        else:
            from ..models.transformers.qwen2_vl import get_rope_index

        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
            attention_mask=attention_mask,
        )
        text_position_ids = torch.arange(len(input_ids), device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
        return torch.cat((text_position_ids, vision_position_ids), dim=0)
    return torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)
