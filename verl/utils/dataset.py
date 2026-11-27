# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for VideoTED.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .prompt_template import build_prompt
from .video import prepare_video_data, tokenize_video


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 64 * 32 * 32,
    max_frames: int = 128,
    video_fps: float = 2.0,
    return_fps: bool = False,
    image_patch_size: int = 16,
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {
        "video": video,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "fps": video_fps,
    }
    return fetch_video(
        vision_info,
        image_patch_size=image_patch_size,
        return_video_sample_fps=return_fps,
        return_video_metadata=return_fps,
    )


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        video_max_frames: int = 128,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        use_preprocessed_videos: bool = True,
        preprocessed_video_dir: Optional[str] = None,
        trajectory_key: Optional[str] = None,
        apply_chat_template_kwargs: Optional[dict] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.use_preprocessed_videos = use_preprocessed_videos
        self.preprocessed_video_dir = preprocessed_video_dir
        self.trajectory_key = trajectory_key

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_prompt_text(self, example: dict[str, Any]) -> str:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # Pass all example fields to the Jinja template to support routing by problem_type.
            # Available variables include problem, problem_type, data_type, options, and data_source.
            prompt_str = format_prompt.render(
                content=prompt_str,  # Support legacy templates.
                problem=prompt_str,  # Question text.
                **{k: v for k, v in example.items() if k != self.prompt_key}  # Other fields.
            )
        else:
            # Use build_prompt to add task-specific instructions only when format_prompt is absent.
            # If the source data already contains full instructions, build_prompt duplicates them.
            prompt_str = build_prompt(prompt_str, example)
        return prompt_str

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str = self._build_prompt_text(example)

        # Check if images exist and is a non-empty list/array
        # Support both list and numpy.ndarray inputs.
        images_data = example.get(self.image_key)
        has_images = (
            self.image_key in example
            and images_data is not None
            and hasattr(images_data, '__len__')
            and len(images_data) > 0
        )
        # Check if videos exist and is a non-empty list/array
        videos_data = example.get(self.video_key)
        has_videos = (
            self.video_key in example
            and videos_data is not None
            and hasattr(videos_data, '__len__')
            and len(videos_data) > 0
        )

        if has_images:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif has_videos:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _prepare_video(self, example):
        data = None
        if self.use_preprocessed_videos and example.get("preprocessed_video"):
            path = example["preprocessed_video"]
            if self.preprocessed_video_dir is not None:
                path = os.path.join(self.preprocessed_video_dir, path)
            if os.path.isfile(path):
                data = {"preprocessed_video_path": path}
        if data is None:
            paths = list(example[self.video_key])
            if self.image_dir is not None:
                paths = [os.path.join(self.image_dir, path) for path in paths]
            data = {"videos": paths}
        prepared = prepare_video_data(
            data, self.processor, min_pixels=self.video_min_pixels, max_pixels=self.video_max_pixels,
            max_frames=self.video_max_frames, video_fps=self.video_fps)
        # Shared caches can be reopened by workers; decoded videos retain their FPS.
        return prepared, data if "preprocessed_video_path" in data else prepared

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        # Check if images exist and is a non-empty list/array
        # Support both list and numpy.ndarray inputs.
        images_data = example.get(self.image_key)
        has_images = (
            self.image_key in example
            and images_data is not None
            and hasattr(images_data, '__len__')
            and len(images_data) > 0
        )
        # Check if videos exist and is a non-empty list/array
        videos_data = example.get(self.video_key)
        has_videos = (
            self.video_key in example
            and videos_data is not None
            and hasattr(videos_data, '__len__')
            and len(videos_data) > 0
        )

        if has_images:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif has_videos:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
            prepared, _ = self._prepare_video(example)
            model_inputs = tokenize_video(self.processor, prompt, prepared)
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **self.apply_chat_template_kwargs)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        teacher_question = example[self.prompt_key]
        prompt_text = self._build_prompt_text(example)
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        # Check if images exist and is a non-empty list/array
        # Support both list and numpy.ndarray inputs.
        images_data = example.get(self.image_key)
        has_images = (
            self.image_key in example
            and images_data is not None
            and hasattr(images_data, '__len__')
            and len(images_data) > 0
        )
        # Check if videos exist and is a non-empty list/array
        videos_data = example.get(self.video_key)
        has_videos = (
            self.video_key in example
            and videos_data is not None
            and hasattr(videos_data, '__len__')
            and len(videos_data) > 0
        )

        if has_images:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            encode = lambda text: self.processor(processed_images, [text], add_special_tokens=False, return_tensors="pt")
            example["multi_modal_data"] = {"images": images}
        elif has_videos:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
            prepared, worker_data = self._prepare_video(example)
            encode = lambda text: tokenize_video(self.processor, text, prepared)
            example["multi_modal_data"] = worker_data
        else:
            # Text-only samples with no images or videos.
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)
            encode = lambda text: self.tokenizer([text], add_special_tokens=False, return_tensors="pt")
            # Use an empty dict for text-only samples to keep batches consistent; None breaks vllm_rollout.
            example["multi_modal_data"] = {}

        model_inputs = encode(prompt)
        original_prompt_length = int(model_inputs["input_ids"].shape[-1])
        removed_text_tokens = 0
        if self.truncation == "left":
            from ..ar.prompt_length import fit_prompt_length
            prompt, model_inputs, removed_text_tokens, original_prompt_length = fit_prompt_length(
                prompt, self.tokenizer, encode, self.max_prompt_length, initial_inputs=model_inputs)
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        example["prompt_original_tokens"] = original_prompt_length
        example["prompt_removed_text_tokens"] = removed_text_tokens

        # Clean up images/videos keys if they still exist
        if self.image_key in example:
            example.pop(self.image_key, None)
        if self.video_key in example:
            example.pop(self.video_key, None)

        from .position_ids import compute_position_ids
        position_ids = compute_position_ids(input_ids, attention_mask, model_inputs, self.processor)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="error" if self.truncation == "left" else self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["raw_prompt"] = prompt  # Keep the raw text prompt for vLLM video preprocessing.
        example["prompt_text"] = prompt_text  # Text before applying the chat template, for OPSD prompt reconstruction.
        example["teacher_question"] = teacher_question  # Original question text for the teacher-specific template.
        example["ground_truth"] = example.pop(self.answer_key)

        # Mix-policy: Preserve precollected trajectory information.
        # Data format: {"has_offline_trajectory": true, "offline_output": "<think>...</think><answer>...</answer>"}
        if "has_offline_trajectory" in example and example.get("has_offline_trajectory"):
            example["has_offline_trajectory"] = True
            example["offline_output"] = example.get("offline_output", "")
        else:
            example["has_offline_trajectory"] = False
            example["offline_output"] = ""

        # OPSD: Extract the larger model trajectory for Navigator prompt generation.
        if self.trajectory_key is not None:
            example["trajectory"] = example.pop(self.trajectory_key, "")
        else:
            example["trajectory"] = ""

        # Ensure every sample has a preprocessed_video field; use None for image samples.
        # This satisfies the DataProto batch consistency check.
        if "preprocessed_video" not in example:
            example["preprocessed_video"] = None

        return example
