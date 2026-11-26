"""Server check of production video preprocessing and optional single-GPU backward."""
from __future__ import annotations

import argparse
import json
import math


def check_qwen3_replay(processor, prompt, encoded, mm_data, mm_kwargs, positions, config):
    """Validate native-video replay and HF mRoPE without allocating model weights."""
    from types import SimpleNamespace
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    # This is exactly the (frames, VideoMetadata) payload sent to vLLM.
    videos, metadata = zip(*mm_data['video'])
    replay = processor(text=[prompt], videos=list(videos), video_metadata=list(metadata),
                       add_special_tokens=False, return_tensors='pt', **mm_kwargs)
    for key in ('input_ids', 'pixel_values_videos', 'video_grid_thw'):
        torch.testing.assert_close(replay[key], encoded[key], rtol=0, atol=0)
    expected, _ = Qwen3VLModel.get_rope_index(
        SimpleNamespace(config=config), input_ids=encoded['input_ids'],
        attention_mask=encoded['attention_mask'], video_grid_thw=encoded['video_grid_thw'])
    torch.testing.assert_close(positions, expected[:, 0], rtol=0, atol=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', default='data/train.rlsd.think.jsonl')
    p.add_argument('--sft-data', default='data/train.sft.think.jsonl')
    p.add_argument('--samples', type=int, default=4, help='Maximum number of selected videos')
    p.add_argument('--selection', choices=('class', 'first'), default='class',
                   help='Select distinct action classes, or distinct sample IDs in input order')
    p.add_argument('--backward', action='store_true')
    p.add_argument('--max-frames', type=int, default=8)
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--fps', type=float, default=2.0)
    a = p.parse_args()
    if a.samples < 1:
        p.error('--samples must be positive')
    if a.max_frames < 1:
        p.error('--max-frames must be positive')
    if a.max_pixels < 1:
        p.error('--max-pixels must be positive')
    if not math.isfinite(a.fps) or a.fps <= 0:
        p.error('--fps must be finite and positive')

    import torch
    from ..utils.dataset import RLHFDataset
    from ..utils.tokenizer import get_processor, get_tokenizer
    from ..utils.video import tokenize_video, video_model_inputs, vllm_video_inputs
    from ..trainer.opsd_algos import _tokenize_prompt
    from .prepare_data import read_rows, validate_rows

    processor = get_processor(a.model, use_fast=True, local_files_only=True)
    tokenizer = get_tokenizer(a.model, use_fast=True, local_files_only=True)
    rows = read_rows(a.data)
    validate_rows(rows, check_media=True)
    dataset = RLHFDataset(
        a.data, tokenizer, processor, max_prompt_length=16384, truncation='error',
        format_prompt='examples/action_recognition/student.jinja', filter_overlong_prompts=False,
        use_preprocessed_videos=False, video_fps=a.fps, video_max_frames=a.max_frames,
        video_min_pixels=a.max_pixels, video_max_pixels=a.max_pixels)
    seen = set()
    selected = []
    for i, row in enumerate(rows):
        key = row['answer'] if a.selection == 'class' else row['metadata']['sample_id']
        if key not in seen:
            seen.add(key)
            selected.append(i)
        if len(selected) == a.samples:
            break
    for i in selected:
        item = dataset[i]
        data = item['multi_modal_data']
        encoded = tokenize_video(processor, item['raw_prompt'], data)
        worker_inputs = video_model_inputs(processor, data)
        for key in ('pixel_values_videos', 'video_grid_thw'):
            torch.testing.assert_close(worker_inputs[key], encoded[key], rtol=0, atol=0)
        ids = item['input_ids'][item['attention_mask'].bool()]
        assert torch.equal(ids, encoded['input_ids'][0]), 'Dataset/processor token mismatch'
        teacher_ids, teacher_mask, teacher_positions, _, _ = _tokenize_prompt(
            item['raw_prompt'], tokenizer, processor, data, max_prompt_length=16384,
            strict_prompt_length=True)
        torch.testing.assert_close(teacher_ids[teacher_mask.bool()], ids, rtol=0, atol=0)
        torch.testing.assert_close(
            teacher_positions[:, teacher_mask.bool()],
            item['position_ids'][:, item['attention_mask'].bool()], rtol=0, atol=0)
        mm_data, mm_kwargs = vllm_video_inputs(processor, data)
        if 'Qwen2_5' in type(processor).__name__:
            # Check the arguments sent to vLLM using the same HF processor.
            rollout_encoded = processor(text=[item['raw_prompt']], videos=mm_data['video'],
                                        add_special_tokens=False, return_tensors='pt', **mm_kwargs)
            assert rollout_encoded['second_per_grid_ts'] == encoded['second_per_grid_ts']
            torch.testing.assert_close(rollout_encoded['pixel_values_videos'], encoded['pixel_values_videos'], rtol=0, atol=0)
            from transformers import AutoConfig
            from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
            config = AutoConfig.from_pretrained(a.model, local_files_only=True)
            positions, _ = MRotaryEmbedding.get_input_positions_tensor(
                input_tokens=ids.tolist(), hf_config=config, image_grid_thw=[],
                video_grid_thw=encoded['video_grid_thw'].tolist(),
                second_per_grid_ts=encoded['second_per_grid_ts'])
            torch.testing.assert_close(positions, item['position_ids'][1:, item['attention_mask'].bool()], rtol=0, atol=0)
        elif 'Qwen3VL' in type(processor).__name__:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(a.model, local_files_only=True)
            check_qwen3_replay(
                processor, item['raw_prompt'], encoded, mm_data, mm_kwargs,
                item['position_ids'][1:, item['attention_mask'].bool()], config)
        print(json.dumps({'sample': rows[i]['metadata']['sample_id'], 'answer': rows[i]['answer'],
                          'frames': [list(v.shape) for v in data['video']], 'sample_fps': data['video_fps'],
                          'grid': encoded['video_grid_thw'].tolist(), 'prompt_tokens': ids.numel(),
                          'second_per_grid_ts': encoded.get('second_per_grid_ts')}, ensure_ascii=False, default=lambda x: x.tolist()), flush=True)

    if a.backward:
        if not torch.cuda.is_available():
            raise RuntimeError('--backward requires a CUDA GPU')
        from transformers import AutoModelForImageTextToText
        from .sft import VideoCollator
        sft_rows = {r['metadata']['sample_id']: r for r in read_rows(a.sft_data)}
        sample = sft_rows[rows[selected[0]]['metadata']['sample_id']]
        batch = VideoCollator(processor, 16384, a.max_frames, a.fps, a.max_pixels)([sample])
        model = AutoModelForImageTextToText.from_pretrained(
            a.model, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2', device_map={'': 'cuda:0'})
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        batch = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite forward loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError('Invalid backward gradient norm')
        print(f'GPU forward/backward passed: loss={loss.item():.6f}, grad_norm={norm.item():.6f}', flush=True)
        model.zero_grad(set_to_none=True)
    print(f'Video preprocessing passed for {len(selected)} videos. vLLM generation is tested by the RLSD smoke run.', flush=True)


if __name__ == '__main__':
    main()
