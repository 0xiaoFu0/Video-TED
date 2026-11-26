"""CPU regressions for production video routing; decoder/GPU runtime is checked on server."""
from __future__ import annotations

import ast
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch
import transformers
from jinja2 import Template
from omegaconf import OmegaConf

from verl.ar.sft import sft_wrap_classes
from verl.ar.validation import validate_training_config
from verl.trainer.config import PPOConfig
from verl.utils import torch_functional as VF
from verl.utils.video import (prepare_video_data, tokenize_video, video_model_inputs,
                              video_processor_kwargs, vllm_video_inputs)

ROOT = Path(__file__).resolve().parents[2]
HF = Path(transformers.__file__).parent


def source_function(path, name, namespace, class_name=None):
    """Run unmodified production bodies without importing optional CUDA dependencies."""
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def hf_video_preprocess():
    ns = dict(torch=torch, defaultdict=defaultdict)
    for name in ('group_videos_by_shape', 'reorder_videos'):
        source_function(HF / 'video_utils.py', name, ns)
    ns.update(get_image_size=lambda frame, **kw: frame.shape[-2:], ChannelDimension=NS(FIRST='channels_first'),
              BatchFeature=lambda data, **kw: data)
    return source_function(HF / 'models/qwen2_vl/video_processing_qwen2_vl.py', '_preprocess', ns, 'Qwen2VLVideoProcessor')


class VideoProcessor:
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2

    def rescale_and_normalize(self, video, *args):
        return video  # Resizing/normalization is outside this shape regression.

    def __call__(self, videos, **kwargs):
        return hf_video_preprocess()(self, videos=videos, do_resize=False, size={}, interpolation=None,
                                    do_rescale=False, rescale_factor=1., do_normalize=False,
                                    image_mean=None, image_std=None, patch_size=self.patch_size,
                                    temporal_patch_size=2, merge_size=2, return_tensors='pt')


class Tokenizer:
    pad_token_id = 0
    def encode(self, text, **kw): return [1, 2]
    def convert_tokens_to_ids(self, text):
        return {'<|image_pad|>': 7, '<|video_pad|>': 8, '<|vision_start|>': 9}[text]


class Qwen2_5_VLProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.image_processor = type('Qwen2VLImageProcessorFast', (), {'merge_size': 2, 'patch_size': 14})()
        self.tokenizer = Tokenizer()
        self.video_tokens_per_second = 4
        self.last_kwargs = None

    def apply_chat_template(self, *args, **kw): return '<video> question'

    def __call__(self, text, videos, **kwargs):
        self.last_kwargs = kwargs
        result = self.video_processor(videos)
        count = int(result['video_grid_thw'].prod(dim=-1).sum()) // 4
        ids = torch.tensor([[9] + [8] * count + [10, 11]])
        result.update(input_ids=ids, attention_mask=torch.ones_like(ids),
                      second_per_grid_ts=[2 / f for f in kwargs['fps']])
        return result


@pytest.fixture
def video_runtime(monkeypatch):
    calls = []
    def fetch(info, image_patch_size, **kwargs):
        calls.append((info, image_patch_size, kwargs))
        frames = torch.arange(4 * 3 * (image_patch_size * 4) ** 2, dtype=torch.float).reshape(4, 3, image_patch_size * 4, image_patch_size * 4)
        return (frames, {'fps': 30., 'total_num_frames': 75, 'frames_indices': [0, 24, 49, 74]}), 1.6
    package = ModuleType('qwen_vl_utils')
    decoder = ModuleType('qwen_vl_utils.vision_process')
    decoder.fetch_video = fetch
    monkeypatch.setitem(sys.modules, 'qwen_vl_utils', package)
    monkeypatch.setitem(sys.modules, 'qwen_vl_utils.vision_process', decoder)
    rope = ModuleType('verl.models.transformers.qwen2_vl')
    rope.get_rope_index = source_function(ROOT / 'verl/models/transformers/qwen2_vl.py', 'get_rope_index', dict(torch=torch))
    monkeypatch.setitem(sys.modules, rope.__name__, rope)
    return calls, rope.get_rope_index


def test_qwen25_decode_worker_rollout_share_frames_and_fractional_fps(video_runtime):
    processor = Qwen2_5_VLProcessor()
    prepared = prepare_video_data({'videos': ['sample.avi']}, processor, max_frames=8)
    assert video_runtime[0][0][1] == 14
    encoded = tokenize_video(processor, '<video>', prepared)
    assert encoded['second_per_grid_ts'] == [1.25]
    assert 'video_metadata' not in processor.last_kwargs
    worker = video_model_inputs(processor, prepared)
    torch.testing.assert_close(worker['pixel_values_videos'], encoded['pixel_values_videos'])
    mm, kwargs = vllm_video_inputs(processor, prepared)
    assert mm['video'][0] is prepared['video'][0]
    assert kwargs['fps'] == [1.6] and kwargs['do_resize'] is False


def test_video_check_logs_tensor_timestamps(capsys):
    # HF converts timestamps to tensors when the processor uses return_tensors='pt'.
    from transformers.feature_extraction_utils import BatchFeature

    encoded = BatchFeature(data={'video_grid_thw': [[2, 4, 4]],
                                 'second_per_grid_ts': [1.25]}, tensor_type='pt')
    path = ROOT / 'verl/ar/check_video.py'
    tree = ast.parse(path.read_text())
    report = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print'
                  and node.value.args and isinstance(node.value.args[0], ast.Call)
                  and isinstance(node.value.args[0].func, ast.Attribute)
                  and node.value.args[0].func.attr == 'dumps')
    namespace = dict(json=json, rows=[{'metadata': {'sample_id': 'sample'}, 'answer': 'brush_hair'}],
                     i=0, data={'video': [torch.zeros(4, 3, 56, 56)], 'video_fps': [1.6]},
                     encoded=encoded, ids=torch.arange(10))
    exec(compile(ast.Module(body=[report], type_ignores=[]), str(path), 'exec'), namespace)
    result = json.loads(capsys.readouterr().out)
    assert result['second_per_grid_ts'] == [1.25]
    assert result['grid'] == [[2, 4, 4]] and result['prompt_tokens'] == 10
    assert result['sample'] == 'sample' and result['frames'] == [[4, 3, 56, 56]]


def video_check_cli(argv, monkeypatch):
    """Run the actual CLI prefix before optional model/decoder imports."""
    path = ROOT / 'verl/ar/check_video.py'
    main = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    prefix = []
    for node in main.body:
        if isinstance(node, ast.Import) and any(a.name == 'torch' for a in node.names):
            break
        prefix.append(node)
    else:
        raise AssertionError('CLI should validate input before importing torch')
    prefix.append(ast.Assign(targets=[ast.Name(id='parsed', ctx=ast.Store())],
                             value=ast.Name(id='a', ctx=ast.Load())))
    namespace = dict(argparse=argparse, math=math, __doc__='Video check')
    monkeypatch.setattr(sys, 'argv', ['check_video', '--model', '/test/model', *argv])
    module = ast.Module(body=prefix, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['parsed'], main


@pytest.mark.parametrize('argv,expected', [
    ([], (8, 262144, 2.0)),
    (['--max-frames', '16', '--max-pixels', '100352', '--fps', '2.5'], (16, 100352, 2.5)),
])
def test_video_check_cli_settings_reach_forward_and_backward(argv, expected, monkeypatch):
    args, main = video_check_cli(argv, monkeypatch)
    frames, pixels, fps = expected
    captured = {}

    def dataset(*positional, **keywords):
        captured['dataset'] = keywords
        return object()

    def collator(processor, max_length, max_frames, video_fps, max_pixels):
        captured['collator'] = (max_frames, max_pixels, video_fps)
        return lambda rows: {}

    forward = next(n for n in ast.walk(main)
                   if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Name) and n.value.func.id == 'RLHFDataset')
    backward = next(n for n in ast.walk(main)
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Call)
                    and isinstance(n.value.func.func, ast.Name)
                    and n.value.func.func.id == 'VideoCollator')
    namespace = dict(a=args, tokenizer=object(), processor=object(), sample={},
                     RLHFDataset=dataset, VideoCollator=collator)
    for call in (forward.value, backward.value):
        module = ast.Module(body=[ast.Expr(value=call)], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, '<production video constructors>', 'exec'), namespace)
    assert captured['dataset']['video_max_frames'] == frames
    assert captured['dataset']['video_fps'] == fps
    assert captured['dataset']['video_min_pixels'] == pixels
    assert captured['dataset']['video_max_pixels'] == pixels
    assert captured['collator'] == expected


@pytest.mark.parametrize('selection,expected', [('class', [0]), ('first', [0, 2, 3, 4])])
def test_video_check_selection_keeps_same_label_source_probes(selection, expected, monkeypatch):
    args, main = video_check_cli(['--selection', selection, '--samples', '4'], monkeypatch)
    loop = next(n for n in main.body if isinstance(n, ast.For)
                and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name)
                and n.iter.func.id == 'enumerate')
    namespace = dict(a=args, rows=[
        {'answer': 'pushup', 'metadata': {'sample_id': sid}}
        for sid in ('v1', 'v1', 'llava', 'k700', 'ssv2')
    ], seen=set(), selected=[])
    exec(compile(ast.Module(body=[loop], type_ignores=[]), '<production selection>', 'exec'), namespace)
    assert namespace['selected'] == expected


@pytest.mark.parametrize('argv,flag', [
    (['--max-frames', '0'], '--max-frames'),
    (['--max-frames', '-1'], '--max-frames'),
    (['--max-pixels', '0'], '--max-pixels'),
    (['--max-pixels', '-1'], '--max-pixels'),
    (['--fps', '0'], '--fps'),
    (['--fps', '-1'], '--fps'),
    (['--fps', 'nan'], '--fps'),
    (['--fps', 'inf'], '--fps'),
])
def test_video_check_rejects_invalid_sampling_before_model_import(argv, flag, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        video_check_cli(argv, monkeypatch)
    assert exc.value.code == 2
    assert flag in capsys.readouterr().err


def test_transformers_patch_reshape_accepts_504_rejects_old_512():
    processor = VideoProcessor()
    result = processor([torch.zeros(2, 3, 504, 504)])
    assert result['video_grid_thw'].tolist() == [[1, 36, 36]]
    with pytest.raises(RuntimeError):
        processor([torch.zeros(2, 3, 512, 512)])


def test_cache_preserves_fps_and_rejects_incompatible_frames(tmp_path, video_runtime):
    processor = Qwen2_5_VLProcessor()
    prepared = prepare_video_data({'videos': ['sample.avi']}, processor)
    path = tmp_path / 'video.pt'
    torch.save({'frames': prepared['video'][0], 'metadata': prepared['video_metadatas'][0], 'sample_fps': 1.6}, path)
    loaded = prepare_video_data({'preprocessed_video_path': str(path)}, processor)
    assert loaded['video_fps'] == [1.6]
    with pytest.raises(ValueError, match='sampled FPS'):
        prepare_video_data({'video': prepared['video']}, processor)
    with pytest.raises(ValueError, match='divisible by 28'):
        prepare_video_data({'video': [torch.zeros(2, 3, 64, 64)], 'video_fps': [2.]}, processor)


@pytest.mark.parametrize('rate', [2, 4, 25])
def test_temporal_positions_keep_fractional_time_and_model_rate(video_runtime, rate):
    processor = Qwen2_5_VLProcessor()
    processor.video_tokens_per_second = rate
    ids = torch.tensor([9] + [8] * 12 + [10, 11])
    positions = video_runtime[1](processor, ids, video_grid_thw=torch.tensor([[3, 4, 4]]), second_per_grid_ts=[1.25])
    expected = [1] * 4 + [1 + int(1.25 * rate)] * 4 + [1 + int(2.5 * rate)] * 4
    assert positions[0, 1:13].tolist() == expected


def test_qwen3_retains_metadata_route(video_runtime):
    processor = type('Qwen3VLProcessor', (), {})()
    processor.video_processor = NS(patch_size=16, merge_size=2)
    prepared = prepare_video_data({'videos': ['sample.avi']}, processor)
    assert video_runtime[0][0][1] == 16
    kwargs = video_processor_kwargs(processor, prepared)
    assert kwargs['video_metadata'][0]['fps'] == 30.
    assert 'fps' not in kwargs


def test_teacher_and_fsdp_use_shared_video_contract(video_runtime):
    processor = Qwen2_5_VLProcessor()
    prepared = prepare_video_data({'videos': ['sample.avi']}, processor)
    ns = dict(torch=torch, VF=VF, prepare_video_data=prepare_video_data, tokenize_video=tokenize_video,
              __name__='verl.trainer.test_video', __package__='verl.trainer')
    source_function(ROOT / 'verl/trainer/opsd_algos.py', '_compute_position_ids', ns)
    tokenize = source_function(ROOT / 'verl/trainer/opsd_algos.py', '_tokenize_prompt', ns)
    ids, mask, positions, _, _ = tokenize('<video> teacher', processor.tokenizer, processor, prepared, 32)
    assert processor.last_kwargs['fps'] == [1.6]
    assert positions.shape == (4, 32)
    meta = dict(image_min_pixels=None, image_max_pixels=None, video_min_pixels=None,
                video_max_pixels=None, video_fps=2., video_max_frames=8)
    batch = NS(meta_info=meta, non_tensor_batch={'uid': np.array(['a', 'a']), 'multi_modal_data': [prepared, prepared]})
    worker = NS(_cache={}, processor=processor)
    fn = source_function(ROOT / 'verl/workers/fsdp_workers.py', '_build_multi_modal_inputs',
                         dict(np=np, prepare_video_data=prepare_video_data, video_model_inputs=video_model_inputs), 'FSDPWorker')
    fn(worker, batch, 'uid', 'multi_modal_data', 'multi_modal_inputs')
    assert len(batch.non_tensor_batch['multi_modal_inputs']) == 2
    torch.testing.assert_close(batch.non_tensor_batch['multi_modal_inputs'][0]['video_grid_thw'], torch.tensor([[2, 4, 4]]))


def test_dataset_filter_and_getitem_and_rollout_entry(video_runtime):
    class Rows:
        def __getitem__(self, index):
            return {'prompt': '<video> question', 'videos': ['sample.avi'], 'answer': 'pushup'}
    tree = ast.parse((ROOT / 'verl/utils/dataset.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RLHFDataset')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = dict(torch=torch, os=os, Dataset=object, Template=Template, VF=VF,
              load_dataset=lambda *args, **kw: Rows(), prepare_video_data=prepare_video_data,
              tokenize_video=tokenize_video, __name__='verl.utils.test_dataset', __package__='verl.utils')
    exec(compile(module, str(ROOT / 'verl/utils/dataset.py'), 'exec'), ns)
    processor = Qwen2_5_VLProcessor()
    dataset = ns['RLHFDataset'](str(ROOT / 'data/train.rlsd.think.jsonl'), processor.tokenizer, processor,
        max_prompt_length=32, filter_overlong_prompts=False, use_preprocessed_videos=False,
        format_prompt=str(ROOT / 'examples/action_recognition/student.jinja'))
    assert dataset._filter_overlong_prompts(Rows()[0])
    item = dataset[0]
    assert item['multi_modal_data']['video_fps'] == [1.6]
    assert item['position_ids'].shape == (4, 32)
    rollout = source_function(ROOT / 'verl/workers/rollout/vllm_rollout_spmd.py', '_process_multi_modal_data',
                              dict(prepare_video_data=prepare_video_data, vllm_video_inputs=vllm_video_inputs))
    mm, kwargs = rollout(item['multi_modal_data'], None, None, None, None, 8, 2., processor)
    assert kwargs['fps'] == [1.6]
    assert mm['video'][0] is item['multi_modal_data']['video'][0]
    assert rollout({'images': [], 'videos': []}, None, None, None, None, 8, 2., processor) == (None, None)
    dataset.max_prompt_length = 8
    assert not dataset._filter_overlong_prompts(Rows()[0])


def test_sft_masks_user_and_supplies_matching_temporal_positions(video_runtime, monkeypatch):
    from verl.ar.sft import VideoCollator
    original_call = Qwen2_5_VLProcessor.__call__
    def call(self, text, **kwargs):
        result = original_call(self, text, **kwargs)
        if text == ['full']:
            result['input_ids'] = torch.cat([result['input_ids'], torch.tensor([[12, 13, 14]])], dim=-1)
            result['attention_mask'] = torch.ones_like(result['input_ids'])
        return result
    monkeypatch.setattr(Qwen2_5_VLProcessor, '__call__', call)
    processor = Qwen2_5_VLProcessor()
    processor.apply_chat_template = lambda messages, **kw: 'full' if len(messages) == 2 else 'prefix'
    batch = VideoCollator(processor, 64, 8, 2., 65536)([
        {'videos': ['sample.avi'], 'messages': [{'role': 'user', 'content': '<video> question'},
                                              {'role': 'assistant', 'content': 'answer'}]}])
    assert batch['labels'][0, -3:].tolist() == [12, 13, 14]
    assert (batch['labels'][0, :-3] == -100).all()
    assert batch['position_ids'].shape == (3, 1, batch['input_ids'].shape[-1])
    assert batch['second_per_grid_ts'] == [1.25]


def test_sft_model_selection_and_batch_config():
    assert sft_wrap_classes('qwen2_5_vl')[0] == 'Qwen2_5_VLDecoderLayer'
    assert sft_wrap_classes('qwen3_vl')[0] == 'Qwen3VLTextDecoderLayer'
    with pytest.raises(ValueError): sft_wrap_classes('qwen2')
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PPOConfig()),
                                  OmegaConf.load(ROOT / 'examples/action_recognition/rlsd.yaml')))
    config.deep_post_init()
    validate_training_config(config)
    assert config.trainer.nnodes == 4
    world_size = config.trainer.nnodes * config.trainer.n_gpus_per_node
    assert config.worker.actor.global_batch_size * config.worker.rollout.n // world_size == 4
    config.worker.actor.global_batch_size = 64
    with pytest.raises(ValueError, match='divisible'):
        validate_training_config(config)
