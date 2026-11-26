"""Exercise production left truncation while retaining full teacher context."""
import ast
import copy
import os
from pathlib import Path

from jinja2 import Template
import pytest
import torch

from verl.utils import torch_functional as VF
from verl.ar.prompt_length import fit_prompt_length


ROOT=Path(__file__).resolve().parents[2]


class Tokenizer:
    pad_token_id=0
    is_fast=True
    all_special_ids=[]
    all_special_tokens=[]
    def encode(self,text,**kwargs):
        return [ord(char) for char in text]
    def apply_chat_template(self,messages,**kwargs):
        content=messages[0]['content']
        text=content if isinstance(content,str) else content[0]['text']
        return text+'|assistant'
    def __call__(self,texts,**kwargs):
        if isinstance(texts,str):
            return dict(input_ids=self.encode(texts),offset_mapping=[(i,i+1) for i in range(len(texts))])
        ids=torch.tensor([self.encode(texts[0])])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids))


def production_dataset(row,truncation):
    path=ROOT/'verl/utils/dataset.py'
    cls=next(node for node in ast.parse(path.read_text()).body
        if isinstance(node,ast.ClassDef) and node.name == 'RLHFDataset')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    ast.fix_missing_locations(module)
    class Rows:
        def __len__(self):return 1
        def __getitem__(self,index):return copy.deepcopy(row)
    namespace=dict(torch=torch,os=os,Dataset=object,Template=Template,VF=VF,
        load_dataset=lambda *args,**kwargs:Rows(),build_prompt=lambda text,row:text,
        __name__='verl.utils.ar_left_test',__package__='verl.utils')
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['RLHFDataset']('fake.jsonl',Tokenizer(),None,max_prompt_length=8,
        filter_overlong_prompts=False,truncation=truncation)


def test_student_left_truncation_keeps_tail_positions_and_original_teacher_fields():
    row=dict(prompt='A long original action question',answer='pushup',
        teacher_context='Reference class: pushup\nMotion\n  Arms bend.',
        reference_response='<think>Motion</think><answer>pushup</answer>')
    dataset=production_dataset(row,'left')
    item=dataset[0]
    full=row['prompt']+'|assistant'
    assert len(dataset) == 1
    assert item['input_ids'].tolist() == [ord(char) for char in full[-8:]]
    assert item['raw_prompt_ids'] == item['input_ids'].tolist()
    assert item['position_ids'].tolist() == list(range(8))
    assert item['teacher_question'] == row['prompt']
    assert item['teacher_context'] == row['teacher_context']
    assert item['reference_response'] == row['reference_response']
    assert item['raw_prompt'] == full[-8:]
    assert item['prompt_original_tokens'] == len(full)
    assert item['prompt_removed_text_tokens'] == len(full)-8


def test_legacy_error_truncation_still_rejects_overlong_prompt():
    dataset=production_dataset(dict(prompt='Overlong original question',answer='pushup'),'error')
    with pytest.raises((ValueError,RuntimeError),match='longer|exceed'):
        dataset[0]


def test_long_visual_prompt_removes_old_text_and_preserves_whole_media_and_chat_skeleton():
    vision='<|vision_start|>00:00:01<|video_pad|><|vision_end|>'
    prompt='<|im_start|>user\n'+('old text '*40)+vision+'Latest question<|im_end|>\n<|im_start|>assistant\n'
    calls=[]
    grid=torch.tensor([[2,4,4]])
    def encode(text):
        calls.append(text)
        return dict(input_ids=torch.ones(1,len(text)+40,dtype=torch.long),
            attention_mask=torch.ones(1,len(text)+40),video_grid_thw=grid)
    fitted,inputs,removed,original=fit_prompt_length(prompt,Tokenizer(),encode,220)
    assert original == len(prompt)+40
    assert removed > 0 and inputs['input_ids'].shape[-1] <= 220
    assert fitted.startswith('<|im_start|>user\n')
    assert fitted.endswith('<|im_end|>\n<|im_start|>assistant\n')
    assert vision in fitted and 'Latest question' in fitted
    assert all(vision in value for value in calls)
    assert inputs['video_grid_thw'] is grid
    assert calls[-1] == fitted


def test_visual_skeleton_cannot_be_silently_cropped_to_fit_too_small_budget():
    prompt='<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|><|im_end|>'
    def encode(text):return dict(input_ids=torch.ones(1,len(text)+40,dtype=torch.long))
    with pytest.raises(ValueError,match='Complete visual blocks'):
        fit_prompt_length(prompt,Tokenizer(),encode,8)


def test_real_fast_byte_tokenizer_overlapping_chinese_offsets_keep_characters_and_media_intact():
    from tokenizers import Tokenizer as FastBackend, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    alphabet=sorted(pre_tokenizers.ByteLevel.alphabet())
    backend=FastBackend(models.BPE(vocab={char:index for index,char in enumerate(alphabet)},merges=[]))
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder=decoders.ByteLevel()
    special=['<|im_start|>','<|im_end|>','<|vision_start|>','<|vision_end|>','<|video_pad|>']
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,additional_special_tokens=special)
    vision='<|vision_start|>00:00:01<|video_pad|><|vision_end|>'
    prefix='<|im_start|>user\n'
    tail=vision+'\u6700\u540e\u95ee\u9898\uff1f<|im_end|>\n<|im_start|>assistant\n'
    prompt=prefix+('\u65e7\u7684\u63cf\u8ff0'*20)+tail
    offsets=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    assert len(set(offsets)) < len(offsets)  # Byte-level Chinese tokens share character spans.
    calls=[]
    grid=torch.tensor([[2,4,4]])
    def encode(text):
        calls.append(text)
        ids=torch.tensor([tokenizer.encode(text,add_special_tokens=False)+[0]*24])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids),video_grid_thw=grid)
    budget=len(tokenizer.encode(prefix+tail,add_special_tokens=False))+24+12
    fitted,inputs,removed,original=fit_prompt_length(prompt,tokenizer,encode,budget)
    assert removed > 0 and original > budget
    assert inputs['input_ids'].shape[-1] <= budget
    assert fitted.startswith(prefix) and fitted.endswith(tail)
    assert '\ufffd' not in fitted and fitted.encode('utf-8').decode('utf-8') == fitted
    assert inputs['video_grid_thw'] is grid
    assert all(vision in text for text in calls)
    assert calls[-1] == fitted


def test_production_teacher_tokenizer_uses_safe_left_fit_before_positions():
    path=ROOT/'verl/trainer/opsd_algos.py'
    functions=[node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)
        and node.name in ('_compute_position_ids','_tokenize_prompt')]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0),*functions],type_ignores=[])
    ast.fix_missing_locations(module)
    namespace=dict(torch=torch,VF=VF,__name__='verl.trainer.ar_prompt_test',__package__='verl.trainer')
    exec(compile(module,str(path),'exec'),namespace)
    full='Long teacher annotation then the final instruction'
    ids,mask,positions,raw_ids,prompt=namespace['_tokenize_prompt'](
        full,Tokenizer(),None,{},8,safe_left_truncation=True)
    assert prompt == full[-8:]
    assert ids.tolist() == raw_ids == [ord(char) for char in prompt]
    assert positions.tolist() == list(range(8))
    assert mask.tolist() == [1]*8
