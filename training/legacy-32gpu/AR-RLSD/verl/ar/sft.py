"""Optional Qwen2.5-VL/Qwen3-VL cold-start SFT; one video per microbatch, assistant-only loss."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from .prepare_data import read_rows, validate_rows


def assistant_labels(input_ids, prefix_ids, attention_mask):
    import torch
    if prefix_ids.shape[-1] >= input_ids.shape[-1]:
        raise ValueError('No assistant tokens remain')
    if not torch.equal(input_ids[:,:prefix_ids.shape[-1]],prefix_ids):
        raise ValueError('Chat-template prefix mismatch; refuse incorrect SFT masking')
    labels=input_ids.clone()
    labels[:,:prefix_ids.shape[-1]]=-100
    labels[attention_mask==0]=-100
    return labels


class VideoCollator:
    def __init__(self,processor,max_length,max_frames,fps,max_pixels):
        self.processor=processor;self.max_length=max_length
        self.max_frames=max_frames;self.fps=fps;self.max_pixels=max_pixels

    def __call__(self,rows):
        from ..utils.video import prepare_video_data, video_processor_kwargs
        if len(rows)!=1:
            raise ValueError('AR SFT V1 supports one video per microbatch; use gradient accumulation')
        row=rows[0]
        # Decode once; prefix and full text share exactly the same frames/metadata.
        prepared=prepare_video_data({'videos':row['videos']},self.processor,
                                    max_pixels=self.max_pixels,max_frames=self.max_frames,video_fps=self.fps)
        question=row['messages'][0]['content']
        before,after=question.split('<video>')
        user={'role':'user','content':[{'type':'text','text':before},
              {'type':'video','video':row['videos'][0]},{'type':'text','text':after}]}
        full=self.processor.apply_chat_template([user,row['messages'][1]],tokenize=False,
                                                add_generation_prompt=False,enable_thinking=False)
        prefix=self.processor.apply_chat_template([user],tokenize=False,
                                                  add_generation_prompt=True,enable_thinking=False)
        kwargs=dict(videos=prepared['video'],return_tensors='pt',add_special_tokens=False,
                    **video_processor_kwargs(self.processor,prepared))
        inputs=self.processor(text=[full],**kwargs)
        prefix_ids=self.processor(text=[prefix],**kwargs)['input_ids']
        if 'Qwen2_5' in type(self.processor).__name__:
            # Match RLSD/vLLM temporal positions, retaining fractional sampled FPS.
            from ..models.transformers.qwen2_vl import get_rope_index
            inputs['position_ids']=get_rope_index(
                self.processor,inputs['input_ids'][0],video_grid_thw=inputs['video_grid_thw'],
                second_per_grid_ts=inputs['second_per_grid_ts'],
                attention_mask=inputs['attention_mask'][0]).unsqueeze(1)
        if inputs['input_ids'].shape[-1]>self.max_length:
            raise ValueError('SFT sample exceeds max length; increase limit or reduce video frames')
        inputs['labels']=assistant_labels(inputs['input_ids'],prefix_ids,inputs['attention_mask'])
        return inputs


def sft_wrap_classes(model_type):
    if model_type == 'qwen2_5_vl':
        return ['Qwen2_5_VLDecoderLayer','Qwen2_5_VLVisionBlock']
    if model_type == 'qwen3_vl':
        return ['Qwen3VLTextDecoderLayer','Qwen3VLVisionBlock']
    raise ValueError('AR SFT supports qwen2_5_vl and qwen3_vl; got '+model_type)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--data',default='data/train.sft.think.jsonl')
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=float,default=1.)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--gradient-accumulation',type=int,default=16)
    p.add_argument('--max-length',type=int,default=16384)
    p.add_argument('--max-frames',type=int,default=32)
    p.add_argument('--fps',type=float,default=2.)
    p.add_argument('--max-pixels',type=int,default=262144)
    p.add_argument('--max-samples',type=int,default=0,help='Seeded optional cold-start subset; 0 uses all rows')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty SFT output directory')
    rows=read_rows(a.data);validate_rows(rows,True)
    if a.max_samples:
        import random
        rows=random.Random(a.seed).sample(rows,min(a.max_samples,len(rows)))
    import torch
    from transformers import AutoModelForImageTextToText,AutoConfig,Trainer,TrainingArguments
    from ..utils.tokenizer import get_processor
    config=AutoConfig.from_pretrained(a.model)
    wrap_classes=sft_wrap_classes(config.model_type)
    processor=get_processor(a.model,use_fast=True)
    model=AutoModelForImageTextToText.from_pretrained(a.model,torch_dtype=torch.bfloat16,
                                                         attn_implementation='sdpa')
    model.config.use_cache=False
    distributed=int(os.environ.get('WORLD_SIZE','1'))>1
    fsdp_options=dict(fsdp='full_shard auto_wrap',fsdp_config={
        'transformer_layer_cls_to_wrap':wrap_classes,
        'use_orig_params':True}) if distributed else {}
    args=TrainingArguments(output_dir=a.output,num_train_epochs=a.epochs,learning_rate=a.learning_rate,
        per_device_train_batch_size=1,gradient_accumulation_steps=a.gradient_accumulation,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        bf16=True,remove_unused_columns=False,report_to='none',logging_steps=1,
        save_strategy='epoch',save_total_limit=2,seed=a.seed,ddp_find_unused_parameters=False,**fsdp_options)
    trainer=Trainer(model=model,args=args,train_dataset=rows,
        data_collator=VideoCollator(processor,a.max_length,a.max_frames,a.fps,a.max_pixels))
    trainer.train();trainer.save_model(a.output)
    if trainer.is_world_process_zero():
        processor.save_pretrained(a.output)
        import json
        (out/'sft_sample_ids.json').write_text(json.dumps([r['metadata']['sample_id'] for r in rows],indent=2))

if __name__=='__main__':main()
