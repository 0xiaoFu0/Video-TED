"""Production actor update with CPU forwards and single-rank collective stubs.
This checks loss wiring, not Ray/FSDP or video decoding.
"""
from __future__ import annotations
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from verl.trainer.core_algos import compute_policy_loss
from verl.utils.py_functional import append_to_dict
from verl.utils import torch_functional as VF

ROOT=Path(__file__).resolve().parents[2]

def production_actor(all_reduce=None):
    path=ROOT/'verl/workers/actor/dp_opsd_actor.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DataParallelOPSDActor')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in
             {'update_policy_grpo_opsd','_build_stgca_advantages','_sync_teacher_from_actor',
              '_ema_teacher_from_actor'}]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
        ast.ClassDef(name='Harness',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(module)
    env=dict(torch=torch,defaultdict=defaultdict,compute_policy_loss=compute_policy_loss,VF=VF,
        append_to_dict=append_to_dict,dist=NS(all_reduce=all_reduce or (lambda *a,**k:None),ReduceOp=NS(SUM=0)),
        tqdm=lambda value,**kwargs:value,__name__='verl.workers.actor.ar_test',__package__='verl.workers.actor')
    exec(compile(module,str(path),'exec'),env)
    return env['Harness']()

class Batch:
    def __init__(self,tensors,non_tensors=None,meta=None):
        self.batch=tensors;self.non_tensor_batch=non_tensors or {};self.meta_info=meta or {}
    def __len__(self):return next(iter(self.batch.values())).shape[0]
    def select(self,keys,nonkeys):return Batch({k:self.batch[k] for k in keys},{k:self.non_tensor_batch[k] for k in nonkeys},self.meta_info)
    def split(self,size):
        return [Batch({k:v[i:i+size] for k,v in self.batch.items()},
                      {k:v[i:i+size] for k,v in self.non_tensor_batch.items()},self.meta_info)
                for i in range(0,len(self),size)]

def run_update(microbatch,weight,coef=.1,return_calls=False,all_reduce=None,*,
               alignment='legacy',teacher_enabled=True,step=1,teacher_rate=.01,
               teacher_interval=1,optimizer_value=None,return_actor=False,row_advantages=None):
    actor=production_actor(all_reduce=all_reduce)
    actor.actor_module=torch.nn.Linear(1,1,bias=False)
    actor.teacher_module=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():actor.actor_module.weight.zero_();actor.teacher_module.weight.fill_(99)
    actor._get_teacher_module=lambda:actor.teacher_module
    actor.rank=1;actor.world_size=1
    actor.config=NS(global_batch_size_per_device=4,ppo_epochs=1,dynamic_batching=False,
        micro_batch_size_per_device_for_update=microbatch,clip_ratio_low=.2,clip_ratio_high=.2,
        clip_ratio_dual=3.,loss_type='default',
        loss_avg_mode='seq' if alignment == 'mesd' else 'token',use_kl_loss=False)
    calls=dict(student=0,teacher=0,reference=0)
    def forward(inputs,temperature,module=None):
        kind='teacher' if module is not None else 'reference' if torch.all(inputs['input_ids']==8) else 'student'
        calls[kind]+=1
        m=actor.actor_module if module is None else module
        return torch.nn.functional.logsigmoid(m.weight[0,0]).expand_as(inputs['responses'])
    actor._forward_micro_batch=forward;actor._forward_micro_batch_with_module=forward
    def optimizer_step():
        norm=actor.actor_module.weight.grad.norm()
        if optimizer_value is not None:
            with torch.no_grad():actor.actor_module.weight.fill_(optimizer_value)
        return norm
    actor._optimizer_step=optimizer_step
    B=4;T=3
    tensors=dict(input_ids=torch.zeros(B,5,dtype=torch.long),attention_mask=torch.ones(B,5),
        position_ids=torch.zeros(B,5,dtype=torch.long),responses=torch.zeros(B,T,dtype=torch.long),
        response_mask=torch.tensor([[1,1,0],[1,1,1],[1,0,0],[1,1,0]],dtype=torch.float),
        old_log_probs=torch.full((B,T),-np.log(2)),advantages=torch.zeros(B,T),
        grpo_opsd_teacher_input_ids=torch.ones(B,7,dtype=torch.long),
        grpo_opsd_teacher_attention_mask=torch.ones(B,7),grpo_opsd_teacher_position_ids=torch.zeros(B,7,dtype=torch.long))
    for key in ('input_ids','attention_mask','position_ids','responses','response_mask','old_log_probs'):
        tensors['ar_reference_'+key]=tensors[key].clone()
    tensors['ar_reference_input_ids'].fill_(8)
    weights=torch.as_tensor(weight,dtype=torch.float)
    tensors['ar_reference_weights']=weights.expand(B).clone() if weights.ndim == 0 else weights
    inactive=tensors['ar_reference_weights']==0
    tensors['ar_reference_response_mask'][inactive]=0
    tensors['ar_reference_old_log_probs'][inactive]=float('nan')
    if row_advantages is not None:
        tensors['advantages']=torch.tensor(row_advantages,dtype=torch.float)[:,None].expand(B,T).clone()
    if alignment == 'mesd' and not teacher_enabled:
        for key in list(tensors):
            if key.startswith('grpo_opsd_teacher_'):del tensors[key]
    meta=dict(temperature=1.,opsd_config=dict(ar_enabled=True,ar_reference_coef=coef,
        ar_reference_mode='clipped',ar_reference_clip=.2,stgca_lambda=.5,stgca_reweight_clip_range=.2,
        stgca_teacher_sync_interval=1,global_step=step,ar_alignment_mode=alignment,
        ar_teacher_update_mode='ema' if alignment == 'mesd' else 'snapshot',
        ar_teacher_enabled=teacher_enabled,ar_teacher_update_rate=teacher_rate,
        ar_teacher_update_interval=teacher_interval))
    metrics=actor.update_policy_grpo_opsd(Batch(tensors,dict(multi_modal_inputs=np.array([{}]*B,dtype=object)),meta))
    if alignment == 'legacy':assert actor.teacher_module.weight.item()==0
    assert actor.teacher_module.weight.grad is None
    result=(actor.actor_module.weight.grad.item(),metrics)
    if return_calls:result=result+(calls,)
    if return_actor:result=result+(actor,)
    return result

def test_production_update_reference_rescues_all_wrong_and_microbatch_invariance():
    one,metrics=run_update(1,1.)
    full,_=run_update(4,1.)
    assert one==pytest.approx(-.05)
    assert one==pytest.approx(full)
    assert 'ar/reference_loss' in metrics

def test_production_update_all_correct_and_baseline():
    gradient,_=run_update(1,0.)
    assert gradient==0
    gradient,metrics=run_update(1,1.,coef=0.)
    assert gradient==0 and 'ar/reference_loss' not in metrics


def test_all_inactive_references_skip_actor_reference_forward():
    gradient,metrics,calls=run_update(1,0.,return_calls=True)
    assert gradient == 0.
    assert calls == dict(student=4,teacher=4,reference=0)
    assert metrics['ar/reference_loss'] == [0.]*4


def test_mixed_empty_references_preserve_all_rows_normalization_and_microbatch_gradient():
    one,_,calls=run_update(1,[0.,1.,0.,1.],return_calls=True)
    full,_,full_calls=run_update(4,[0.,1.,0.,1.],return_calls=True)
    assert one == pytest.approx(-.025)
    assert full == pytest.approx(one)
    assert calls['reference'] == 2 and full_calls['reference'] == 1


def test_local_inactive_rank_still_forwards_when_another_rank_is_active():
    collective_calls=[]
    def all_reduce(tensor,op):
        if tensor.dtype == torch.long:
            collective_calls.append(tensor.item())
            # Simulate a positive reference weight on a different FSDP rank.
            tensor.fill_(1)
    gradient,metrics,calls=run_update(1,0.,return_calls=True,all_reduce=all_reduce)
    assert gradient == 0. and collective_calls == [0]*4
    assert calls == dict(student=4,teacher=4,reference=4)
    assert metrics['ar/reference_loss'] == [0.]*4


def test_aligned_actor_sequence_gradient_and_microbatch_partition_are_consistent():
    for microbatch in (1,4):
        aligned,metrics=run_update(microbatch,0.,coef=0.,alignment='mesd',
            teacher_enabled=False,step=1001,row_advantages=[1.,-1.,1.,-1.])
        assert aligned == pytest.approx(0.,abs=1e-7)
        assert metrics['ar/teacher_ema_applied'] == [0.]
    legacy,_=run_update(1,0.,coef=0.,row_advantages=[1.,-1.,1.,-1.])
    assert legacy == pytest.approx(.125)


def test_teacher_cutoff_skips_teacher_forward_but_keeps_reference_gradient():
    for microbatch in (1,4):
        gradient,metrics,calls,actor=run_update(microbatch,1.,alignment='mesd',
            teacher_enabled=False,step=1001,return_calls=True,return_actor=True)
        assert gradient == pytest.approx(-.05)
        assert calls['teacher'] == 0 and calls['reference'] > 0
        assert actor.teacher_module.weight.item() == 99.
        assert metrics['ar/teacher_ema_applied'] == [0.]
        assert 'ar/reference_loss' in metrics


def test_actor_ema_runs_once_after_optimizer_and_does_not_replace_teacher_before_forward():
    gradient,metrics,calls,actor=run_update(1,1.,alignment='mesd',step=1000,
        optimizer_value=2.,return_calls=True,return_actor=True)
    assert gradient == pytest.approx(-.05)
    assert calls['teacher'] == 4
    assert actor.teacher_module.weight.item() == pytest.approx(.99*99.+.01*2.)
    assert metrics['ar/teacher_ema_applied'] == [1.]


def test_ema_interval_can_skip_update_while_teacher_forward_remains_enabled():
    _,metrics,calls,actor=run_update(4,0.,alignment='mesd',step=3,teacher_interval=2,
        return_calls=True,return_actor=True)
    assert calls['teacher'] == 1
    assert actor.teacher_module.weight.item() == 99.
    assert metrics['ar/teacher_ema_applied'] == [0.]


def test_production_ema_updates_float_buffers_and_copies_integer_counters():
    class Model(torch.nn.Module):
        def __init__(self,param,float_buffer,counter):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor([param]))
            self.register_buffer('running_mean',torch.tensor([float_buffer]))
            self.register_buffer('counter',torch.tensor(counter,dtype=torch.long))
    actor=production_actor()
    actor.actor_module=Model(10.,20.,7)
    actor.teacher_module=Model(2.,4.,1)
    actor._ema_teacher_from_actor(.25)
    torch.testing.assert_close(actor.teacher_module.weight,torch.tensor([4.]))
    torch.testing.assert_close(actor.teacher_module.running_mean,torch.tensor([8.]))
    assert actor.teacher_module.counter.item() == 7
    assert actor.teacher_module.weight.grad is None


def test_ema_shape_mismatch_is_rejected_before_mutating_any_teacher_weight():
    actor=production_actor()
    actor.teacher_module=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,1))
    actor.actor_module=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,2))
    before={key:value.clone() for key,value in actor.teacher_module.state_dict().items()}
    with pytest.raises(RuntimeError,match='shape mismatch'):
        actor._ema_teacher_from_actor(.01)
    for key,value in actor.teacher_module.state_dict().items():
        torch.testing.assert_close(value,before[key])
