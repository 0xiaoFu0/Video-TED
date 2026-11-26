import numpy as np
import pytest
import torch
import ast
from contextlib import nullcontext
from types import SimpleNamespace as NS
from pathlib import Path
from omegaconf import OmegaConf
from verl.protocol import DataProto
from verl.trainer.config import PPOConfig
from verl.ar.validation import validate_training_config
from verl.ar.trainer import augment_reference
from verl.ar.batch import concat_prompt_response
from verl.ar.objectives import group_statistics
from verl.ar.reward import compute_score

ROOT=Path(__file__).resolve().parents[2]

class Tokenizer:
    pad_token_id=0;eos_token_id=2
    def encode(self,text,add_special_tokens=False):return [3,4,5]

def test_config_and_unsupported_combination():
    config=OmegaConf.merge(OmegaConf.structured(PPOConfig()),OmegaConf.load(ROOT/'examples/action_recognition/rlsd.yaml'))
    config=OmegaConf.to_object(config);config.deep_post_init();validate_training_config(config)
    assert config.opsd.stgca_teacher_sync_interval==1
    config.worker.rollout.enable_mix_policy=True
    with pytest.raises(ValueError):validate_training_config(config)
    config.worker.rollout.enable_mix_policy=False
    config.worker.actor.dynamic_batching=True
    with pytest.raises(ValueError):validate_training_config(config)

def test_reference_attachment_does_not_change_rollout_rewards_or_rows():
    B=4
    tensors=dict(prompts=torch.tensor([[0,8,9]]*B),
        input_ids=torch.tensor([[0,8,9,7,2]]*B),attention_mask=torch.tensor([[0,1,1,1,1]]*B),
        position_ids=torch.tensor([[[0,0,1,2,3]]*4]*B),responses=torch.tensor([[7,2]]*B),
        response_mask=torch.ones(B,2),token_level_scores=torch.tensor([[0.,0.],[0.,1.],[0.,0.],[0.,1.]]))
    reference='<think>Exercise</think><answer>pushup</answer>'
    non=dict(uid=np.array(['wrong','right','wrong','right'],dtype=object),
        reference_response=np.array([reference]*B,dtype=object),reference_status=np.array(['unreviewed']*B,dtype=object),
        ground_truth=np.array(['pushup']*B,dtype=object),multi_modal_data=np.array([{}]*B,dtype=object))
    batch=DataProto.from_dict(tensors=tensors,non_tensors=non,meta_info={'temperature':1.})
    calls=[]
    def compute_old(data):
        calls.append(data)
        assert torch.equal(data.batch['input_ids'][:,:3],tensors['prompts'])
        assert 'teacher_context' not in data.non_tensor_batch
        return DataProto.from_dict(tensors={'old_log_probs':torch.zeros_like(data.batch['responses'],dtype=torch.float)})
    cfg=NS(ar_reference_coef=.1,ar_reference_kappa=1.,ar_reference_max_length=100,ar_allow_unreviewed_reference=True)
    trainer=NS(opsd_config=cfg,tokenizer=Tokenizer(),config=NS(worker=NS(rollout=NS(n=2))),
               actor_rollout_ref_wg=NS(compute_log_probs=compute_old))
    original=tensors['token_level_scores'].clone()
    result=augment_reference(trainer,batch,{})
    assert len(result)==4 and len(calls)==1
    assert torch.equal(result.batch['token_level_scores'],original)
    assert torch.equal(result.batch['advantages'],torch.zeros(B,2))
    assert result.batch['ar_reference_weights'].tolist()==[1.,0.,1.,0.]
    assert result.batch['ar_reference_position_ids'].shape==(B,4,7)
    assert result.batch['ar_reference_position_ids'][0,0].tolist()==[0,0,1,2,3,4,5]
    assert not result.batch['ar_reference_old_log_probs'].requires_grad


def make_reference_batch(references,statuses,allow_unreviewed=True,max_length=100):
    tensors=dict(prompts=torch.tensor([[0,8,9]]*4),
        input_ids=torch.tensor([[0,8,9,7,2]]*4),attention_mask=torch.tensor([[0,1,1,1,1]]*4),
        position_ids=torch.tensor([[[0,0,1,2,3]]*4]*4),responses=torch.tensor([[7,2]]*4),
        response_mask=torch.ones(4,2),token_level_scores=torch.tensor([[0.,0.],[0.,0.],[0.,1.],[0.,1.]]))
    non=dict(uid=np.array(['a','b','a','b'],dtype=object),
        reference_response=np.array(references,dtype=object),reference_status=np.array(statuses,dtype=object),
        ground_truth=np.array(['pushup']*4,dtype=object),multi_modal_data=np.array([{}]*4,dtype=object),
        teacher_context=np.array(['LABEL_ONLY_TEACHER']*4,dtype=object))
    batch=DataProto.from_dict(tensors=tensors,non_tensors=non,meta_info={'temperature':1.})
    calls=[]
    def compute_old(data):
        calls.append(data)
        assert torch.equal(data.batch['input_ids'][:,:3],batch.batch['prompts'])
        assert 'teacher_context' not in data.non_tensor_batch
        return DataProto.from_dict(tensors={'old_log_probs':torch.zeros_like(data.batch['responses'],dtype=torch.float)})
    cfg=NS(ar_reference_coef=.1,ar_reference_kappa=1.,ar_reference_max_length=max_length,
           ar_allow_unreviewed_reference=allow_unreviewed)
    trainer=NS(opsd_config=cfg,tokenizer=Tokenizer(),config=NS(worker=NS(rollout=NS(n=2))),
               actor_rollout_ref_wg=NS(compute_log_probs=compute_old))
    return trainer,batch,calls


def test_mixed_reference_statuses_preserve_rollout_groups_and_mask_empty_rows():
    reference='<think>Exercise</think><answer>pushup</answer>'
    trainer,batch,calls=make_reference_batch([None,reference,None,reference],
                                             ['rejected','verified','rejected','verified'])
    scores=batch.batch['token_level_scores'].clone()
    sampled=batch.batch['responses'].clone()
    expected_adv,_=group_statistics(scores.sum(-1),batch.non_tensor_batch['uid'],2)
    metrics={}
    result=augment_reference(trainer,batch,metrics)
    assert len(result) == 4 and len(calls) == 1
    torch.testing.assert_close(result.batch['advantages'],expected_adv[:,None].expand(-1,2))
    assert torch.equal(result.batch['token_level_scores'],scores)
    assert torch.equal(result.batch['responses'],sampled)
    assert result.batch['ar_reference_weights'].tolist() == [0.,.25,0.,.25]
    assert result.batch['ar_reference_response_mask'].tolist() == [[0]*4,[1]*4,[0]*4,[1]*4]
    assert result.batch['ar_reference_responses'][[0,2]].tolist() == [[0]*4,[0]*4]
    assert metrics['ar/reference_eligible_fraction'] == .5


@pytest.mark.parametrize('status,allow_unreviewed',[('rejected',True),('unreviewed',False)])
def test_all_inactive_references_skip_old_policy_forward(status,allow_unreviewed):
    trainer,batch,calls=make_reference_batch([None,'',None,''],[status]*4,allow_unreviewed)
    expected_adv,_=group_statistics(batch.batch['token_level_scores'].sum(-1),batch.non_tensor_batch['uid'],2)
    metrics={}
    result=augment_reference(trainer,batch,metrics)
    assert calls == []
    assert result.batch['ar_reference_responses'].shape == (4,1)
    assert result.batch['ar_reference_response_mask'].sum().item() == 0
    assert result.batch['ar_reference_weights'].sum().item() == 0
    assert torch.equal(result.batch['ar_reference_old_log_probs'],torch.zeros(4,1))
    torch.testing.assert_close(result.batch['advantages'],expected_adv[:,None].expand(-1,2))
    assert metrics['ar/reference_eligible_fraction'] == 0.


@pytest.mark.parametrize('reference,status,max_length',[(None,'verified',100),
    ('','unreviewed',100),('<answer>pushup</answer>','reviewed',100),
    ('<think>Exercise</think><answer>run</answer>','verified',100),
    ('<think>Exercise</think><answer>pushup</answer>','verified',3)])
def test_active_invalid_reference_fails_before_forward(reference,status,max_length):
    trainer,batch,calls=make_reference_batch([reference]*4,[status]*4,max_length=max_length)
    with pytest.raises(ValueError):
        augment_reference(trainer,batch,{})
    assert calls == []


def test_zero_difficulty_weight_still_validates_eligible_reference():
    trainer,batch,calls=make_reference_batch([None]*4,['verified']*4)
    batch.batch['token_level_scores'][:,-1]=1.
    with pytest.raises(ValueError,match='Reference answer does not match target'):
        augment_reference(trainer,batch,{})
    assert calls == []


def test_all_correct_labels_without_think_activate_protocol_reference_demonstration():
    reference='<think>Exercise</think><answer>pushup</answer>'
    trainer,batch,calls=make_reference_batch([reference]*4,['verified']*4)
    scores=compute_score([dict(response='<answer>pushup</answer>',ground_truth='pushup')]*4,
                         reward_mode='protocol')
    assert [row['accuracy'] for row in scores] == [1.]*4
    assert [row['protocol_valid'] for row in scores] == [0.]*4
    batch.batch['token_level_scores'].zero_()
    batch.batch['token_level_scores'][:,-1]=torch.tensor([row['overall'] for row in scores])
    trainer.opsd_config.ar_reward_mode='protocol'
    metrics={}
    result=augment_reference(trainer,batch,metrics)
    assert len(calls) == 1
    assert torch.equal(result.batch['advantages'],torch.zeros(4,2))
    assert result.batch['ar_reference_weights'].tolist() == [1.]*4
    assert result.batch['ar_reference_response_mask'].sum().item() > 0
    assert metrics['ar/group_joint_success_mean'] == 0.
    assert 'ar/group_accuracy_mean' not in metrics
    assert metrics['ar/reference_coef'] == .1


def test_mesd_mode_attaches_sample_std_advantages_and_preserves_reference_gate():
    reference='<think>Exercise</think><answer>pushup</answer>'
    trainer,batch,calls=make_reference_batch([reference]*4,['verified']*4)
    trainer.opsd_config.ar_alignment_mode='mesd'
    metrics={}
    result=augment_reference(trainer,batch,metrics)
    expected=.5/(2.**-.5+1e-4)
    torch.testing.assert_close(result.batch['advantages'],
        torch.tensor([[-expected,-expected],[-expected,-expected],
                      [expected,expected],[expected,expected]]))
    assert result.batch['ar_reference_weights'].tolist() == [.25]*4
    assert len(calls) == 1
    assert metrics['ar/advantage_sample_std'] == 1.
    assert metrics['ar/advantage_epsilon'] == 1e-4


@pytest.mark.parametrize('step,teacher_active,weight',[(1,True,.5),(1000,True,.0005),(1001,False,0.)])
def test_production_teacher_schedule_cutoff_preserves_reference_attachment(step,teacher_active,weight):
    cfg=OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PPOConfig()),
        OmegaConf.load(ROOT/'examples/action_recognition/rlsd.yaml')))
    cfg.deep_post_init()
    reference='<think>Exercise</think><answer>pushup</answer>'
    trainer,batch,reference_calls=make_reference_batch([reference]*4,['verified']*4)
    trainer.config=cfg;trainer.opsd_config=cfg.opsd;trainer.global_step=step
    trainer.processor=None;trainer._teacher_template_path=None
    trainer._get_teacher_questions=lambda data:['Full original question']*4
    batch.non_tensor_batch['uid']=np.array(['one_group']*4,dtype=object)
    prompt_calls=[]
    def construct(**kwargs):
        prompt_calls.append(kwargs)
        return NS(batch=dict(input_ids=torch.ones(4,8,dtype=torch.long),attention_mask=torch.ones(4,8),
            position_ids=torch.arange(8)[None,None,:].expand(4,4,8)))
    path=ROOT/'verl/trainer/opsd_trainer.py'
    cls=next(node for node in ast.parse(path.read_text()).body
        if isinstance(node,ast.ClassDef) and node.name == 'RaySimpleGRPOOPSDTrainer')
    fn=next(node for node in cls.body if isinstance(node,ast.FunctionDef)
        and node.name == '_augment_batch_with_grpo_opsd')
    concat=next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)
        and node.name == '_concat_prompt_and_response')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0),concat,fn],type_ignores=[])
    ast.fix_missing_locations(module)
    namespace=dict(torch=torch,np=np,timer=lambda *args:nullcontext(),construct_teacher_prompts=construct,
        __name__='verl.trainer.ar_schedule_test',__package__='verl.trainer')
    exec(compile(module,str(path),'exec'),namespace)
    result=namespace['_augment_batch_with_grpo_opsd'](trainer,batch,{}, {})
    assert len(prompt_calls) == int(teacher_active)
    if teacher_active:
        assert prompt_calls[0]['max_prompt_length'] == 18432
        assert prompt_calls[0]['safe_left_truncation'] is True
        assert prompt_calls[0]['strict_prompt_length'] is False
        assert prompt_calls[0]['hints'] == [[value] for value in batch.non_tensor_batch['teacher_context']]
        assert 'grpo_opsd_teacher_input_ids' in result.batch
    else:
        assert 'grpo_opsd_teacher_input_ids' not in result.batch
    meta=result.meta_info['opsd_config']
    assert meta['ar_teacher_enabled'] is teacher_active
    assert meta['stgca_lambda'] == pytest.approx(weight)
    assert meta['ar_teacher_update_mode'] == 'ema'
    assert meta['ar_teacher_update_rate'] == .01
    assert result.batch['ar_reference_weights'].tolist() == [.25]*4
    assert len(reference_calls) == 1


@pytest.mark.parametrize('changed_field',['reference_response','reference_status'])
def test_inactive_references_still_require_consistent_group_annotations(changed_field):
    trainer,batch,calls=make_reference_batch([None]*4,['rejected']*4)
    batch.non_tensor_batch[changed_field][2]='' if changed_field == 'reference_response' else 'verified'
    with pytest.raises(ValueError,match='Reference differs within a rollout group'):
        augment_reference(trainer,batch,{})
    assert calls == []
