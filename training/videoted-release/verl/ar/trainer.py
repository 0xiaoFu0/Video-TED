"""Attach reference tensors after rewards; never add oracle rows to rollouts."""
from __future__ import annotations
import torch
from .batch import encode_references, reference_eligibility, concat_prompt_response
from .objectives import group_statistics, reference_weights


def augment_reference(trainer, batch, metrics):
    from ..protocol import DataProto
    cfg=trainer.opsd_config
    alignment=getattr(cfg,'ar_alignment_mode','legacy')
    if alignment not in ('legacy','mesd'):
        raise ValueError('Unknown AR alignment mode')
    sample_std=alignment == 'mesd'
    eps=1e-4 if sample_std else 1e-6
    advantages,means=group_statistics(batch.batch['token_level_scores'].sum(-1),
        batch.non_tensor_batch['uid'], expected_size=trainer.config.worker.rollout.n,
        eps=eps,unbiased=sample_std)
    # Protocol mode uses joint class-and-format success for both advantages
    # and the reference failure gate; raw accuracy is a separate reward metric.
    metric='ar/group_joint_success_mean' if getattr(cfg,'ar_reward_mode','answer_only') == 'protocol' else 'ar/group_accuracy_mean'
    metrics[metric]=means.mean().item()
    metrics['ar/reference_coef']=float(cfg.ar_reference_coef)
    metrics['ar/advantage_sample_std']=float(sample_std)
    metrics['ar/advantage_epsilon']=eps
    batch.batch['advantages']=advantages[:,None]*batch.batch['response_mask']
    batch.batch['returns']=batch.batch['advantages'].clone()
    if cfg.ar_reference_coef == 0:
        return batch
    texts=batch.non_tensor_batch['reference_response']
    targets=batch.non_tensor_batch['ground_truth']
    statuses=batch.non_tensor_batch['reference_status']
    if any(len(values) != len(batch) for values in (texts,targets,statuses)):
        raise ValueError('Reference batch shape mismatch')
    eligible=reference_eligibility(statuses,cfg.ar_allow_unreviewed_reference)
    weights=reference_weights(means,eligible.to(means.device),cfg.ar_reference_kappa)
    # Group members must share a reference, including its eligibility.
    grouped={}
    for uid,text,status in zip(batch.non_tensor_batch['uid'],texts,statuses):
        pair=(text,status)
        if str(uid) in grouped and grouped[str(uid)] != pair:
            raise ValueError('Reference differs within a rollout group')
        grouped[str(uid)]=pair
    responses,mask=encode_references(trainer.tokenizer,texts,targets,
                                    cfg.ar_reference_max_length,eligible=eligible)
    prompt_length=batch.batch['prompts'].shape[-1]
    ids,attention,positions=concat_prompt_response(
        batch.batch['prompts'], batch.batch['attention_mask'][:,:prompt_length],
        batch.batch['position_ids'][...,:prompt_length],responses,mask)
    tensors=dict(input_ids=ids,attention_mask=attention,position_ids=positions,
                 responses=responses,response_mask=mask)
    non_tensors={k:batch.non_tensor_batch[k] for k in ('multi_modal_data','multi_modal_inputs')
                 if k in batch.non_tensor_batch}
    meta=dict(batch.meta_info)
    meta['global_token_num']=attention.sum(-1).tolist()
    if torch.any(weights > 0):
        old_batch=DataProto.from_dict(tensors=tensors,non_tensors=non_tensors,meta_info=meta)
        old=trainer.actor_rollout_ref_wg.compute_log_probs(old_batch).batch['old_log_probs']
    else:
        old=torch.zeros_like(responses,dtype=torch.float)
    for name,tensor in tensors.items():
        batch.batch['ar_reference_'+name]=tensor
    batch.batch['ar_reference_old_log_probs']=old.detach()
    batch.batch['ar_reference_weights']=weights.detach()
    metrics['ar/reference_weight_mean']=weights.mean().item()
    metrics['ar/reference_eligible_fraction']=eligible.float().mean().item()
    return batch
