"""Token-level credit modulation and reference trajectory losses for VideoTED."""
from __future__ import annotations
import math
from collections import defaultdict
import torch


@torch.no_grad()
def group_statistics(rewards, group_ids, expected_size=None, eps=1e-6, *, unbiased=False):
    if rewards.ndim != 1 or len(group_ids) != len(rewards):
        raise ValueError('Reward/group shape mismatch')
    if not torch.all(torch.isfinite(rewards) & ((rewards == 0) | (rewards == 1))):
        raise ValueError('VideoTED requires finite binary rewards')
    groups=defaultdict(list)
    for i, uid in enumerate(group_ids):
        groups[str(uid)].append(i)
    advantages=torch.zeros_like(rewards)
    means=torch.zeros_like(rewards)
    for indices in groups.values():
        if len(indices)<2 or (expected_size is not None and len(indices)!=expected_size):
            raise ValueError('Incomplete rollout group')
        values=rewards[indices]
        mean=values.mean()
        means[indices]=mean
        advantages[indices]=(values-mean)/(values.std(unbiased=unbiased)+eps)
    return advantages.detach(), means.detach()


def mesd_aligned_policy_loss(*, old_log_probs, log_probs, advantages, response_mask,
                             clip_ratio_low=.2, clip_ratio_high=.2, sequence_valid_mask=None):
    """Two-sided token PPO without dual clipping, averaged equally by sequence.

    Returns a LOCAL batch mean: each valid sequence first averages its response
    tokens; an invalid sequence contributes zero while still occupying a row.
    The actor must multiply this loss by local batch size * DP world size /
    global valid sequence count. AR's unfiltered batch treats every row as valid.
    Reference supervision is separate and does not enter this normalization.
    """
    shape=log_probs.shape
    if len(shape) != 2 or any(value.shape != shape for value in
                             (old_log_probs,advantages,response_mask)):
        raise ValueError('Policy tensors must have the same [batch, response] shape')
    if not 0 <= clip_ratio_low < 1 or not math.isfinite(clip_ratio_high) or clip_ratio_high < 0:
        raise ValueError('Invalid policy clipping range')
    if not torch.all(torch.isfinite(response_mask) & ((response_mask == 0) | (response_mask == 1))):
        raise ValueError('Policy response mask must be finite and binary')
    if sequence_valid_mask is None:
        valid=torch.ones(shape[0],device=log_probs.device,dtype=torch.float32)
    else:
        if sequence_valid_mask.shape != shape[:1] or not torch.all(
                torch.isfinite(sequence_valid_mask) & ((sequence_valid_mask == 0) | (sequence_valid_mask == 1))):
            raise ValueError('Policy sequence mask must be finite binary [batch]')
        valid=sequence_valid_mask.detach().to(device=log_probs.device,dtype=torch.float32)
    mask=response_mask.to(dtype=torch.float32)*valid[:,None]
    active=mask.bool()
    if any(not torch.isfinite(value[active]).all() for value in (old_log_probs,log_probs,advantages)):
        raise ValueError('Nonfinite policy value on a valid response token')
    lengths=response_mask.sum(-1)
    if ((valid > 0) & (lengths <= 0)).any():
        raise ValueError('Empty valid policy response')
    current=torch.where(active,log_probs.float(),0.)
    old=torch.where(active,old_log_probs.detach().float(),0.)
    advantage=torch.where(active,advantages.detach().float(),0.)
    log_ratio=current-old
    ratio=log_ratio.exp()
    clipped_ratio=ratio.clamp(min=1.-clip_ratio_low,max=1.+clip_ratio_high)
    token_loss=torch.maximum(-advantage*ratio,-advantage*clipped_ratio)
    sequence_loss=(token_loss*mask).sum(-1)/(lengths+1e-8)
    loss=sequence_loss.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite MeSD-aligned policy loss')
    token_count=mask.sum().clamp_min(1.)
    def token_mean(value):
        return ((value.detach()*mask).sum()/token_count).item()
    sequence_count=valid.sum().clamp_min(1.)
    metrics=dict(
        ppo_kl=token_mean(-log_ratio),
        entropy_loss=(((-current.detach()*mask).sum(-1)/(lengths+1e-8)).sum()/sequence_count).item(),
        pg_clipfrac_higher=token_mean(((ratio > 1.+clip_ratio_high) & (advantage > 0)).float()),
        pg_clipfrac_lower=token_mean(((ratio < 1.-clip_ratio_low) & (advantage < 0)).float()),
    )
    return loss,metrics


@torch.no_grad()
def reweight_advantages(teacher, old_student, advantages, mask, lam, clip):
    if not 0 <= lam <= 1 or not 0 < clip < 1:
        raise ValueError('Invalid RLSD parameters')
    valid=mask.bool()
    if any(not torch.isfinite(x[valid]).all() for x in (teacher, old_student, advantages)):
        raise ValueError('Nonfinite score on a valid token')
    delta=torch.where(valid, teacher.float()-old_student.float(), 0.)
    log_weight=advantages.sign()*delta
    weights=log_weight.clamp(math.log1p(-clip),math.log1p(clip)).exp()
    return advantages*((1-lam)+lam*weights)*mask


@torch.no_grad()
def reference_weights(means, eligible, kappa=1.):
    if kappa < 0 or not math.isfinite(kappa):
        raise ValueError('Invalid reference scale')
    return kappa*eligible.to(means.dtype)*(1-means).square()


def reference_loss_per_sequence(log_probs, old_log_probs, mask, weights, mode='clipped', clip=.2):
    """Per-sequence token mean. Caller normalizes over ALL rollout rows.

    Reference is duplicated G times for uniform-size groups, making this exactly
    the same mean as one reference per group over the complete rollout batch.
    Old scores and the difficulty/eligibility weights have no gradient path.
    """
    if log_probs.shape != old_log_probs.shape or log_probs.shape != mask.shape:
        raise ValueError('Reference score/mask shape mismatch')
    if weights.shape != log_probs.shape[:1] or not 0 < clip < 1:
        raise ValueError('Invalid reference weights or clipping range')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Reference weights must be finite and nonnegative')
    active=weights.detach() > 0
    effective_mask=mask*active[:,None]
    valid=effective_mask.bool()
    current=torch.where(valid, log_probs.float(), 0.)
    old=torch.where(valid, old_log_probs.detach().float(), 0.)
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError('Nonfinite reference log probability')
    lengths=effective_mask.sum(-1)
    if (active & (lengths <= 0)).any():
        raise ValueError('Empty reference response')
    if mode == 'ce':
        token_loss=-current
    elif mode == 'clipped':
        # weights are nonnegative: min(ratio, clipped_ratio)=min(ratio,1+clip).
        # Clamp in log space before exp, avoiding overflow and matching PPO.
        ratio=(current-old).clamp(max=math.log1p(clip)).exp()
        token_loss=-ratio
    else:
        raise ValueError('Unknown reference objective')
    return (token_loss*effective_mask).sum(-1)/lengths.clamp_min(1)*weights.detach()
