"""Numerical effects of AR's opt-in MeSD optimization alignment on CPU."""
import math

import pytest
import torch

from verl.ar.objectives import group_statistics, mesd_aligned_policy_loss
from verl.trainer.core_algos import compute_policy_loss


def policy(current, advantage, mask, old=None):
    if old is None:
        old=torch.zeros_like(current)
    return mesd_aligned_policy_loss(old_log_probs=old,log_probs=current,
        advantages=advantage,response_mask=mask,clip_ratio_low=.2,clip_ratio_high=.2)


def test_sample_std_changes_advantage_magnitude_without_changing_reference_difficulty():
    rewards=torch.tensor([0.,1.,0.,1.])
    legacy,legacy_mean=group_statistics(rewards,['mixed']*4,4)
    aligned,aligned_mean=group_statistics(rewards,['mixed']*4,4,eps=1e-4,unbiased=True)
    expected=.5/(math.sqrt(1./3.)+1e-4)
    torch.testing.assert_close(aligned,torch.tensor([-expected,expected,-expected,expected]))
    torch.testing.assert_close(legacy,torch.tensor([-1.,1.,-1.,1.]),atol=3e-6,rtol=0.)
    assert torch.equal(aligned_mean,legacy_mean)
    zero,_=group_statistics(torch.ones(4),['all_correct']*4,4,eps=1e-4,unbiased=True)
    assert torch.equal(zero,torch.zeros(4))


def test_sequence_average_gives_equal_total_gradient_to_short_and_long_responses():
    current=torch.zeros(2,3,requires_grad=True)
    mask=torch.tensor([[1.,0.,0.],[1.,1.,1.]])
    loss,_=policy(current,torch.ones_like(current),mask)
    loss.backward()
    torch.testing.assert_close(current.grad,torch.tensor([[-.5,0.,0.],[-1./6.]*3]))
    assert current.grad[0].sum().item() == pytest.approx(current.grad[1].sum().item())
    token_current=torch.zeros(2,3,requires_grad=True)
    token_loss,_=compute_policy_loss(torch.zeros_like(token_current),token_current,
        torch.ones_like(token_current),mask,.2,.2,3.,'default','token')
    token_loss.backward()
    assert token_current.grad[1].sum().item() == pytest.approx(3*token_current.grad[0].sum().item())


def test_no_dual_clip_preserves_negative_advantage_gradient_above_ratio_three():
    old=torch.full((1,1),-math.log(8.))
    current=torch.full((1,1),-math.log(2.),requires_grad=True)
    advantage=-torch.ones(1,1)
    loss,metrics=policy(current,advantage,torch.ones(1,1),old)
    loss.backward()
    assert loss.item() == pytest.approx(4.)
    assert current.grad.item() == pytest.approx(4.)
    # The preserved legacy objective caps the same negative-advantage loss at C=3.
    legacy=current.detach().clone().requires_grad_()
    legacy_loss,_=compute_policy_loss(old,legacy,advantage,torch.ones(1,1),.2,.2,3.,'default','token')
    legacy_loss.backward()
    assert legacy_loss.item() == pytest.approx(3.)
    assert legacy.grad.item() == 0.
    assert metrics['pg_clipfrac_lower'] == 0.


@pytest.mark.parametrize('ratio,advantage,expected',[(1.3,1.,-1.2),(.7,-1.,.8)])
def test_ordinary_two_sided_clipping_still_stops_saturated_gradient(ratio,advantage,expected):
    current=torch.tensor([[math.log(ratio)]],requires_grad=True)
    loss,_=policy(current,torch.tensor([[advantage]]),torch.ones(1,1))
    loss.backward()
    assert loss.item() == pytest.approx(expected)
    assert current.grad.item() == 0.


def test_global_sequence_normalization_is_invariant_to_microbatch_partition():
    mask=torch.tensor([[1.,0.,0.],[1.,1.,1.],[1.,1.,0.],[1.,0.,0.]])
    advantage=torch.tensor([[1.]*3,[-1.]*3,[.5]*3,[-.25]*3])
    full=torch.zeros(4,3,requires_grad=True)
    full_loss,_=policy(full,advantage,mask)
    full_loss.backward()
    expected=full.grad.clone()
    split=torch.zeros(4,3,requires_grad=True)
    for start,stop in ((0,1),(1,4)):
        local,_=policy(split[start:stop],advantage[start:stop],mask[start:stop])
        # This is the actor's DP-normalization rule at world_size=1.
        (local*((stop-start)/4.)).backward()
    torch.testing.assert_close(split.grad,expected)


def test_policy_padding_does_not_create_nan_loss_or_gradient():
    current=torch.tensor([[0.,float('nan')]],requires_grad=True)
    old=torch.tensor([[0.,float('nan')]],requires_grad=True)
    advantage=torch.tensor([[1.,float('nan')]],requires_grad=True)
    loss,_=policy(current,advantage,torch.tensor([[1.,0.]]),old)
    loss.backward()
    assert loss.item() == -1.
    torch.testing.assert_close(current.grad,torch.tensor([[-1.,0.]]))
    assert old.grad is None and advantage.grad is None
