import pytest
import torch
from verl.ar.objectives import group_statistics, reference_weights, reweight_advantages, reference_loss_per_sequence
from verl.ar.batch import encode_references, reference_eligibility


def test_shuffled_groups_and_degenerate_rewards():
    rewards=torch.tensor([0.,1.,0.,1.,0.,0.])
    adv,mean=group_statistics(rewards,['a','b','c','b','a','c'],2)
    assert torch.equal(adv,torch.zeros(6))
    assert torch.equal(mean,torch.tensor([0.,1.,0.,1.,0.,0.]))
    assert torch.equal(reference_weights(mean,torch.ones(6,dtype=torch.bool)),1-mean)
    with pytest.raises(ValueError): group_statistics(rewards,['a']*6,2)


def test_mixed_group_population_std():
    adv,mean=group_statistics(torch.tensor([0.,1.,0.,1.]),['x']*4,4)
    torch.testing.assert_close(adv,torch.tensor([-1.,1.,-1.,1.]),atol=3e-6,rtol=0)
    assert torch.equal(reference_weights(mean,torch.ones(4,dtype=torch.bool)),torch.full((4,),.25))


def test_rlsd_detached_sign_and_extremes():
    teacher=torch.tensor([[10000.,-10000.,10000.,-10000.]],requires_grad=True)
    old=torch.zeros_like(teacher,requires_grad=True)
    advantage=torch.tensor([[1.,1.,-1.,-1.]])
    result=reweight_advantages(teacher,old,advantage,torch.ones_like(old),.5,.2)
    torch.testing.assert_close(result,torch.tensor([[1.1,.9,-.9,-1.1]]))
    assert not result.requires_grad
    assert torch.isfinite(result).all()


@pytest.mark.parametrize('mode',['ce','clipped'])
def test_reference_gradient_mask_and_stop_gradient(mode):
    logp=torch.tensor([[-2.,-3.,float('nan')],[-1.,-2.,-4.]],requires_grad=True)
    old=logp.detach().clone().requires_grad_()
    weights=torch.tensor([1.,0.],requires_grad=True)
    mask=torch.tensor([[1.,1.,0.],[1.,1.,1.]])
    loss=reference_loss_per_sequence(logp,old,mask,weights,mode).mean()
    loss.backward()
    torch.testing.assert_close(logp.grad,torch.tensor([[-.25,-.25,0.],[0.,0.,0.]]))
    assert old.grad is None and weights.grad is None


def test_clipping_and_overflow():
    current=torch.tensor([[10000.,-10000.]],requires_grad=True)
    loss=reference_loss_per_sequence(current,torch.zeros_like(current),torch.ones_like(current),torch.ones(1))
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(current.grad).all()
    assert current.grad[0,0]==0


def test_duplicate_reference_normalization():
    x=torch.tensor([[-2.,-3.]],requires_grad=True)
    one=reference_loss_per_sequence(x,x.detach(),torch.ones_like(x),torch.tensor([.25])).mean()
    one.backward();g=x.grad.clone();x.grad.zero_()
    repeated=x.expand(4,-1)
    many=reference_loss_per_sequence(repeated,repeated.detach(),torch.ones_like(repeated),torch.full((4,),.25)).mean()
    many.backward();torch.testing.assert_close(x.grad,g)


class Tokenizer:
    pad_token_id=0
    eos_token_id=2
    def encode(self,text,add_special_tokens=False):return [3]*len(text)


def test_reference_encoding_keeps_eos_and_fails_on_truncation():
    text='<think>Exercise</think><answer>pushup</answer>'
    ids,mask=encode_references(Tokenizer(),[text,text+' '],['pushup']*2,100)
    assert ids[0,mask[0].sum()-1]==2
    assert ids[0,-1]==0 and mask[0,-1]==0
    with pytest.raises(ValueError):encode_references(Tokenizer(),[text],['pushup'],5)
    with pytest.raises(ValueError):encode_references(Tokenizer(),[text],['run'],100)
    assert reference_eligibility(['unreviewed','verified','rejected'],False).tolist()==[False,True,False]
    assert reference_eligibility(['unreviewed'],True).tolist()==[True]


def test_reference_encoding_ignores_only_explicitly_ineligible_rows():
    text='<think>Exercise</think><answer>pushup</answer>'
    ids,mask=encode_references(Tokenizer(),[None,text,'malformed'*1000],
                               ['pushup']*3,100,eligible=[False,True,False])
    assert ids.shape == mask.shape == (3,len(text)+1)
    assert torch.all(ids[[0,2]] == Tokenizer.pad_token_id)
    assert mask[[0,2]].sum().item() == 0
    assert mask[1].sum().item() == len(text)+1
    assert ids[1,-1].item() == Tokenizer.eos_token_id
    ids,mask=encode_references(Tokenizer(),[None,''],['pushup']*2,100,
                               eligible=[False,False])
    assert ids.shape == (2,1) and mask.sum().item() == 0
    assert torch.all(ids == Tokenizer.pad_token_id)
    with pytest.raises(ValueError,match='eligibility shape'):
        encode_references(Tokenizer(),[text],['pushup'],100,eligible=[True,False])
    with pytest.raises(ValueError,match='batch shape'):
        encode_references(Tokenizer(),[text],[],100,eligible=[False])


@pytest.mark.parametrize('text,max_length',[(None,100),('',100),
    ('<think>Exercise</think><answer>run</answer>',100),
    ('<answer>pushup</answer>',100),
    ('<think>Exercise</think><answer>pushup</answer>',5)])
def test_eligible_reference_validation_is_never_replaced_by_dropping(text,max_length):
    with pytest.raises(ValueError):
        encode_references(Tokenizer(),[None,text],['pushup']*2,max_length,
                          eligible=[False,True])


@pytest.mark.parametrize('mode',['ce','clipped'])
def test_empty_inactive_references_have_zero_loss_and_gradient(mode):
    current=torch.tensor([[-2.,-3.],[float('nan'),float('nan')]],requires_grad=True)
    old=current.detach().clone().requires_grad_()
    weights=torch.tensor([.25,0.],requires_grad=True)
    mask=torch.tensor([[1.,1.],[0.,0.]])
    per_sequence=reference_loss_per_sequence(current,old,mask,weights,mode)
    assert torch.isfinite(per_sequence).all() and per_sequence[1].item() == 0
    per_sequence.mean().backward()
    assert current.grad[1].tolist() == [0.,0.]
    assert current.grad[0].abs().sum().item() > 0
    assert old.grad is None and weights.grad is None
    with pytest.raises(ValueError,match='Empty reference'):
        reference_loss_per_sequence(torch.zeros(1,2),torch.zeros(1,2),
                                    torch.zeros(1,2),torch.ones(1),mode)
    with pytest.raises(ValueError,match='Nonfinite reference'):
        reference_loss_per_sequence(torch.full((1,2),float('nan')),torch.zeros(1,2),
                                    torch.ones(1,2),torch.ones(1),mode)


def test_disable_dropout_preserves_training_for_checkpointing():
    from verl.ar.model import disable_training_dropout
    module=torch.nn.Sequential(torch.nn.Dropout(.8),torch.nn.Linear(2,2))
    module.train();module[1].attention_dropout=.2
    disable_training_dropout(module)
    assert module.training and module[0].training
    assert module[0].p==0 and module[1].attention_dropout==0
    x=torch.ones(2,2)
    torch.testing.assert_close(module(x),module(x))
