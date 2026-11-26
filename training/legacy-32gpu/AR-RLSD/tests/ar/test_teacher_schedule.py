"""One-indexed MeSD schedule and compatibility of the generic OPSD schedule."""
from types import SimpleNamespace

import pytest

from verl.ar.schedule import teacher_schedule


@pytest.mark.parametrize('step,weight',[(0,0.),(5,.25),(10,.5),(20,.25),(30,0.),(31,0.)])
def test_generic_legacy_warmup_and_decay_are_preserved_without_alignment_fields(step,weight):
    cfg=SimpleNamespace(stgca_lambda=.5,stgca_lambda_warmup_steps=10,
        stgca_lambda_decay_steps=20)
    active,value=teacher_schedule(cfg,step)
    assert active is True and value == pytest.approx(weight)


def test_generic_legacy_constant_weight_does_not_enable_ar_cutoff():
    cfg=SimpleNamespace(stgca_lambda=.5,stgca_lambda_warmup_steps=0,
        stgca_lambda_decay_steps=0,ar_alignment_mode='legacy',ar_teacher_disable_after_step=1000)
    assert teacher_schedule(cfg,1001) == (True,.5)


def test_mesd_rejects_zero_indexed_step():
    cfg=SimpleNamespace(stgca_lambda=.5,stgca_lambda_warmup_steps=0,
        stgca_lambda_decay_steps=1000,ar_alignment_mode='mesd',ar_teacher_disable_after_step=1000)
    with pytest.raises(ValueError,match='one-indexed'):
        teacher_schedule(cfg,0)
