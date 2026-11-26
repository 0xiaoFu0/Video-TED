"""Production worker teacher checkpoint roundtrip, with CPU collective stubs."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,get_model_state_dict,set_model_state_dict)


ROOT=Path(__file__).resolve().parents[2]


def production_worker(monkeypatch,all_reduce=None):
    path=ROOT/'verl/workers/fsdp_workers.py'
    cls=next(node for node in ast.parse(path.read_text()).body
        if isinstance(node,ast.ClassDef) and node.name == 'FSDPWorker')
    methods=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in
        {'_uses_ar_ema_teacher','_require_ar_ema_teacher','save_checkpoint','load_checkpoint'}]
    for node in methods:node.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0),
        ast.ClassDef(name='Harness',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(module)
    namespace=dict(torch=torch,os=os,StateDictOptions=StateDictOptions,
        get_model_state_dict=get_model_state_dict,set_model_state_dict=set_model_state_dict,
        dist=NS(barrier=lambda:None,all_reduce=all_reduce or (lambda *args,**kwargs:None),ReduceOp=NS(MIN=0)))
    exec(compile(module,str(path),'exec'),namespace)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:'cpu')
    worker=namespace['Harness']()
    worker._has_actor=True;worker._has_critic=False;worker._has_ref=True
    worker._use_param_offload=False;worker._use_ref_param_offload=False;worker._use_optimizer_offload=False
    worker.config=NS(actor=NS(opsd_enabled=True,ar_teacher_update_mode='ema',freeze_teacher_model=True))
    worker.rank=3;worker.world_size=32
    worker.fsdp_module=torch.nn.Linear(2,1)
    worker.ref_fsdp_module=torch.nn.Linear(2,1).requires_grad_(False)
    worker.print_rank0=lambda message:None
    class ActorCheckpoint:
        def save_checkpoint(self,path,save_model_only):
            Path(path).mkdir(parents=True,exist_ok=True)
            torch.save(worker.fsdp_module.state_dict(),Path(path)/'actor.pt')
        def load_checkpoint(self,path):
            worker.fsdp_module.load_state_dict(torch.load(Path(path)/'actor.pt',weights_only=False))
    worker.checkpoint_manager=ActorCheckpoint()
    return worker


def test_teacher_roundtrip_restores_exact_ema_state_distinct_from_actor(tmp_path,monkeypatch):
    worker=production_worker(monkeypatch)
    with torch.no_grad():
        worker.fsdp_module.weight.fill_(10.);worker.fsdp_module.bias.fill_(11.)
        worker.ref_fsdp_module.weight.fill_(2.5);worker.ref_fsdp_module.bias.fill_(3.5)
    worker.save_checkpoint(str(tmp_path))
    assert (tmp_path/'teacher_world_size_32_rank_3.pt').is_file()
    with torch.no_grad():
        worker.fsdp_module.weight.fill_(-99.);worker.ref_fsdp_module.weight.fill_(-88.)
        worker.ref_fsdp_module.bias.fill_(-77.)
    worker.load_checkpoint(str(tmp_path))
    torch.testing.assert_close(worker.fsdp_module.weight,torch.full((1,2),10.))
    torch.testing.assert_close(worker.ref_fsdp_module.weight,torch.full((1,2),2.5))
    torch.testing.assert_close(worker.ref_fsdp_module.bias,torch.tensor([3.5]))


@pytest.mark.parametrize('failure',['missing','empty','corrupt'])
def test_resume_cannot_silently_reset_missing_or_corrupt_ema_teacher(tmp_path,monkeypatch,failure):
    worker=production_worker(monkeypatch)
    worker.save_checkpoint(str(tmp_path))
    teacher=tmp_path/'teacher_world_size_32_rank_3.pt'
    if failure == 'missing':teacher.unlink()
    else:teacher.write_bytes(b'' if failure == 'empty' else b'not a torch checkpoint')
    before=worker.ref_fsdp_module.weight.detach().clone()
    error=RuntimeError if failure == 'corrupt' else FileNotFoundError
    with pytest.raises(error,match='teacher shard'):
        worker.load_checkpoint(str(tmp_path))
    torch.testing.assert_close(worker.ref_fsdp_module.weight,before)


def test_remote_rank_missing_teacher_shard_also_blocks_local_restore(tmp_path,monkeypatch):
    def another_rank_missing(value,op):value.zero_()
    worker=production_worker(monkeypatch,all_reduce=another_rank_missing)
    worker.save_checkpoint(str(tmp_path))
    with pytest.raises(FileNotFoundError,match='at least one'):
        worker.load_checkpoint(str(tmp_path))


def test_legacy_snapshot_checkpoint_does_not_require_or_save_ema_shards(tmp_path,monkeypatch):
    worker=production_worker(monkeypatch)
    worker.config.actor.ar_teacher_update_mode='snapshot'
    worker.save_checkpoint(str(tmp_path))
    assert not list(tmp_path.glob('teacher_*.pt'))
    worker.load_checkpoint(str(tmp_path))
