"""Verify v3 configuration through the real structured CLI/YAML route."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from verl.ar.validation import validate_training_config
from verl.trainer.config import PPOConfig


ROOT = Path(__file__).resolve().parents[2]
FULL_RUN = [
    'trainer.nnodes=4', 'trainer.n_gpus_per_node=8',
    'data.rollout_batch_size=32', 'worker.actor.global_batch_size=32',
    'opsd.ar_reference_max_length=1501',
]


def configuration(*overrides):
    cfg = OmegaConf.merge(OmegaConf.structured(PPOConfig()),
        OmegaConf.load(ROOT / 'examples/action_recognition/rlsd.yaml'),
        OmegaConf.from_cli(FULL_RUN + list(overrides)))
    cfg = OmegaConf.to_object(cfg)
    cfg.deep_post_init()
    return cfg


def test_full_32gpu_v3_uses_protocol_and_explicit_mesd_alignment():
    assert PPOConfig().opsd.ar_reward_mode == 'answer_only'
    assert PPOConfig().opsd.ar_alignment_mode == 'legacy'
    cfg = configuration()
    validate_training_config(cfg)
    assert cfg.opsd.ar_reward_mode == 'protocol'
    assert cfg.worker.reward.reward_function_kwargs == {'reward_mode': 'protocol'}
    assert cfg.opsd.ar_reference_coef == .1
    assert cfg.opsd.ar_reference_mode == 'clipped'
    assert cfg.opsd.ar_reference_max_length == 1501
    assert cfg.worker.rollout.n == 4
    assert cfg.data.mini_rollout_batch_size is None
    assert cfg.opsd.ar_alignment_mode == 'mesd'
    assert cfg.opsd.ar_teacher_update_mode == 'ema'
    assert cfg.opsd.ar_teacher_update_rate == .01
    assert cfg.opsd.ar_teacher_update_interval == 1
    assert cfg.opsd.ar_teacher_disable_after_step == 1000
    assert cfg.opsd.ar_teacher_max_prompt_length == 18432
    assert cfg.opsd.stgca_lambda == .5
    assert cfg.opsd.stgca_lambda_warmup_steps == 0
    assert cfg.opsd.stgca_lambda_decay_steps == 1000
    assert cfg.data.truncation == 'left'
    assert cfg.worker.actor.loss_avg_mode == 'seq'
    assert cfg.worker.actor.ar_teacher_update_mode == 'ema'


def test_answer_only_remains_an_explicit_supported_baseline_configuration():
    cfg = configuration('opsd.ar_reward_mode=answer_only',
        'worker.reward.reward_function_kwargs={"reward_mode":"answer_only"}',
        'opsd.ar_alignment_mode=legacy','opsd.ar_teacher_update_mode=snapshot',
        'worker.actor.loss_avg_mode=token','data.truncation=error')
    validate_training_config(cfg)


@pytest.mark.parametrize('overrides', [
    ['opsd.ar_reward_mode=answer_only'],
    ['worker.reward.reward_function_kwargs={"reward_mode":"answer_only"}'],
    ['opsd.ar_reward_mode=protocol_tree'],
    ['worker.reward.reward_function_kwargs={"reward_mode":"protocol","evidence_lambda":1.0}'],
])
def test_structured_configuration_rejects_reward_route_mismatch_or_removed_options(overrides):
    cfg = configuration(*overrides)
    with pytest.raises(ValueError):
        validate_training_config(cfg)


def test_empty_cli_mapping_does_not_silently_clear_yaml_reward_mode():
    cfg = configuration('worker.reward.reward_function_kwargs={}')
    assert cfg.worker.reward.reward_function_kwargs == {'reward_mode': 'protocol'}
    validate_training_config(cfg)


@pytest.mark.parametrize('overrides', [
    ['data.rollout_batch_size=16', 'worker.actor.global_batch_size=16'],
    ['data.mini_rollout_batch_size=16'],
])
def test_four_rollouts_do_not_rescue_a_prompt_batch_too_small_for_32_ranks(overrides):
    cfg = configuration(*overrides)
    with pytest.raises(ValueError, match='GPU count'):
        validate_training_config(cfg)


@pytest.mark.parametrize('override',[
    'data.filter_overlong_prompts=true',
    'opsd.ar_alignment_mode=invalid','opsd.ar_teacher_update_mode=snapshot',
    'opsd.ar_teacher_update_rate=1.0','opsd.ar_teacher_disable_after_step=999',
    'opsd.rlsd_lambda_decay_steps=0','opsd.rlsd_lambda_warmup_steps=100',
    'worker.actor.loss_avg_mode=token','data.truncation=error',
    'worker.ref.offload.offload_params=true',
    'worker.ref.fsdp.use_orig_params=true',
])
def test_real_v3_config_rejects_silent_optimizer_teacher_or_fsdp_misalignment(override):
    with pytest.raises(ValueError):
        cfg=configuration(override)
        validate_training_config(cfg)


def test_production_runner_rejects_row_filtering_before_tokenizer_data_or_worker_creation():
    cfg=configuration('data.filter_overlong_prompts=true')
    path=ROOT/'verl/trainer/opsd_main.py'
    cls=next(node for node in ast.parse(path.read_text()).body
        if isinstance(node,ast.ClassDef) and node.name == 'OPSDRunner')
    run=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name == 'run')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0),run],type_ignores=[])
    ast.fix_missing_locations(module)
    calls=[]
    def forbid(name):
        def invoked(*args,**kwargs):
            calls.append(name)
            raise AssertionError(f'{name} ran before AR configuration validation')
        return invoked
    namespace=dict(json=json,print=lambda *args,**kwargs:None,
        get_tokenizer=forbid('tokenizer'),get_processor=forbid('processor'),
        create_dataloader=forbid('dataloader'),ray=SimpleNamespace(remote=forbid('worker')),
        ResourcePoolManager=forbid('resources'),
        __name__='verl.trainer.ar_runner_test',__package__='verl.trainer')
    exec(compile(module,str(path),'exec'),namespace)
    with pytest.raises(ValueError,match='disable filter_overlong_prompts'):
        namespace['run'](object(),cfg)
    assert calls == []
