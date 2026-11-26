"""Fail fast on unsupported combinations in the first AR implementation."""
import math
from collections.abc import Mapping


def validate_dispatch_batches(data, world_size):
    # Ray chunks prompts by all ranks before vLLM's TP grouping or n-repeat.
    for name in ('rollout_batch_size', 'mini_rollout_batch_size'):
        value = getattr(data, name)
        if value is not None and (value < 1 or value % world_size):
            raise ValueError(f'{name} must be positive and divisible by the GPU count ({world_size})')


def validate_training_config(config):
    c=config.opsd
    actor=config.worker.actor
    alignment=getattr(c,'ar_alignment_mode','legacy')
    if alignment not in {'legacy','mesd'}:
        raise ValueError('AR alignment mode must be legacy or mesd')
    mode=getattr(c,'ar_reward_mode','answer_only')
    if mode not in ('answer_only','protocol'):
        raise ValueError('AR ar_reward_mode must be answer_only or protocol')
    reward_kwargs=config.worker.reward.reward_function_kwargs
    if not isinstance(reward_kwargs,Mapping) or set(reward_kwargs)-{'reward_mode'}:
        raise ValueError('AR reward kwargs only support reward_mode')
    if reward_kwargs.get('reward_mode','answer_only') != mode:
        raise ValueError('opsd.ar_reward_mode must match worker.reward.reward_function_kwargs.reward_mode')
    if config.data.rollout_batch_size < 1 or actor.global_batch_size < 1 or config.data.rollout_batch_size % actor.global_batch_size:
        raise ValueError('Rollout batch size must be divisible by actor global batch size')
    world_size=config.trainer.nnodes * config.trainer.n_gpus_per_node
    trajectories=actor.global_batch_size * config.worker.rollout.n
    if world_size < 1 or trajectories % world_size:
        raise ValueError('Actor batch times rollout.n must be divisible by the GPU count')
    validate_dispatch_batches(config.data, world_size)
    if (trajectories // world_size) % actor.micro_batch_size_per_device_for_update:
        raise ValueError('Per-GPU actor batch must be divisible by the update microbatch')
    if not c.enable_grpo_opsd_simple or not c.freeze_teacher_model or c.stgca_teacher_sync_interval != 1:
        raise ValueError('AR requires simple RLSD with a frozen teacher refreshed each step')
    if c.stgca_thought_only or c.stgca_negative_only:
        raise ValueError('AR V1 reweights all action-tree tokens and both advantage signs')
    if config.worker.rollout.enable_mix_policy or config.worker.rollout.n <= 1:
        raise ValueError('AR requires G>1 genuine student rollouts, without mix-policy injection')
    if config.worker.actor.ulysses_size != 1:
        raise ValueError('AR V1 has not integrated sequence-parallel reference normalization')
    if config.worker.actor.dynamic_batching:
        raise ValueError('AR V1 requires fixed microbatches for synchronized reference forwards')
    expected_average='seq' if alignment=='mesd' else 'token'
    if actor.loss_avg_mode != expected_average:
        raise ValueError(f'AR {alignment} policy loss requires {expected_average} normalization')
    if config.algorithm.use_kl_loss or not config.algorithm.disable_kl or config.algorithm.online_filtering:
        raise ValueError('AR requires unfiltered binary success rewards without KL shaping')
    expected_truncation='left' if alignment=='mesd' else 'error'
    if config.data.truncation != expected_truncation or config.data.trajectory_key is not None:
        raise ValueError(f'AR {alignment} uses explicit teacher_context and requires {expected_truncation} prompt handling')
    if c.ar_reference_mode not in {'ce','clipped'} or not 0 < c.ar_reference_clip < 1:
        raise ValueError('Invalid reference objective')
    for x in (c.ar_reference_coef,c.ar_reference_kappa):
        if not math.isfinite(x) or x<0:
            raise ValueError('Reference coefficients must be finite and nonnegative')
    if c.ar_reference_max_length < 1:
        raise ValueError('Reference maximum length must be positive')
    if alignment=='mesd':
        if config.data.filter_overlong_prompts:
            raise ValueError('AR MeSD alignment keeps all rows; disable filter_overlong_prompts and fit prompt text instead')
        # Reject deprecated aliases or overrides that would silently alter the
        # agreed comparison, after OPSDConfig has resolved its aliases.
        expected={
            'ar_teacher_update_mode':'ema', 'ar_teacher_update_rate':0.01,
            'ar_teacher_update_interval':1, 'ar_teacher_disable_after_step':1000,
            'ar_teacher_max_prompt_length':18432,
            'stgca_lambda':0.5, 'stgca_lambda_warmup_steps':0,
            'stgca_lambda_decay_steps':1000, 'stgca_reweight_clip_range':0.2,
            'ar_reward_mode':'protocol', 'ar_reference_coef':0.1,
        }
        for name,value in expected.items():
            if getattr(c,name)!=value:
                raise ValueError(f'AR MeSD alignment requires opsd.{name}={value}')
        if actor.clip_ratio_low!=0.2 or actor.clip_ratio_high!=0.2:
            raise ValueError('AR MeSD alignment requires PPO clip 0.2/0.2')
        if config.worker.ref.offload.offload_params:
            raise ValueError('AR EMA does not support reference parameter offload')
        for name in ('enable_full_shard','use_orig_params','fsdp_size'):
            if getattr(actor.fsdp,name)!=getattr(config.worker.ref.fsdp,name):
                raise ValueError(f'AR EMA requires matching actor/teacher FSDP {name}')
