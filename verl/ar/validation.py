"""Validate the VideoTED training configuration before Ray starts."""

import math
from collections.abc import Mapping


def validate_training_config(config):
    data = config.data
    opsd = config.opsd
    actor = config.worker.actor
    rollout = config.worker.rollout
    aligned = opsd.ar_alignment_mode == "mesd"

    reward_mode = opsd.ar_reward_mode
    reward_kwargs = config.worker.reward.reward_function_kwargs
    if reward_mode not in {"answer_only", "protocol"}:
        raise ValueError("opsd.ar_reward_mode must be answer_only or protocol")
    if not isinstance(reward_kwargs, Mapping) or set(reward_kwargs) - {"reward_mode"}:
        raise ValueError("VideoTED reward options only support reward_mode")
    if reward_kwargs.get("reward_mode", "answer_only") != reward_mode:
        raise ValueError("The trainer and reward function must use the same reward_mode")

    if actor.global_batch_size < 1 or data.rollout_batch_size % actor.global_batch_size:
        raise ValueError("Rollout batch size must be divisible by a positive actor global batch size")
    world_size = config.trainer.nnodes * config.trainer.n_gpus_per_node
    if world_size < 1:
        raise ValueError("Training requires at least one GPU")
    # Ray splits prompts across ranks before expanding them into rollouts.
    for name in ("rollout_batch_size", "mini_rollout_batch_size"):
        size = getattr(data, name)
        if size is not None and (size < 1 or size % world_size):
            raise ValueError(f"{name} must be positive and divisible by the GPU count ({world_size})")
    trajectories = actor.global_batch_size * rollout.n
    if trajectories % world_size:
        raise ValueError("Actor batch size times rollout.n must be divisible by the GPU count")
    microbatch = actor.micro_batch_size_per_device_for_update
    if microbatch < 1 or (trajectories // world_size) % microbatch:
        raise ValueError("Per-GPU actor batch must be divisible by a positive update microbatch")

    if not opsd.enable_grpo_opsd_simple or not opsd.freeze_teacher_model:
        raise ValueError("VideoTED requires GRPO with a separate teacher")
    if not aligned and opsd.stgca_teacher_sync_interval != 1:
        raise ValueError("Snapshot alignment requires a teacher sync interval of one")
    if opsd.stgca_thought_only or opsd.stgca_negative_only:
        raise ValueError("VideoTED modulates credit across the response and for both advantage signs")
    if rollout.enable_mix_policy or rollout.n <= 1:
        raise ValueError("VideoTED requires multiple student rollouts without mix-policy injection")
    if actor.ulysses_size != 1 or actor.dynamic_batching:
        raise ValueError("Reference training requires fixed microbatches and ulysses_size=1")
    expected_average = "seq" if aligned else "token"
    if actor.loss_avg_mode != expected_average:
        raise ValueError(f"{opsd.ar_alignment_mode} alignment requires {expected_average} loss averaging")
    if config.algorithm.use_kl_loss or not config.algorithm.disable_kl or config.algorithm.online_filtering:
        raise ValueError("VideoTED uses unfiltered binary rewards without KL shaping")
    expected_truncation = "left" if aligned else "error"
    if data.truncation != expected_truncation or data.trajectory_key is not None:
        raise ValueError(f"Use teacher_context and data.truncation={expected_truncation}")

    if opsd.ar_reference_mode not in {"ce", "clipped"} or not 0 < opsd.ar_reference_clip < 1:
        raise ValueError("Invalid reference objective or clipping range")
    for name in ("ar_reference_coef", "ar_reference_kappa"):
        value = getattr(opsd, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"opsd.{name} must be finite and nonnegative")
    if opsd.ar_reference_max_length < 1:
        raise ValueError("Reference maximum length must be positive")
    credit_clip = opsd.stgca_reweight_clip_range
    if not 0 < (actor.clip_ratio_low if credit_clip is None else credit_clip) < 1:
        raise ValueError("The credit reweighting clip must be in (0, 1)")

    if aligned:
        if data.filter_overlong_prompts:
            raise ValueError("Disable prompt filtering so left truncation can retain every row")
        if opsd.ar_teacher_update_mode != "ema" or opsd.stgca_lambda_warmup_steps != 0:
            raise ValueError("MeSD alignment uses an EMA teacher and an annealing schedule without warmup")
        if reward_mode != "protocol":
            raise ValueError("MeSD alignment uses the joint class-and-format reward")
        if not 0 <= actor.clip_ratio_low < 1 or not math.isfinite(actor.clip_ratio_high) or actor.clip_ratio_high < 0:
            raise ValueError("Invalid policy clipping range")
        if config.worker.ref.offload.offload_params:
            raise ValueError("EMA requires the teacher parameters to stay on device")
        for name in ("enable_full_shard", "use_orig_params", "fsdp_size"):
            if getattr(actor.fsdp, name) != getattr(config.worker.ref.fsdp, name):
                raise ValueError(f"EMA requires matching actor and teacher FSDP {name}")
