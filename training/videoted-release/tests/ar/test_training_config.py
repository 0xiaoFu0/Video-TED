from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl.ar.schedule import teacher_schedule
from verl.ar.validation import validate_training_config
from verl.trainer.config import PPOConfig


def configuration(*overrides):
    config_path = Path(__file__).resolve().parents[2] / "configs/videoted.yaml"
    config = OmegaConf.merge(
        OmegaConf.structured(PPOConfig()),
        OmegaConf.load(config_path),
        OmegaConf.from_cli(list(overrides)),
    )
    config = OmegaConf.to_object(config)
    config.deep_post_init()
    return config


def test_training_recipe_uses_protocol_reward_and_ema_teacher():
    config = configuration()
    validate_training_config(config)
    assert config.worker.reward.reward_function_kwargs == {"reward_mode": "protocol"}
    assert config.opsd.ar_teacher_update_mode == "ema"
    assert config.opsd.ar_reference_coef > 0


def test_training_recipe_accepts_hyperparameter_overrides():
    config = configuration(
        "opsd.ar_teacher_update_rate=0.05",
        "opsd.rlsd_lambda=0.25",
        "opsd.ar_reference_coef=0.2",
    )
    validate_training_config(config)
    assert config.opsd.ar_teacher_update_rate == pytest.approx(0.05)
    assert teacher_schedule(config.opsd, 1) == (True, 0.25)
    assert config.opsd.ar_reference_coef == pytest.approx(0.2)


def test_rollout_batch_must_be_divisible_by_gpu_count():
    config = configuration("trainer.nnodes=4", "trainer.n_gpus_per_node=8")
    validate_training_config(config)
    config.data.mini_rollout_batch_size = 16
    with pytest.raises(ValueError, match="GPU count"):
        validate_training_config(config)


def test_teacher_credit_decays_through_the_cutoff():
    config = configuration().opsd
    cutoff = config.ar_teacher_disable_after_step
    assert teacher_schedule(config, 1) == (True, config.stgca_lambda)
    active, weight = teacher_schedule(config, cutoff)
    assert active and 0 < weight < config.stgca_lambda
    assert teacher_schedule(config, cutoff + 1) == (False, 0.0)
