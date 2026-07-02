"""RSL-RL config for the local Go2 LiDAR task."""

from __future__ import annotations

import os

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _set_float_attr(obj, attr: str, env_name: str) -> None:
    if hasattr(obj, attr):
        setattr(obj, attr, _env_float(env_name, getattr(obj, attr)))


def _set_int_attr(obj, attr: str, env_name: str) -> None:
    if hasattr(obj, attr):
        setattr(obj, attr, _env_int(env_name, getattr(obj, attr)))


def _checkpoint_save_interval() -> int:
    explicit = os.environ.get("GO2_SAVE_INTERVAL_ITERATIONS")
    if explicit:
        try:
            return max(1, int(explicit))
        except ValueError:
            pass

    num_envs = _env_int("GO2_NUM_ENVS", 4096)
    rollout_steps = _env_int("GO2_ROLLOUT_STEPS_PER_ENV", 24)
    spark_sps = _env_int("GO2_CHECKPOINT_SPS", 70_000)
    wallclock_seconds = _env_int("GO2_CHECKPOINT_SECONDS", 30 * 60)
    max_steps = _env_int("GO2_CHECKPOINT_MAX_STEPS", 1_000_000_000)

    target_steps = min(max_steps, spark_sps * wallclock_seconds)
    steps_per_iteration = max(1, num_envs * rollout_steps)
    return max(1, round(target_steps / steps_per_iteration))


@configclass
class Go2LidarFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = os.environ.get("GO2_EXPERIMENT_NAME", "go2_lidar_walk_flat")
        self.save_interval = _checkpoint_save_interval()
        # Extra range observations are still low dimensional, but larger than the stock flat task.
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        # RSL-RL's default scalar std is an unconstrained learnable parameter.
        # Use log-std so PPO cannot drive the Normal distribution scale negative.
        self.policy.noise_std_type = os.environ.get("GO2_NOISE_STD_TYPE", "log")
        self.policy.init_noise_std = _env_float("GO2_INIT_NOISE_STD", self.policy.init_noise_std)
        self.algorithm.learning_rate = _env_float("GO2_LEARNING_RATE", self.algorithm.learning_rate)
        self.algorithm.max_grad_norm = _env_float("GO2_MAX_GRAD_NORM", self.algorithm.max_grad_norm)
        _set_float_attr(self.algorithm, "entropy_coef", "GO2_ENTROPY_COEF")
        _set_float_attr(self.algorithm, "clip_param", "GO2_CLIP_PARAM")
        _set_float_attr(self.algorithm, "gamma", "GO2_GAMMA")
        _set_float_attr(self.algorithm, "lam", "GO2_GAE_LAMBDA")
        _set_float_attr(self.algorithm, "desired_kl", "GO2_DESIRED_KL")
        _set_float_attr(self.algorithm, "value_loss_coef", "GO2_VALUE_LOSS_COEF")
        _set_int_attr(self.algorithm, "num_learning_epochs", "GO2_NUM_LEARNING_EPOCHS")
        _set_int_attr(self.algorithm, "num_mini_batches", "GO2_NUM_MINI_BATCHES")
        self.num_steps_per_env = _env_int("GO2_ROLLOUT_STEPS_PER_ENV", self.num_steps_per_env)
