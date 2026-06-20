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
