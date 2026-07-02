"""Local IsaacLab task registrations for the Go2 restart."""

from __future__ import annotations

import gymnasium as gym

from . import agents


if "Go2-Lidar-Walk-Flat-v0" not in gym.envs.registry:
    gym.register(
        id="Go2-Lidar-Walk-Flat-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarFlatEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarFlatPPORunnerCfg",
        },
    )


if "Go2-Lidar-Walk-Flat-Play-v0" not in gym.envs.registry:
    gym.register(
        id="Go2-Lidar-Walk-Flat-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_lidar_env_cfg:Go2LidarFlatEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2LidarFlatPPORunnerCfg",
        },
    )
