#!/usr/bin/env python3
"""Train LiDAR navigation on top of frozen WTW Go2 walking weights.

This is the path intended for Spark-edeb when the low-level walker should come
from the external walk-these-ways Go2 repo, not from our gaitfix checkpoints.
Only the high-level nav actor/critic is optimized.  The WTW TorchScript modules
are loaded in eval/inference mode and never enter the optimizer.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclasses.dataclass
class WtwHierNavConfig:
    task: str = "Go2-Lidar-Walk-Flat-v0"
    experiment: str = "go2_wtw_hier_nav_frozen_walk"
    run_name: str = "smooth_sparkedeb_wtw_frozen_walk_nav"
    log_root: str = "/logs/wtw_hier_nav"
    artifact_dir: str = "/workspace/go2/artifacts"
    num_envs: int = 128
    seed: int = 2207
    max_updates: int = 20000
    wallclock_seconds: int = 21600
    rollout_steps: int = 24
    update_epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 1.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.16
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.004
    max_grad_norm: float = 0.5
    init_log_std: float = -0.7
    log_std_min: float = -4.0
    log_std_max: float = 0.7
    max_x_velocity: float = 0.65
    max_yaw_rate: float = 0.85
    slow_distance: float = 1.45
    stop_distance: float = 0.48
    clearance_distance: float = 0.85
    progress_reward: float = 4.0
    goal_reward: float = 12.0
    clearance_penalty: float = 2.0
    termination_penalty: float = 8.0
    smooth_penalty: float = 0.08
    intervention_penalty: float = 0.05
    time_penalty: float = 0.002
    goal_distance: float = 5.5
    checkpoint_interval: int = 5
    trace_interval: int = 5
    top_lidar_max_distance: float = 5.0
    front_lidar_max_distance: float = 3.0
    top_lidar_rays: int = 24
    front_lidar_rays: int = 11
    obstacle_radius: float = 0.42
    top_lidar_offset_x: float = 0.18
    front_lidar_offset_x: float = 0.31
    warmup_steps: int = 80
    use_env_reward: bool = True

    @classmethod
    def from_env(cls, args: argparse.Namespace) -> "WtwHierNavConfig":
        cfg = cls()
        cfg.task = args.task or os.environ.get("GO2_TASK", cfg.task)
        cfg.experiment = os.environ.get("GO2_WTW_HIER_NAV_EXPERIMENT", cfg.experiment)
        cfg.run_name = os.environ.get("GO2_WTW_HIER_NAV_RUN_NAME", cfg.run_name)
        cfg.log_root = os.environ.get("GO2_WTW_HIER_NAV_LOG_ROOT", cfg.log_root)
        cfg.artifact_dir = os.environ.get("GO2_WTW_ARTIFACT_DIR", cfg.artifact_dir)
        cfg.num_envs = args.num_envs or env_int("GO2_WTW_HIER_NAV_NUM_ENVS", cfg.num_envs)
        cfg.seed = env_int("GO2_WTW_HIER_NAV_SEED", cfg.seed)
        cfg.max_updates = args.max_updates or env_int("GO2_WTW_HIER_NAV_MAX_UPDATES", cfg.max_updates)
        cfg.wallclock_seconds = args.wallclock_seconds or env_int("GO2_WTW_HIER_NAV_WALLCLOCK_SECONDS", cfg.wallclock_seconds)
        cfg.rollout_steps = env_int("GO2_WTW_HIER_NAV_ROLLOUT_STEPS", cfg.rollout_steps)
        cfg.learning_rate = env_float("GO2_WTW_HIER_NAV_LEARNING_RATE", cfg.learning_rate)
        cfg.max_x_velocity = env_float("GO2_WTW_HIER_NAV_MAX_X_VELOCITY", cfg.max_x_velocity)
        cfg.max_yaw_rate = env_float("GO2_WTW_HIER_NAV_MAX_YAW_RATE", cfg.max_yaw_rate)
        cfg.slow_distance = env_float("GO2_WTW_HIER_NAV_SLOW_DISTANCE", cfg.slow_distance)
        cfg.stop_distance = env_float("GO2_WTW_HIER_NAV_STOP_DISTANCE", cfg.stop_distance)
        cfg.use_env_reward = env_bool("GO2_WTW_HIER_NAV_USE_ENV_REWARD", cfg.use_env_reward)
        return cfg


WTW_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
DEFAULT_DOF_POS = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
COMMANDS_SCALE = [2.0, 2.0, 0.25, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15, 0.3, 0.3, 1.0, 1.0, 1.0]
HIP_INDICES = [0, 3, 6, 9]
NUM_OBS = 70
HISTORY_LENGTH = 30
HISTORY_DIM = NUM_OBS * HISTORY_LENGTH
CONTROL_DT = 0.02


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")


def write_json(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(item, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reorder_indices(source_names: list[str], target_names: list[str]) -> list[int]:
    source = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source]
    if missing:
        raise RuntimeError(f"missing required Go2 joints: {missing}; available={source_names}")
    return [source[name] for name in target_names]


def yaw_from_quat_wxyz(torch, quat):
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def base_command(torch, num_envs: int, device):
    command = torch.zeros((num_envs, 15), dtype=torch.float32, device=device)
    command[:, 4] = 3.0
    command[:, 5] = 0.5
    command[:, 8] = 0.5
    command[:, 9] = 0.09
    command[:, 12] = 0.25
    command[:, 13] = 0.40
    return command


def update_clock(torch, gait_index, command):
    gait_index = torch.remainder(gait_index + CONTROL_DT * command[:, 4], 1.0)
    foot_indices = torch.stack(
        [
            gait_index + command[:, 5] + command[:, 6] + command[:, 7],
            gait_index + command[:, 6],
            gait_index + command[:, 7],
            gait_index + command[:, 5],
        ],
        dim=-1,
    )
    return gait_index, torch.sin(2.0 * math.pi * foot_indices)


def import_isaac(args: argparse.Namespace):
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    return app_launcher.app


def build_env(args: argparse.Namespace, cfg: WtwHierNavConfig):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import go2_isaaclab_tasks  # noqa: F401

    env_cfg = parse_env_cfg(
        cfg.task,
        device=args.device,
        num_envs=cfg.num_envs,
        use_fabric=not getattr(args, "disable_fabric", False),
    )
    env_cfg.sim.render_interval = 2
    return gym.make(cfg.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)


def load_wtw_modules(torch, cfg: WtwHierNavConfig, device):
    artifact_dir = Path(cfg.artifact_dir)
    adaptation_path = artifact_dir / "adaptation_module_latest.jit"
    body_path = artifact_dir / "body_latest.jit"
    if not adaptation_path.exists() or not body_path.exists():
        raise FileNotFoundError(f"missing WTW artifacts under {artifact_dir}")
    adaptation = torch.jit.load(str(adaptation_path), map_location=device).eval()
    body = torch.jit.load(str(body_path), map_location=device).eval()
    for module in (adaptation, body):
        for param in module.parameters():
            param.requires_grad_(False)
    return adaptation, body, adaptation_path, body_path


def create_nav_model(torch, nn, obs_dim: int, action_dim: int, cfg: WtwHierNavConfig):
    class NavActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 1),
            )
            self.log_std = nn.Parameter(torch.full((action_dim,), float(cfg.init_log_std)))

        def forward(self, obs):
            mean = torch.clamp(self.actor(obs), -3.0, 3.0)
            value = self.critic(obs).squeeze(-1)
            log_std = torch.clamp(self.log_std, cfg.log_std_min, cfg.log_std_max)
            return mean, log_std, value

    return NavActorCritic()


class WtwIsaacAdapter:
    def __init__(self, torch, cfg: WtwHierNavConfig, env, device):
        from isaaclab.managers import SceneEntityCfg
        from go2_isaaclab_tasks import mdp

        self.torch = torch
        self.cfg = cfg
        self.env = env
        self.base_env = env.unwrapped
        self.device = device
        self.SceneEntityCfg = SceneEntityCfg
        self.mdp = mdp
        self.obstacle_cfgs = [SceneEntityCfg("obstacle_0"), SceneEntityCfg("obstacle_1"), SceneEntityCfg("obstacle_2")]
        self.robot = self.base_env.scene["robot"]
        action_term = self.base_env.action_manager.get_term("joint_pos")
        robot_joint_names = list(self.robot.data.joint_names)
        action_joint_names = list(action_term._joint_names)
        self.wtw_indices_from_robot = reorder_indices(robot_joint_names, WTW_JOINT_NAMES)
        self.action_indices_from_wtw = reorder_indices(WTW_JOINT_NAMES, action_joint_names)
        self.wtw_indices_from_robot_t = torch.tensor(self.wtw_indices_from_robot, dtype=torch.long, device=device)
        self.action_indices_from_wtw_t = torch.tensor(self.action_indices_from_wtw, dtype=torch.long, device=device)
        self.default_dof_pos = torch.tensor(DEFAULT_DOF_POS, dtype=torch.float32, device=device)
        self.command_scale = torch.tensor(COMMANDS_SCALE, dtype=torch.float32, device=device)
        self.hip_indices = torch.tensor(HIP_INDICES, dtype=torch.long, device=device)
        self.history = torch.zeros((cfg.num_envs, HISTORY_LENGTH, NUM_OBS), dtype=torch.float32, device=device)
        self.action_wtw = torch.zeros((cfg.num_envs, 12), dtype=torch.float32, device=device)
        self.last_action_wtw = torch.zeros((cfg.num_envs, 12), dtype=torch.float32, device=device)
        self.gait_index = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
        self.clock_inputs = torch.zeros((cfg.num_envs, 4), dtype=torch.float32, device=device)
        self.prev_command = torch.zeros((cfg.num_envs, 2), dtype=torch.float32, device=device)
        self.prev_progress = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
        self.steps = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
        self.reset_state(torch.arange(cfg.num_envs, device=device), refresh_progress=True)

    def reset_state(self, ids, refresh_progress: bool = False):
        if ids.numel() == 0:
            return
        self.history[ids] = 0.0
        self.action_wtw[ids] = 0.0
        self.last_action_wtw[ids] = 0.0
        self.gait_index[ids] = 0.0
        self.clock_inputs[ids] = 0.0
        self.prev_command[ids] = 0.0
        self.steps[ids] = 0
        if refresh_progress:
            self.prev_progress[ids] = self.progress()[ids]
        else:
            self.prev_progress[ids] = 0.0

    def progress(self):
        return self.robot.data.root_pos_w[:, 0] - self.base_env.scene.env_origins[:, 0]

    def lidar_m(self, max_distance: float, fov: tuple[float, float], rays: int, offset_x: float):
        return self.mdp._analytic_lidar_ranges_m(
            self.base_env,
            obstacle_cfgs=self.obstacle_cfgs,
            max_distance=max_distance,
            horizontal_fov_range=fov,
            num_rays=rays,
            obstacle_radius=self.cfg.obstacle_radius,
            sensor_offset_xy=(offset_x, 0.0),
        )

    def high_obs(self):
        torch = self.torch
        root_pos = self.robot.data.root_pos_w
        root_xy = root_pos[:, :2] - self.base_env.scene.env_origins[:, :2]
        yaw = yaw_from_quat_wxyz(torch, self.robot.data.root_quat_w)
        top = self.lidar_m(
            self.cfg.top_lidar_max_distance,
            (-180.0, 180.0),
            self.cfg.top_lidar_rays,
            self.cfg.top_lidar_offset_x,
        ) / self.cfg.top_lidar_max_distance
        front = self.lidar_m(
            self.cfg.front_lidar_max_distance,
            (-45.0, 45.0),
            self.cfg.front_lidar_rays,
            self.cfg.front_lidar_offset_x,
        ) / self.cfg.front_lidar_max_distance
        dx_w = self.cfg.goal_distance - root_xy[:, 0]
        dy_w = -root_xy[:, 1]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        dx_b = cos_yaw * dx_w + sin_yaw * dy_w
        dy_b = -sin_yaw * dx_w + cos_yaw * dy_w
        goal_dist = torch.sqrt(dx_w.square() + dy_w.square()).clamp_min(1.0e-6)
        goal_heading = torch.atan2(dy_b, dx_b)
        extras = torch.stack(
            [
                torch.clamp(dx_b / self.cfg.goal_distance, -1.5, 1.5),
                torch.clamp(dy_b / 2.0, -2.0, 2.0),
                torch.clamp(goal_dist / self.cfg.goal_distance, 0.0, 2.0),
                torch.sin(goal_heading),
                torch.cos(goal_heading),
                self.prev_command[:, 0],
                self.prev_command[:, 1],
                torch.sin(yaw),
                torch.cos(yaw),
            ],
            dim=-1,
        )
        return torch.cat([top, front, extras], dim=-1)

    def action_to_command(self, action, front_min):
        torch = self.torch
        desired_x = (action[:, 0] + 1.0) * 0.5 * self.cfg.max_x_velocity
        desired_yaw = action[:, 1] * self.cfg.max_yaw_rate
        slow_band = max(self.cfg.slow_distance - self.cfg.stop_distance, 1.0e-4)
        scale = torch.ones_like(desired_x)
        scale = torch.where(
            front_min < self.cfg.slow_distance,
            torch.clamp((front_min - self.cfg.stop_distance) / slow_band, min=0.0, max=1.0),
            scale,
        )
        safe_x = desired_x * scale
        intervention = (safe_x - desired_x).abs() > 1.0e-4
        return safe_x, desired_yaw, desired_x, intervention

    def wtw_observation(self, command):
        joint_pos = self.robot.data.joint_pos.index_select(1, self.wtw_indices_from_robot_t)
        joint_vel = self.robot.data.joint_vel.index_select(1, self.wtw_indices_from_robot_t)
        obs = torch.cat(
            [
                self.robot.data.projected_gravity_b,
                command * self.command_scale,
                joint_pos - self.default_dof_pos,
                joint_vel * 0.05,
                torch.clamp(self.action_wtw, -10.0, 10.0),
                torch.clamp(self.last_action_wtw, -10.0, 10.0),
                self.clock_inputs,
            ],
            dim=-1,
        )
        if obs.shape[-1] != NUM_OBS:
            raise RuntimeError(f"bad WTW observation shape: {tuple(obs.shape)}")
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1] = obs
        return self.history.reshape(command.shape[0], HISTORY_DIM)

    def frozen_wtw_action(self, torch, adaptation, body, command, wtw_device, trace=None):
        if trace is not None:
            trace("clock_start")
        self.gait_index, self.clock_inputs = update_clock(torch, self.gait_index, command)
        if trace is not None:
            trace("obs_start")
        obs_history = self.wtw_observation(command)
        if trace is not None:
            trace("obs_done")
        obs_history_wtw = obs_history.to(wtw_device)
        if trace is not None:
            trace("adaptation_start")
        latent = adaptation(obs_history_wtw)
        if trace is not None:
            trace("adaptation_done")
            trace("body_start")
        next_action = body(torch.cat((obs_history_wtw, latent), dim=-1))
        if trace is not None:
            trace("body_done")
        next_action_wtw = torch.clamp(next_action[:, :12].to(self.device), -10.0, 10.0)
        if trace is not None:
            trace("postprocess_start")
        raw_action_wtw = next_action_wtw.clone()
        raw_action_wtw[:, self.hip_indices] *= 0.5
        env_action = raw_action_wtw.index_select(1, self.action_indices_from_wtw_t)
        self.last_action_wtw = self.action_wtw
        self.action_wtw = next_action_wtw
        if trace is not None:
            trace("postprocess_done")
        return env_action

    def reward(self, compact_command, desired_x, intervention, terminated, truncated):
        torch = self.torch
        progress_now = self.progress()
        progress_delta = progress_now - self.prev_progress
        self.prev_progress = progress_now.detach()
        front_min = self.lidar_m(
            self.cfg.front_lidar_max_distance,
            (-45.0, 45.0),
            self.cfg.front_lidar_rays,
            self.cfg.front_lidar_offset_x,
        ).amin(dim=1)
        full_min = self.lidar_m(
            self.cfg.top_lidar_max_distance,
            (-180.0, 180.0),
            self.cfg.top_lidar_rays,
            self.cfg.top_lidar_offset_x,
        ).amin(dim=1)
        goal = progress_now >= self.cfg.goal_distance
        done = terminated.bool() | truncated.bool() | goal
        bad_done = done & ~goal
        command_delta = (compact_command - self.prev_command).square().sum(dim=1)
        clearance_shortfall = torch.clamp(self.cfg.clearance_distance - full_min, min=0.0) / self.cfg.clearance_distance
        reward = (
            self.cfg.progress_reward * progress_delta
            + goal.float() * self.cfg.goal_reward
            - bad_done.float() * self.cfg.termination_penalty
            - clearance_shortfall * self.cfg.clearance_penalty
            - command_delta * self.cfg.smooth_penalty
            - intervention.float() * self.cfg.intervention_penalty
            - self.cfg.time_penalty
        )
        self.prev_command = torch.where(done[:, None], torch.zeros_like(compact_command), compact_command.detach())
        reset_ids = torch.nonzero(done, as_tuple=False).flatten()
        self.reset_state(reset_ids, refresh_progress=False)
        return reward, done.float(), {
            "goal": goal.float(),
            "bad_done": bad_done.float(),
            "front_min": front_min,
            "full_min": full_min,
            "desired_x": desired_x,
            "safe_x": compact_command[:, 0],
            "intervention": intervention.float(),
            "progress_delta": progress_delta,
        }


def squash_log_prob(torch, dist, raw_action, action):
    correction = torch.log(torch.clamp(1.0 - action.square(), min=1.0e-6))
    return (dist.log_prob(raw_action) - correction).sum(dim=-1)


def assert_wtw_frozen(adaptation, body, optimizer) -> None:
    frozen_ids = {id(param) for module in (adaptation, body) for param in module.parameters()}
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) in frozen_ids:
                raise RuntimeError("WTW walking parameter was added to the nav optimizer.")
    trainable = [param for module in (adaptation, body) for param in module.parameters() if param.requires_grad]
    if trainable:
        raise RuntimeError(f"WTW walking modules still have {len(trainable)} trainable parameters.")
    if adaptation.training or body.training:
        raise RuntimeError("WTW walking modules are not in eval mode.")


def save_nav_checkpoint(torch, path: Path, model, optimizer, cfg: WtwHierNavConfig, update: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": dataclasses.asdict(cfg),
            "update": update,
            "metrics": metrics,
            "frozen_low_level_policy": "walk-these-ways-go2-torchscript",
        },
        path,
    )


def train(args: argparse.Namespace) -> int:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal

    cfg = WtwHierNavConfig.from_env(args)
    torch.set_num_threads(env_int("GO2_WTW_HIER_NAV_TORCH_THREADS", 1))
    torch.set_num_interop_threads(env_int("GO2_WTW_HIER_NAV_TORCH_INTEROP_THREADS", 1))
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = build_env(args, cfg)
    sim_device = torch.device(str(env.unwrapped.device))
    nav_device = torch.device(os.environ.get("GO2_WTW_HIER_NAV_LEARN_DEVICE", "cpu"))
    wtw_device = torch.device(os.environ.get("GO2_WTW_HIER_NAV_WTW_DEVICE", "cpu"))
    env.reset()
    adaptation, body, adaptation_path, body_path = load_wtw_modules(torch, cfg, wtw_device)
    adapter = WtwIsaacAdapter(torch, cfg, env, sim_device)
    high_obs = adapter.high_obs().to(nav_device)
    obs_dim = high_obs.shape[1]
    action_dim = 2
    nav_model = create_nav_model(torch, nn, obs_dim, action_dim, cfg).to(nav_device)
    optimizer = torch.optim.Adam(nav_model.parameters(), lr=cfg.learning_rate)
    assert_wtw_frozen(adaptation, body, optimizer)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.log_root) / cfg.experiment / f"{timestamp}_{cfg.run_name}"
    ckpt_dir = run_dir / "checkpoints"
    trace_dir = run_dir / "traces"
    metrics_path = run_dir / "metrics.jsonl"
    progress_path = run_dir / "progress.log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", dataclasses.asdict(cfg))
    write_json(
        run_dir / "frozen_wtw_manifest.json",
        {
            "source": "walk-these-ways-go2",
            "adaptation_module": str(adaptation_path),
            "adaptation_module_sha256": sha256_file(adaptation_path),
            "body": str(body_path),
            "body_sha256": sha256_file(body_path),
            "optimizer": "nav_policy_only",
            "trainable_wtw_params": 0,
        },
    )
    append_jsonl(
        metrics_path,
        {
            "event": "startup_complete",
            "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sim_device": str(sim_device),
            "nav_device": str(nav_device),
            "wtw_device": str(wtw_device),
        },
    )

    start = time.time()
    last_metrics: dict[str, Any] = {}

    for update in range(1, cfg.max_updates + 1):
        assert_wtw_frozen(adaptation, body, optimizer)
        append_jsonl(metrics_path, {"event": "update_start", "update": update, "elapsed_seconds": round(time.time() - start, 3)})
        obs_buf = torch.zeros((cfg.rollout_steps, cfg.num_envs, obs_dim), device=nav_device)
        raw_buf = torch.zeros((cfg.rollout_steps, cfg.num_envs, action_dim), device=nav_device)
        action_buf = torch.zeros((cfg.rollout_steps, cfg.num_envs, action_dim), device=nav_device)
        logprob_buf = torch.zeros((cfg.rollout_steps, cfg.num_envs), device=nav_device)
        reward_buf = torch.zeros((cfg.rollout_steps, cfg.num_envs), device=nav_device)
        done_buf = torch.zeros((cfg.rollout_steps, cfg.num_envs), device=nav_device)
        value_buf = torch.zeros((cfg.rollout_steps, cfg.num_envs), device=nav_device)
        rollout_stats: dict[str, list[Any]] = {key: [] for key in ("goal", "bad_done", "front_min", "full_min", "desired_x", "safe_x", "intervention", "progress_delta")}

        for step in range(cfg.rollout_steps):
            global_policy_step = (update - 1) * cfg.rollout_steps + step
            trace_step = update <= 3 or step == 0
            if trace_step:
                append_jsonl(metrics_path, {"event": "rollout_step_start", "update": update, "step": step, "elapsed_seconds": round(time.time() - start, 3)})
            with torch.no_grad():
                mean, log_std, value = nav_model(high_obs)
                dist = Normal(mean, log_std.exp())
                raw_action = dist.rsample()
                action = torch.tanh(raw_action)
                logprob = squash_log_prob(torch, dist, raw_action, action)
                action_sim = action.to(sim_device)
                front_min = adapter.lidar_m(
                    cfg.front_lidar_max_distance,
                    (-45.0, 45.0),
                    cfg.front_lidar_rays,
                    cfg.front_lidar_offset_x,
                ).amin(dim=1)
                safe_x, desired_yaw, desired_x, intervention = adapter.action_to_command(action_sim, front_min)
                command = base_command(torch, cfg.num_envs, sim_device)
                command[:, 0] = safe_x
                command[:, 2] = desired_yaw
                compact_command = torch.stack([safe_x, desired_yaw], dim=-1)
                if global_policy_step < cfg.warmup_steps:
                    if trace_step:
                        append_jsonl(
                            metrics_path,
                            {
                                "event": "wtw_warmup_zero_action",
                                "update": update,
                                "step": step,
                                "global_policy_step": global_policy_step,
                                "warmup_steps": cfg.warmup_steps,
                                "elapsed_seconds": round(time.time() - start, 3),
                            },
                        )
                    env_action = torch.zeros((cfg.num_envs, 12), dtype=torch.float32, device=sim_device)
                else:
                    if trace_step:
                        append_jsonl(metrics_path, {"event": "wtw_action_start", "update": update, "step": step, "elapsed_seconds": round(time.time() - start, 3), "wtw_device": str(wtw_device)})

                    def trace_wtw(phase: str) -> None:
                        if trace_step:
                            append_jsonl(
                                metrics_path,
                                {
                                    "event": f"wtw_{phase}",
                                    "update": update,
                                    "step": step,
                                    "elapsed_seconds": round(time.time() - start, 3),
                                },
                            )

                    env_action = adapter.frozen_wtw_action(torch, adaptation, body, command, wtw_device, trace_wtw)
                    if trace_step:
                        append_jsonl(metrics_path, {"event": "wtw_action_done", "update": update, "step": step, "elapsed_seconds": round(time.time() - start, 3)})
            _obs, _env_reward, terminated, truncated, _info = env.step(env_action)
            if trace_step:
                append_jsonl(metrics_path, {"event": "rollout_step_done", "update": update, "step": step, "elapsed_seconds": round(time.time() - start, 3)})
            if cfg.use_env_reward:
                front_from_obs = high_obs[:, cfg.top_lidar_rays : cfg.top_lidar_rays + cfg.front_lidar_rays] * cfg.front_lidar_max_distance
                full_from_obs = high_obs[:, : cfg.top_lidar_rays] * cfg.top_lidar_max_distance
                front_min_nav = front_from_obs.amin(dim=1)
                full_min_nav = full_from_obs.amin(dim=1)
                desired_x_nav = (action[:, 0] + 1.0) * 0.5 * cfg.max_x_velocity
                slow_band_nav = max(cfg.slow_distance - cfg.stop_distance, 1.0e-4)
                safe_scale_nav = torch.where(
                    front_min_nav < cfg.slow_distance,
                    torch.clamp((front_min_nav - cfg.stop_distance) / slow_band_nav, min=0.0, max=1.0),
                    torch.ones_like(front_min_nav),
                )
                safe_x_nav = desired_x_nav * safe_scale_nav
                clearance_shortfall_nav = torch.clamp(cfg.clearance_distance - full_min_nav, min=0.0) / cfg.clearance_distance
                intervention_nav = (safe_x_nav - desired_x_nav).abs() > 1.0e-4
                reward = (
                    safe_x_nav * cfg.progress_reward * CONTROL_DT
                    - clearance_shortfall_nav * cfg.clearance_penalty
                    - intervention_nav.float() * cfg.intervention_penalty
                    - cfg.time_penalty
                )
                done = torch.zeros(cfg.num_envs, dtype=torch.float32, device=nav_device)
                info = {
                    "goal": torch.zeros_like(done),
                    "bad_done": done,
                    "front_min": front_min_nav.detach(),
                    "full_min": full_min_nav.detach(),
                    "desired_x": desired_x_nav.detach(),
                    "safe_x": safe_x_nav.detach(),
                    "intervention": intervention_nav.float().detach(),
                    "progress_delta": reward.detach(),
                }
            else:
                reward, done, info = adapter.reward(compact_command, desired_x, intervention, terminated, truncated)
            if trace_step:
                append_jsonl(metrics_path, {"event": "reward_done", "update": update, "step": step, "elapsed_seconds": round(time.time() - start, 3), "use_env_reward": cfg.use_env_reward})
            if trace_step:
                append_jsonl(metrics_path, {"event": "high_obs_start", "update": update, "step": step, "elapsed_seconds": round(time.time() - start, 3)})
            next_high_obs = adapter.high_obs().to(nav_device)
            if trace_step:
                append_jsonl(metrics_path, {"event": "high_obs_done", "update": update, "step": step, "elapsed_seconds": round(time.time() - start, 3)})

            obs_buf[step] = high_obs
            raw_buf[step] = raw_action
            action_buf[step] = action
            logprob_buf[step] = logprob
            reward_buf[step] = reward.to(nav_device)
            done_buf[step] = done.to(nav_device)
            value_buf[step] = value
            for key in rollout_stats:
                rollout_stats[key].append(info[key].detach())
            high_obs = next_high_obs

        with torch.no_grad():
            _, _, next_value = nav_model(high_obs)
            advantages = torch.zeros_like(reward_buf)
            last_gae = torch.zeros(cfg.num_envs, device=nav_device)
            for step in reversed(range(cfg.rollout_steps)):
                next_nonterminal = 1.0 - done_buf[step]
                next_values = next_value if step == cfg.rollout_steps - 1 else value_buf[step + 1]
                delta = reward_buf[step] + cfg.gamma * next_values * next_nonterminal - value_buf[step]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                advantages[step] = last_gae
            returns = advantages + value_buf
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        batch_size = cfg.rollout_steps * cfg.num_envs
        minibatch_size = max(1, batch_size // cfg.minibatches)
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_raw = raw_buf.reshape(-1, action_dim)
        b_action = action_buf.reshape(-1, action_dim)
        b_logprob = logprob_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_buf.reshape(-1)
        append_jsonl(metrics_path, {"event": "ppo_start", "update": update, "elapsed_seconds": round(time.time() - start, 3), "learn_device": str(nav_device)})
        indices = torch.randperm(batch_size, device=nav_device)
        losses = []
        grad_norms = []
        clip_fracs = []
        for _epoch in range(cfg.update_epochs):
            for start_idx in range(0, batch_size, minibatch_size):
                if update == 1:
                    append_jsonl(metrics_path, {"event": "ppo_minibatch_start", "update": update, "epoch": _epoch, "start_idx": start_idx, "elapsed_seconds": round(time.time() - start, 3)})
                mb = indices[start_idx : start_idx + minibatch_size]
                mean, log_std, new_value = nav_model(b_obs[mb])
                dist = Normal(mean, log_std.exp())
                new_logprob = squash_log_prob(torch, dist, b_raw[mb], b_action[mb])
                ratio = (new_logprob - b_logprob[mb]).exp()
                pg_loss = torch.max(
                    -b_adv[mb] * ratio,
                    -b_adv[mb] * torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param),
                ).mean()
                value_clipped = b_values[mb] + (new_value - b_values[mb]).clamp(-cfg.clip_param, cfg.clip_param)
                value_loss = 0.5 * torch.max((new_value - b_returns[mb]).square(), (value_clipped - b_returns[mb]).square()).mean()
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = pg_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(nav_model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    nav_model.log_std.clamp_(cfg.log_std_min, cfg.log_std_max)
                    losses.append(loss.detach())
                    grad_norms.append(torch.as_tensor(grad_norm, device=nav_device).detach())
                    clip_fracs.append(((ratio - 1.0).abs() > cfg.clip_param).float().mean().detach())
                if update == 1:
                    append_jsonl(metrics_path, {"event": "ppo_minibatch_done", "update": update, "epoch": _epoch, "start_idx": start_idx, "elapsed_seconds": round(time.time() - start, 3)})
        append_jsonl(metrics_path, {"event": "ppo_done", "update": update, "elapsed_seconds": round(time.time() - start, 3), "learn_device": str(nav_device)})

        elapsed = time.time() - start

        def stat_mean(key: str) -> float:
            values = torch.cat([x.reshape(-1) for x in rollout_stats[key]])
            return float(values.mean().item())

        metrics = {
            "update": update,
            "elapsed_seconds": round(elapsed, 3),
            "fps": round((update * cfg.rollout_steps * cfg.num_envs) / max(elapsed, 1.0e-6), 2),
            "mean_reward": float(reward_buf.mean().item()),
            "goal_rate": stat_mean("goal"),
            "bad_done_rate": stat_mean("bad_done"),
            "front_min_mean": stat_mean("front_min"),
            "full_min_mean": stat_mean("full_min"),
            "desired_x_mean": stat_mean("desired_x"),
            "safe_x_mean": stat_mean("safe_x"),
            "intervention_rate": stat_mean("intervention"),
            "progress_delta_mean": stat_mean("progress_delta"),
            "loss": float(torch.stack(losses).mean().item()),
            "grad_norm": float(torch.stack(grad_norms).mean().item()),
            "clip_fraction": float(torch.stack(clip_fracs).mean().item()),
            "log_std_mean": float(nav_model.log_std.mean().item()),
            "trainable_wtw_params": sum(1 for module in (adaptation, body) for p in module.parameters() if p.requires_grad),
        }
        last_metrics = metrics
        append_jsonl(metrics_path, metrics)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(
                "update={update} reward={reward:.4f} goal={goal:.4f} bad_done={bad_done:.4f} "
                "front_min={front:.3f} safe_x={safe_x:.3f} intervention={intervention:.4f} fps={fps:.1f}\n".format(
                    update=update,
                    reward=metrics["mean_reward"],
                    goal=metrics["goal_rate"],
                    bad_done=metrics["bad_done_rate"],
                    front=metrics["front_min_mean"],
                    safe_x=metrics["safe_x_mean"],
                    intervention=metrics["intervention_rate"],
                    fps=metrics["fps"],
                )
            )
        write_json(run_dir / "heartbeat.json", metrics | {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        if update % cfg.trace_interval == 0:
            write_json(trace_dir / f"trace_update_{update:07d}.json", {"update": update, "metrics": metrics})
        if update % cfg.checkpoint_interval == 0:
            save_nav_checkpoint(torch, ckpt_dir / f"nav_model_update_{update:07d}.pt", nav_model, optimizer, cfg, update, metrics)
            save_nav_checkpoint(torch, ckpt_dir / "latest.pt", nav_model, optimizer, cfg, update, metrics)
        if elapsed >= cfg.wallclock_seconds:
            append_jsonl(metrics_path, {"event": "wallclock_stop", "update": update, "elapsed_seconds": elapsed})
            break

    save_nav_checkpoint(torch, ckpt_dir / "final.pt", nav_model, optimizer, cfg, update, last_metrics)
    write_json(run_dir / "DONE.json", {"status": "complete", "final_update": update, "metrics": last_metrics})
    env.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--wallclock-seconds", type=int, default=None)
    parser.add_argument("--video", action="store_true", default=env_bool("GO2_WTW_HIER_NAV_VIDEO", False))
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    simulation_app = None
    try:
        simulation_app = import_isaac(args)
        return train(args)
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
