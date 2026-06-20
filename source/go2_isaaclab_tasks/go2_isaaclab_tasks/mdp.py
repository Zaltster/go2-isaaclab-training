"""MDP terms for the local Go2 LiDAR task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster


def _forward_progress_m(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]


def _base_height_m(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def normalized_lidar_ranges(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, max_distance: float) -> torch.Tensor:
    """Return ray-hit distance normalized to [0, 1].

    A value near 0 means an object is very close to the sensor. A value near 1
    means the ray either hit at max range or did not hit a valid mesh.
    """

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_vectors = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    ranges = torch.linalg.norm(ray_vectors, dim=-1)
    ranges = torch.nan_to_num(ranges, nan=max_distance, posinf=max_distance, neginf=0.0)
    ranges = ranges.clamp(min=0.0, max=max_distance)
    return ranges / max_distance


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _analytic_lidar_ranges_m(
    env: ManagerBasedEnv,
    obstacle_cfgs: list[SceneEntityCfg],
    max_distance: float,
    horizontal_fov_range: tuple[float, float],
    num_rays: int,
    obstacle_radius: float,
    sensor_offset_xy: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fast 2D range sensor over the randomized obstacle centers.

    This intentionally avoids IsaacLab's mesh ray-caster during training. The
    policy still receives LiDAR-like range bins, while the simulator keeps real
    obstacle collision geometry for contact and video validation.
    """

    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    offset_x = float(sensor_offset_xy[0])
    offset_y = float(sensor_offset_xy[1])
    sensor_xy = torch.empty_like(root_pos[:, :2])
    sensor_xy[:, 0] = root_pos[:, 0] + cos_yaw * offset_x - sin_yaw * offset_y
    sensor_xy[:, 1] = root_pos[:, 1] + sin_yaw * offset_x + cos_yaw * offset_y

    ray_angles = torch.linspace(
        horizontal_fov_range[0],
        horizontal_fov_range[1],
        max(1, int(num_rays)),
        device=env.device,
        dtype=root_pos.dtype,
    )
    ray_angles = torch.deg2rad(ray_angles).unsqueeze(0)
    ranges = torch.full((env.num_envs, ray_angles.shape[1]), float(max_distance), device=env.device, dtype=root_pos.dtype)

    for obstacle_cfg in obstacle_cfgs:
        obstacle = env.scene[obstacle_cfg.name]
        rel_xy = obstacle.data.root_pos_w[:, :2] - sensor_xy
        center_distance = torch.linalg.norm(rel_xy, dim=-1).clamp(min=1.0e-6)
        bearing = _wrap_to_pi(torch.atan2(rel_xy[:, 1], rel_xy[:, 0]) - yaw)
        angular_radius = torch.asin(torch.clamp(torch.as_tensor(obstacle_radius, device=env.device) / center_distance, max=0.95))
        angle_error = torch.abs(_wrap_to_pi(ray_angles - bearing.unsqueeze(1)))
        hit_mask = angle_error <= angular_radius.unsqueeze(1)
        hit_distance = (center_distance - obstacle_radius).clamp(min=0.0, max=max_distance).unsqueeze(1)
        ranges = torch.where(hit_mask, torch.minimum(ranges, hit_distance), ranges)

    return ranges


def analytic_lidar_ranges(
    env: ManagerBasedEnv,
    obstacle_cfgs: list[SceneEntityCfg],
    max_distance: float,
    horizontal_fov_range: tuple[float, float],
    num_rays: int,
    obstacle_radius: float,
    sensor_offset_xy: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return analytic LiDAR ranges normalized to [0, 1]."""

    ranges = _analytic_lidar_ranges_m(
        env,
        obstacle_cfgs=obstacle_cfgs,
        max_distance=max_distance,
        horizontal_fov_range=horizontal_fov_range,
        num_rays=num_rays,
        obstacle_radius=obstacle_radius,
        sensor_offset_xy=sensor_offset_xy,
        asset_cfg=asset_cfg,
    )
    return ranges / max(max_distance, 1.0e-6)


def randomize_obstacle_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    obstacle_cfgs: list[SceneEntityCfg],
    goal_distance: float,
    curriculum_steps: int,
) -> None:
    """Randomize obstacle positions with an automatic easy-to-hard schedule.

    The first resets use a single centered obstacle with wide side space. As the
    global step counter rises, additional obstacles enter and lateral jitter grows.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    aggregate_steps = float(getattr(env, "common_step_counter", 0)) * float(env.num_envs)
    progress = min(aggregate_steps / max(float(curriculum_steps), 1.0), 1.0)
    num_active = 1
    if progress > 0.35:
        num_active = 2
    if progress > 0.70:
        num_active = 3

    base_x = torch.tensor([1.8, 3.25, 4.55], device=env.device)
    max_x_jitter = 0.10 + 0.35 * progress
    max_y_jitter = 0.08 + 0.42 * progress
    env_origins = env.scene.env_origins[env_ids]

    for obstacle_index, obstacle_cfg in enumerate(obstacle_cfgs):
        obstacle = env.scene[obstacle_cfg.name]
        root_state = obstacle.data.default_root_state[env_ids].clone()
        active = obstacle_index < num_active

        if active:
            x_jitter = (torch.rand(env_ids.numel(), device=env.device) * 2.0 - 1.0) * max_x_jitter
            y_jitter = (torch.rand(env_ids.numel(), device=env.device) * 2.0 - 1.0) * max_y_jitter
            if obstacle_index == 0:
                y_jitter *= 0.35
            root_state[:, 0] = env_origins[:, 0] + base_x[obstacle_index] + x_jitter
            root_state[:, 1] = env_origins[:, 1] + y_jitter
        else:
            root_state[:, 0] = env_origins[:, 0] + goal_distance + 8.0 + obstacle_index
            root_state[:, 1] = env_origins[:, 1]
        root_state[:, 2] = obstacle.data.default_root_state[env_ids, 2]
        root_state[:, 3:7] = obstacle.data.default_root_state[env_ids, 3:7]
        root_state[:, 7:] = 0.0

        obstacle.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def forward_progress(
    env: ManagerBasedEnv,
    target_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward being farther down the lane toward the goal."""

    progress = _forward_progress_m(env, asset_cfg=asset_cfg)
    return torch.clamp(progress / max(target_distance, 1.0e-6), min=0.0, max=1.0)


def forward_velocity_reward(
    env: ManagerBasedEnv,
    target_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward motion so clearance penalties do not teach freezing."""

    asset = env.scene[asset_cfg.name]
    forward_velocity = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=target_velocity)
    return forward_velocity / max(target_velocity, 1.0e-6)


def goal_reached_bonus(
    env: ManagerBasedEnv,
    target_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-step success bonus when the robot reaches the target zone."""

    progress = _forward_progress_m(env, asset_cfg=asset_cfg)
    return (progress >= target_distance).float()


def goal_reached_termination(
    env: ManagerBasedEnv,
    target_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode once the robot reaches the target zone."""

    progress = _forward_progress_m(env, asset_cfg=asset_cfg)
    return progress >= target_distance


def no_progress_termination(
    env: ManagerBasedEnv,
    min_episode_steps: int,
    min_progress: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate policies that stay near the start instead of navigating."""

    progress = _forward_progress_m(env, asset_cfg=asset_cfg)
    old_enough = env.episode_length_buf >= min_episode_steps
    return torch.logical_and(old_enough, progress < min_progress)


def low_base_height_penalty(
    env: ManagerBasedEnv,
    min_height: float,
    margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize trunk heights below the standing envelope.

    Contact sensors can miss visually obvious ground scraping depending on the
    body set and threshold. Height is a direct guard against crawling policies
    that still make forward progress.
    """

    height = _base_height_m(env, asset_cfg=asset_cfg)
    normalized_violation = ((float(min_height) - height) / max(float(margin), 1.0e-6)).clamp(0.0, 1.0)
    return normalized_violation.square()


def low_base_height_termination(
    env: ManagerBasedEnv,
    min_height: float,
    min_episode_steps: int = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the trunk is too low to count as standing/walking."""

    height = _base_height_m(env, asset_cfg=asset_cfg)
    old_enough = env.episode_length_buf >= int(min_episode_steps)
    return torch.logical_and(old_enough, height < float(min_height))


def lidar_clearance_penalty(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    max_distance: float,
    safe_distance: float,
    critical_distance: float,
    forward_only: bool = True,
    max_abs_vertical: float = 0.16,
) -> torch.Tensor:
    """Penalize close forward LiDAR returns before a collision happens.

    The vertical filter keeps downward rays that see the floor from dominating
    the obstacle penalty.
    """

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_vectors = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    ranges = torch.linalg.norm(ray_vectors, dim=-1)
    ranges = torch.nan_to_num(ranges, nan=max_distance, posinf=max_distance, neginf=0.0)
    ranges = ranges.clamp(min=0.0, max=max_distance)

    ray_dirs = sensor.ray_directions[0] if sensor.ray_directions.ndim == 3 else sensor.ray_directions
    ray_mask = torch.abs(ray_dirs[:, 2]) <= max_abs_vertical
    if forward_only:
        ray_mask = torch.logical_and(ray_mask, ray_dirs[:, 0] > 0.0)
    if not torch.any(ray_mask):
        ray_mask = torch.ones_like(ranges[0], dtype=torch.bool)

    min_range = torch.min(ranges[:, ray_mask], dim=1).values
    denom = max(safe_distance - critical_distance, 1.0e-6)
    normalized_violation = ((safe_distance - min_range) / denom).clamp(0.0, 1.0)
    return normalized_violation.square()


def analytic_lidar_clearance_penalty(
    env: ManagerBasedEnv,
    obstacle_cfgs: list[SceneEntityCfg],
    max_distance: float,
    safe_distance: float,
    critical_distance: float,
    horizontal_fov_range: tuple[float, float],
    num_rays: int,
    obstacle_radius: float,
    sensor_offset_xy: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize close analytic LiDAR returns before a collision happens."""

    ranges = _analytic_lidar_ranges_m(
        env,
        obstacle_cfgs=obstacle_cfgs,
        max_distance=max_distance,
        horizontal_fov_range=horizontal_fov_range,
        num_rays=num_rays,
        obstacle_radius=obstacle_radius,
        sensor_offset_xy=sensor_offset_xy,
        asset_cfg=asset_cfg,
    )
    min_range = torch.min(ranges, dim=1).values
    denom = max(safe_distance - critical_distance, 1.0e-6)
    normalized_violation = ((safe_distance - min_range) / denom).clamp(0.0, 1.0)
    return normalized_violation.square()


def lateral_velocity_l2(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize sideways body velocity for the straight-walk phase."""

    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def excessive_lateral_velocity_l2(
    env: ManagerBasedEnv,
    deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Allow avoidance sidesteps while discouraging excessive sideways motion."""

    asset = env.scene[asset_cfg.name]
    lateral_speed = torch.abs(asset.data.root_lin_vel_b[:, 1])
    return torch.square(torch.clamp(lateral_speed - deadband, min=0.0))


def lateral_drift_l1(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    deadband: float = 0.25,
) -> torch.Tensor:
    """Weakly penalize drifting away from the lane center."""

    asset = env.scene[asset_cfg.name]
    lateral_offset = torch.abs(asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1])
    return torch.clamp(lateral_offset - deadband, min=0.0)


def non_foot_contact_penalty(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize non-foot contact as an obstacle/body-scrape proxy.

    This mirrors IsaacLab's undesired contact pattern while leaving the default
    base-contact fall termination intact.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def non_foot_contact_termination(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Terminate on non-foot body contact.

    This is the first hard-failure rule for obstacle training. It intentionally
    allows normal foot contacts and catches body/leg scrapes on obstacles.
    """

    return non_foot_contact_penalty(env, sensor_cfg=sensor_cfg, threshold=threshold) > 0.0
