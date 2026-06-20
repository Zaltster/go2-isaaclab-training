"""Go2 flat locomotion task with real-layout LiDAR observations.

Sensor layout modeled from the user's robot:

- top LiDAR above/behind the head near the start of the spine
- front depth/LiDAR fan below the face
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
    UnitreeGo2FlatEnvCfg_PLAY,
)

from . import mdp


TOP_LIDAR_MAX_DISTANCE_M = 5.0
FRONT_DEPTH_LIDAR_MAX_DISTANCE_M = 3.0
GOAL_DISTANCE_M = 5.5
TOP_LIDAR_FOV_DEG = (-180.0, 180.0)
FRONT_DEPTH_LIDAR_FOV_DEG = (-45.0, 45.0)
TOP_LIDAR_RAYS = 24
FRONT_DEPTH_LIDAR_RAYS = 11
TOP_LIDAR_OFFSET_XY_M = (0.18, 0.0)
FRONT_DEPTH_LIDAR_OFFSET_XY_M = (0.31, 0.0)
OBSTACLE_APPROX_RADIUS_M = 0.42


def _obstacle_scene_cfgs() -> list[SceneEntityCfg]:
    return [
        SceneEntityCfg("obstacle_0"),
        SceneEntityCfg("obstacle_1"),
        SceneEntityCfg("obstacle_2"),
    ]


def _obstacle_cfg(name: str, pos: tuple[float, float, float], size: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.18, 0.08)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _add_static_obstacles(scene) -> None:
    scene.obstacle_0 = _obstacle_cfg("Obstacle_0", pos=(1.8, 0.0, 0.20), size=(0.28, 0.55, 0.40))
    scene.obstacle_1 = _obstacle_cfg("Obstacle_1", pos=(3.2, -0.45, 0.18), size=(0.35, 0.30, 0.36))
    scene.obstacle_2 = _obstacle_cfg("Obstacle_2", pos=(4.6, 0.45, 0.18), size=(0.35, 0.30, 0.36))


def _configure_lidar_navigation(env_cfg, debug_vis: bool) -> None:
    _add_static_obstacles(env_cfg.scene)
    obstacle_cfgs = _obstacle_scene_cfgs()

    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.85)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
    env_cfg.rewards.track_ang_vel_z_exp.weight = 0.20

    env_cfg.observations.policy.top_lidar_ranges = ObsTerm(
        func=mdp.analytic_lidar_ranges,
        params={
            "obstacle_cfgs": obstacle_cfgs,
            "max_distance": TOP_LIDAR_MAX_DISTANCE_M,
            "horizontal_fov_range": TOP_LIDAR_FOV_DEG,
            "num_rays": TOP_LIDAR_RAYS,
            "obstacle_radius": OBSTACLE_APPROX_RADIUS_M,
            "sensor_offset_xy": TOP_LIDAR_OFFSET_XY_M,
        },
        noise=None if debug_vis else Unoise(n_min=-0.01, n_max=0.01),
        clip=(0.0, 1.0),
    )
    env_cfg.observations.policy.front_depth_lidar_ranges = ObsTerm(
        func=mdp.analytic_lidar_ranges,
        params={
            "obstacle_cfgs": obstacle_cfgs,
            "max_distance": FRONT_DEPTH_LIDAR_MAX_DISTANCE_M,
            "horizontal_fov_range": FRONT_DEPTH_LIDAR_FOV_DEG,
            "num_rays": FRONT_DEPTH_LIDAR_RAYS,
            "obstacle_radius": OBSTACLE_APPROX_RADIUS_M,
            "sensor_offset_xy": FRONT_DEPTH_LIDAR_OFFSET_XY_M,
        },
        noise=None if debug_vis else Unoise(n_min=-0.01, n_max=0.01),
        clip=(0.0, 1.0),
    )

    env_cfg.events.randomize_obstacles = EventTerm(
        func=mdp.randomize_obstacle_layout,
        mode="reset",
        params={
            "obstacle_cfgs": obstacle_cfgs,
            "goal_distance": GOAL_DISTANCE_M,
            "curriculum_steps": 25_000_000,
        },
    )

    env_cfg.rewards.forward_progress = RewTerm(
        func=mdp.forward_progress,
        weight=2.5,
        params={"target_distance": GOAL_DISTANCE_M},
    )
    env_cfg.rewards.forward_velocity = RewTerm(
        func=mdp.forward_velocity_reward,
        weight=0.75,
        params={"target_velocity": 0.65},
    )
    env_cfg.rewards.goal_reached = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=8.0,
        params={"target_distance": GOAL_DISTANCE_M},
    )
    env_cfg.rewards.top_lidar_clearance = RewTerm(
        func=mdp.analytic_lidar_clearance_penalty,
        weight=-0.75,
        params={
            "obstacle_cfgs": obstacle_cfgs,
            "max_distance": TOP_LIDAR_MAX_DISTANCE_M,
            "safe_distance": 0.75,
            "critical_distance": 0.25,
            "horizontal_fov_range": (-90.0, 90.0),
            "num_rays": 13,
            "obstacle_radius": OBSTACLE_APPROX_RADIUS_M,
            "sensor_offset_xy": TOP_LIDAR_OFFSET_XY_M,
        },
    )
    env_cfg.rewards.front_depth_lidar_clearance = RewTerm(
        func=mdp.analytic_lidar_clearance_penalty,
        weight=-0.50,
        params={
            "obstacle_cfgs": obstacle_cfgs,
            "max_distance": FRONT_DEPTH_LIDAR_MAX_DISTANCE_M,
            "safe_distance": 0.55,
            "critical_distance": 0.18,
            "horizontal_fov_range": FRONT_DEPTH_LIDAR_FOV_DEG,
            "num_rays": FRONT_DEPTH_LIDAR_RAYS,
            "obstacle_radius": OBSTACLE_APPROX_RADIUS_M,
            "sensor_offset_xy": FRONT_DEPTH_LIDAR_OFFSET_XY_M,
        },
    )
    env_cfg.rewards.lateral_velocity = RewTerm(
        func=mdp.excessive_lateral_velocity_l2,
        weight=-0.04,
        params={"deadband": 0.25},
    )
    env_cfg.rewards.lateral_drift = RewTerm(
        func=mdp.lateral_drift_l1,
        weight=-0.04,
        params={"deadband": 0.95},
    )
    env_cfg.rewards.low_base_height = RewTerm(
        func=mdp.low_base_height_penalty,
        weight=-4.0,
        params={
            "min_height": 0.30,
            "margin": 0.12,
        },
    )
    env_cfg.rewards.non_foot_contact = RewTerm(
        func=mdp.non_foot_contact_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base|.*_hip|.*_thigh|.*_calf"),
            "threshold": 1.0,
        },
    )

    env_cfg.terminations.goal_reached = DoneTerm(
        func=mdp.goal_reached_termination,
        params={"target_distance": GOAL_DISTANCE_M},
    )
    env_cfg.terminations.no_progress = DoneTerm(
        func=mdp.no_progress_termination,
        params={
            "min_episode_steps": 150,
            "min_progress": 0.08,
        },
    )
    env_cfg.terminations.low_base_height = DoneTerm(
        func=mdp.low_base_height_termination,
        params={
            "min_height": 0.23,
            "min_episode_steps": 10,
        },
    )
    env_cfg.terminations.obstacle_contact = DoneTerm(
        func=mdp.non_foot_contact_termination,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base|.*_hip|.*_thigh|.*_calf"),
            "threshold": 1.0,
        },
    )


@configclass
class Go2LidarFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_lidar_navigation(self, debug_vis=False)


@configclass
class Go2LidarFlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _configure_lidar_navigation(self, debug_vis=True)
