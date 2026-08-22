#!/usr/bin/env python3
"""Visualize a task's curriculum terrain without creating any robots."""

import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Isaac Gym must be imported before torch-backed legged_gym modules.
from isaacgym import gymapi, gymutil

from legged_gym.envs import *  # noqa: F401,F403 - registers task configs
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, parse_sim_params
from legged_gym.utils.terrain import Terrain


def ensure_default_task():
    """Use W1W unless the command line explicitly selects another task."""
    has_task = any(
        argument == "--task" or argument.startswith("--task=")
        for argument in sys.argv[1:]
    )
    if not has_task:
        sys.argv.append("--task=w1w")


def add_terrain_mesh(gym, sim, terrain):
    params = gymapi.TriangleMeshParams()
    params.nb_vertices = terrain.vertices.shape[0]
    params.nb_triangles = terrain.triangles.shape[0]
    params.transform.p.x = -terrain.cfg.border_size
    params.transform.p.y = -terrain.cfg.border_size
    params.transform.p.z = 0.0
    params.static_friction = terrain.cfg.static_friction
    params.dynamic_friction = terrain.cfg.dynamic_friction
    params.restitution = terrain.cfg.restitution
    gym.add_triangle_mesh(
        sim,
        terrain.vertices.flatten(order="C"),
        terrain.triangles.flatten(order="C"),
        params,
    )


def curriculum_values(cfg, level):
    """Return the same normalized difficulty and limits used by Terrain."""
    last_level = max(cfg.num_rows - 1, 1)
    ratio = level / last_level

    if hasattr(cfg, "max_smooth_slope_angle_deg"):
        min_smooth_angle = getattr(
            cfg, "min_smooth_slope_angle_deg", 0.0
        )
        smooth_slope_angle = min_smooth_angle + ratio * (
            cfg.max_smooth_slope_angle_deg - min_smooth_angle
        )
    elif hasattr(cfg, "max_slope_angle_deg"):
        smooth_slope_angle = np.degrees(np.arctan(
            ratio * np.tan(np.deg2rad(cfg.max_slope_angle_deg))
        ))
    else:
        raw_difficulty = level / cfg.num_rows
        smooth_slope_angle = np.degrees(
            np.arctan(raw_difficulty * 0.4)
        )

    if hasattr(cfg, "max_slope_angle_deg"):
        rough_slope_angle = np.degrees(np.arctan(
            ratio * np.tan(np.deg2rad(cfg.max_slope_angle_deg))
        ))
    else:
        rough_slope_angle = np.degrees(
            np.arctan((level / cfg.num_rows) * 0.4)
        )

    if hasattr(cfg, "max_step_height"):
        step_height = cfg.min_step_height + ratio * (
            cfg.max_step_height - cfg.min_step_height
        )
    else:
        step_height = 0.05 + 0.18 * (level / cfg.num_rows)

    if hasattr(cfg, "max_discrete_obstacle_height"):
        obstacle_height = cfg.min_discrete_obstacle_height + ratio * (
            cfg.max_discrete_obstacle_height
            - cfg.min_discrete_obstacle_height
        )
    else:
        obstacle_height = 0.05 + 0.2 * (level / cfg.num_rows)

    return (
        ratio,
        smooth_slope_angle,
        rough_slope_angle,
        step_height,
        obstacle_height,
    )


def terrain_description(cfg, level, terrain_column):
    proportions = np.cumsum(cfg.terrain_proportions)
    choice = terrain_column / cfg.num_cols + 0.001
    (
        ratio,
        smooth_slope_angle,
        rough_slope_angle,
        step_height,
        obstacle_height,
    ) = curriculum_values(cfg, level)

    if choice < proportions[0]:
        if hasattr(cfg, "smooth_slope_direction"):
            descending = cfg.smooth_slope_direction < 0.0
        else:
            descending = choice < proportions[0] / 2.0
        name = "平滑下坡" if descending else "平滑上坡"
        platform_size = getattr(cfg, "smooth_slope_platform_size", 3.0)
        detail = (
            f"坡度={smooth_slope_angle:.1f}°，"
            f"顶部平台={platform_size:.1f}×{platform_size:.1f} m"
        )
    elif choice < proportions[1]:
        name = "粗糙斜坡"
        detail = f"坡度={rough_slope_angle:.1f}°"
    elif choice < proportions[3]:
        descending = choice < proportions[2]
        name = "下楼梯" if descending else "上楼梯"
        detail = f"单级高度={step_height * 100.0:.1f} cm"
    elif choice < proportions[4]:
        name = "离散障碍物"
        detail = f"最大高度={obstacle_height * 100.0:.1f} cm"
    elif choice < proportions[5]:
        name = "踏石"
        detail = "石块尺寸随难度变化"
    elif choice < proportions[6]:
        name = "T 型台阶"
        detail = "自定义地形"
    elif choice < proportions[7]:
        name = "长斜坡平台"
        detail = "自定义地形"
    elif choice < proportions[8]:
        name = "碎石/木屑地面"
        detail = "自定义粗糙地形"
    elif choice < proportions[9]:
        name = "密集碎块"
        detail = "自定义粗糙地形"
    elif choice < proportions[10]:
        name = "高墙"
        detail = "高度随难度变化"
    else:
        name = "减速带"
        detail = "高度随难度变化"

    return name, detail, ratio


class TerrainViewer:
    def __init__(self, gym, sim, viewer, terrain):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer
        self.terrain = terrain
        self.level = 0
        self.terrain_column = 0
        self.overview = True

    def show_overview(self):
        cfg = self.terrain.cfg
        total_x = cfg.num_rows * cfg.terrain_length
        total_y = cfg.num_cols * cfg.terrain_width
        center_x = total_x / 2.0
        center_y = total_y / 2.0
        span = max(total_x, total_y)
        camera_position = gymapi.Vec3(
            center_x - 0.45 * span,
            center_y - 0.45 * span,
            0.72 * span,
        )
        camera_target = gymapi.Vec3(center_x, center_y, 0.0)
        self.gym.viewer_camera_look_at(
            self.viewer, None, camera_position, camera_target
        )
        self.overview = True
        print(
            f"[全局视图] {cfg.num_rows} 个难度等级 × "
            f"{cfg.num_cols} 个地形列，共 "
            f"{cfg.num_rows * cfg.num_cols} 块课程地形"
        )

    def show_selected(self):
        cfg = self.terrain.cfg
        origin = self.terrain.env_origins[
            self.level, self.terrain_column
        ]
        target_z = max(float(origin[2]), 0.0)
        camera_position = gymapi.Vec3(
            float(origin[0] - 0.75 * cfg.terrain_length),
            float(origin[1] - 0.75 * cfg.terrain_width),
            target_z + max(cfg.terrain_length, cfg.terrain_width) * 0.8,
        )
        camera_target = gymapi.Vec3(
            float(origin[0]), float(origin[1]), target_z
        )
        self.gym.viewer_camera_look_at(
            self.viewer, None, camera_position, camera_target
        )
        self.overview = False

        name, detail, ratio = terrain_description(
            cfg, self.level, self.terrain_column
        )
        print(
            f"[地形] 类型列 {self.terrain_column + 1}/{cfg.num_cols} | "
            f"难度 {self.level + 1}/{cfg.num_rows} "
            f"({ratio * 100.0:.1f}%) | {name} | {detail}"
        )

    def move(self, level_delta=0, terrain_delta=0):
        cfg = self.terrain.cfg
        self.level = (self.level + level_delta) % cfg.num_rows
        self.terrain_column = (
            self.terrain_column + terrain_delta
        ) % cfg.num_cols
        self.show_selected()

    def handle_event(self, action):
        if action == "previous_type":
            self.move(terrain_delta=-1)
        elif action == "next_type":
            self.move(terrain_delta=1)
        elif action == "easier":
            self.move(level_delta=-1)
        elif action == "harder":
            self.move(level_delta=1)
        elif action == "overview":
            self.show_overview()


def run(args):
    if args.headless:
        raise SystemExit("look_terrain.py requires a viewer; remove --headless")

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    terrain_cfg = env_cfg.terrain
    terrain_cfg.mesh_type = "trimesh"
    terrain_cfg.curriculum = True
    terrain_cfg.selected = False
    terrain_cfg.border_size = 0.0
    if args.seed is not None:
        env_cfg.seed = args.seed
    np.random.seed(env_cfg.seed)

    sim_params = parse_sim_params(
        args, {"sim": class_to_dict(env_cfg.sim)}
    )
    _, sim_device_id = gymutil.parse_device_str(args.sim_device)
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(
        sim_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise SystemExit("Failed to create Isaac Gym simulation")

    viewer = None
    try:
        terrain = Terrain(terrain_cfg, num_robots=0)
        add_terrain_mesh(gym, sim, terrain)
        gym.prepare_sim(sim)

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise SystemExit("Failed to create Isaac Gym viewer")

        gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_ESCAPE, "quit"
        )
        gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_LEFT, "previous_type"
        )
        gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_RIGHT, "next_type"
        )
        gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_DOWN, "easier"
        )
        gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_UP, "harder"
        )
        gym.subscribe_viewer_keyboard_event(
            viewer, gymapi.KEY_M, "overview"
        )

        terrain_viewer = TerrainViewer(gym, sim, viewer, terrain)
        print(f"\n[课程地形查看器] task={args.task}")
        print("←/→: 切换地形类型，↓/↑: 降低/提高难度")
        print("M: 返回完整课程地图，ESC: 退出")
        print("只生成地形，不创建机器人、策略或训练日志。\n")
        terrain_viewer.show_overview()

        quit_requested = False
        while not quit_requested and not gym.query_viewer_has_closed(viewer):
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0:
                    continue
                if event.action == "quit":
                    quit_requested = True
                    break
                terrain_viewer.handle_event(event.action)

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    ensure_default_task()
    run(get_args())
