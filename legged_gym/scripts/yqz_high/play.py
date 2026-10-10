"""Interactive checkpoint demo for the YQZ high-platform curriculum."""

import os
import sys
from types import MethodType

import numpy as np


# Always import this repository when the script is launched from any directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401, E402
from isaacgym import gymapi  # noqa: E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: F401, F403, E402
from legged_gym.utils import get_args, task_registry  # noqa: E402


TASK_NAME = "yqz_high"
FORWARD_SPEED = 0.8
LATERAL_SPEED = 0.5
YAW_SPEED = 0.8
DEMO_TERRAIN_LENGTH = 5.0
DEMO_TERRAIN_WIDTH = 4.0
DEMO_BORDER_SIZE = 1.0
DEMO_PLATFORM_START_X = 0.8
DEMO_PLATFORM_HALF_WIDTH = 1.0
DEMO_PLATFORM_LENGTH = 1.2
DEMO_NUM_LEVELS = 8
DEMO_MIN_PLATFORM_HEIGHT = 0.10
DEMO_MAX_PLATFORM_HEIGHT = 0.80


def pop_demo_args():
    """Remove demo-only arguments unknown to Isaac Gym's parser."""
    argv = sys.argv[1:]
    remaining = [sys.argv[0]]
    model_values = []
    demo = {"auto_vx": None, "duration": 0.0, "debug_interval": 0.5}
    index = 0

    while index < len(argv):
        arg = argv[index]
        if arg == "--model":
            if index + 1 >= len(argv):
                raise SystemExit("--model 后面必须提供 checkpoint 路径")
            index += 1
            model_values.append(argv[index])
        elif arg.startswith("--model="):
            model_values.append(arg.split("=", 1)[1])
        elif arg in ("--auto-vx", "--duration", "--debug-interval"):
            if index + 1 >= len(argv):
                raise SystemExit(f"{arg} 后面必须提供数值")
            index += 1
            key = arg[2:].replace("-", "_")
            demo[key] = float(argv[index])
        elif arg.startswith("--auto-vx="):
            demo["auto_vx"] = float(arg.split("=", 1)[1])
        elif arg.startswith("--duration="):
            demo["duration"] = float(arg.split("=", 1)[1])
        elif arg.startswith("--debug-interval="):
            demo["debug_interval"] = float(arg.split("=", 1)[1])
        else:
            remaining.append(arg)
        index += 1

    if len(model_values) != 1 or not model_values[0]:
        raise SystemExit(
            f"用法: python {os.path.basename(__file__)} "
            "--model /path/to/model.pt"
        )

    model_path = os.path.abspath(os.path.expanduser(model_values[0]))
    if not os.path.isfile(model_path):
        raise SystemExit(f"模型文件不存在: {model_path}")

    # This script is task-specific, so make --task optional for convenience.
    if not any(arg == "--task" or arg.startswith("--task=") for arg in remaining):
        remaining.append(f"--task={TASK_NAME}")
    sys.argv[:] = remaining
    return model_path, demo


class ViewerControls:
    """Command state driven by Isaac Gym's native viewer events."""

    def __init__(self):
        self.pressed = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "turn_left": False,
            "turn_right": False,
        }
        self.reset_requested = False
        self.exit_requested = False
        self.follow_camera = False
        self.render_count = 0

    def handle_event(self, action, value):
        movement_actions = {
            "move_forward": "forward",
            "move_backward": "backward",
            "move_left": "left",
            "move_right": "right",
            "turn_left": "turn_left",
            "turn_right": "turn_right",
        }
        if action in movement_actions:
            self.pressed[movement_actions[action]] = value > 0
        elif value > 0:
            if action == "reset_robot":
                self.reset_requested = True
            elif action == "stop_robot":
                for name in self.pressed:
                    self.pressed[name] = False
            elif action == "toggle_follow_camera":
                self.follow_camera = not self.follow_camera
                state = "开启" if self.follow_camera else "关闭"
                print(f"[相机] 跟随已{state}")
            elif action == "QUIT":
                self.exit_requested = True

    def consume(self):
        vx = FORWARD_SPEED * (
            int(self.pressed["forward"]) - int(self.pressed["backward"])
        )
        vy = LATERAL_SPEED * (
            int(self.pressed["left"]) - int(self.pressed["right"])
        )
        wz = YAW_SPEED * (
            int(self.pressed["turn_left"])
            - int(self.pressed["turn_right"])
        )
        reset_requested = self.reset_requested
        exit_requested = self.exit_requested
        self.reset_requested = False
        return vx, vy, wz, reset_requested, exit_requested


def install_viewer_controls(env, controls):
    """Subscribe viewer keys and replace render so events are not discarded."""
    bindings = (
        (gymapi.KEY_W, "move_forward"),
        (gymapi.KEY_S, "move_backward"),
        (gymapi.KEY_A, "move_left"),
        (gymapi.KEY_D, "move_right"),
        (gymapi.KEY_Q, "turn_left"),
        (gymapi.KEY_E, "turn_right"),
        (gymapi.KEY_SPACE, "stop_robot"),
        (gymapi.KEY_R, "stop_robot"),
        (gymapi.KEY_P, "reset_robot"),
        (gymapi.KEY_C, "toggle_follow_camera"),
    )
    for key, action in bindings:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)

    def render_with_controls(self, sync_frame_time=True):
        if self.viewer is None:
            return
        if self.gym.query_viewer_has_closed(self.viewer):
            controls.exit_requested = True
            return

        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "toggle_viewer_sync" and event.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync
            else:
                controls.handle_event(event.action, event.value)

        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)

        if self.enable_viewer_sync:
            controls.render_count += 1
            if controls.follow_camera and controls.render_count % 3 == 0:
                follow_camera(self)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

    env.render = MethodType(render_with_controls, env)


def configure_demo(env_cfg):
    """Keep the trained terrain/observation layout but remove random effects."""
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.num_rows = DEMO_NUM_LEVELS
    env_cfg.terrain.min_platform_height = DEMO_MIN_PLATFORM_HEIGHT
    env_cfg.terrain.max_platform_height = DEMO_MAX_PLATFORM_HEIGHT
    env_cfg.terrain.reverse_platform_order = True
    # Training uses large 8 x 6 m tiles and a 25 m outer border.  The viewer
    # only needs one ordered lane. Eight levels give exact 10 cm decrements
    # from 80 cm down to 10 cm while keeping the course compact.
    env_cfg.terrain.terrain_length = DEMO_TERRAIN_LENGTH
    env_cfg.terrain.terrain_width = DEMO_TERRAIN_WIDTH
    env_cfg.terrain.border_size = DEMO_BORDER_SIZE
    env_cfg.terrain.platform_start_x = DEMO_PLATFORM_START_X
    env_cfg.terrain.platform_half_width = DEMO_PLATFORM_HALF_WIDTH
    env_cfg.terrain.platform_length = DEMO_PLATFORM_LENGTH
    env_cfg.terrain.measure_heights = True
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1.0e6
    env_cfg.env.episode_length_s = 1.0e6
    env_cfg.noise.add_noise = False
    # Training ends a successful climb immediately to advance curriculum.
    # The ordered interactive course must remain continuous instead.
    env_cfg.rewards.terminate_on_platform_success = False

    domain_rand = env_cfg.domain_rand
    # Use the nominal 20 kg midpoint of the training distribution;
    # disabling this flag would silently remove the payload.
    domain_rand.randomize_payload_mass = True
    domain_rand.payload_mass_range = [20.0, 20.0]
    domain_rand.randomize_com_displacement = False
    domain_rand.randomize_link_mass = False
    domain_rand.randomize_friction = False
    domain_rand.randomize_restitution = False
    domain_rand.randomize_motor_strength = False
    domain_rand.randomize_kp = False
    domain_rand.randomize_kd = False
    domain_rand.randomize_initial_joint_pos = False
    domain_rand.disturbance = False
    domain_rand.push_robots = False
    domain_rand.delay = False


def platform_height(env, course_index):
    ratio = course_index / max(env.cfg.terrain.num_rows - 1, 1)
    if getattr(env.cfg.terrain, "reverse_platform_order", False):
        ratio = 1.0 - ratio
    return env.cfg.terrain.min_platform_height + ratio * (
        env.cfg.terrain.max_platform_height
        - env.cfg.terrain.min_platform_height
    )


_LABEL_SEGMENTS = {
    "a": ((0, 1), (1, 1)), "b": ((1, 1), (1, 0.5)),
    "c": ((1, 0.5), (1, 0)), "d": ((0, 0), (1, 0)),
    "e": ((0, 0.5), (0, 0)), "f": ((0, 1), (0, 0.5)),
    "g": ((0, 0.5), (1, 0.5)),
}
_DIGIT_SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg",
    "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
    "8": "abcdefg", "9": "abcdfg",
}


def draw_platform_height_labels(env):
    """Draw bright ``80cm`` ... ``10cm`` labels on the front faces."""
    line_pairs = []
    color_values = []
    glyph_width = 0.16
    glyph_gap = 0.05
    label = "00cm"
    total_width = len(label) * glyph_width + (len(label) - 1) * glyph_gap

    for index in range(env.cfg.terrain.num_rows):
        height = platform_height(env, index)
        label = f"{int(round(height * 100)):02d}cm"
        tile_x = (index + 0.5) * env.cfg.terrain.terrain_length
        tile_y = 0.5 * env.cfg.terrain.terrain_width
        face_x = tile_x + env.cfg.terrain.platform_start_x - 0.015
        cursor_y = tile_y - total_width / 2.0
        base_z = height + 0.06

        for character in label:
            if character.isdigit():
                strokes = [_LABEL_SEGMENTS[name]
                           for name in _DIGIT_SEGMENTS[character]]
            elif character == "c":
                strokes = [_LABEL_SEGMENTS[name] for name in "afed"]
            else:  # compact lower-case m
                strokes = (
                    ((0, 0), (0, 1)), ((0, 1), (0.5, 0.55)),
                    ((0.5, 0.55), (1, 1)), ((1, 1), (1, 0)),
                )
            for (u0, v0), (u1, v1) in strokes:
                line_pairs.append((
                    (face_x, cursor_y + u0 * glyph_width,
                     base_z + v0 * 0.24),
                    (face_x, cursor_y + u1 * glyph_width,
                     base_z + v1 * 0.24),
                ))
                color_values.append((1.0, 0.9, 0.05))
            cursor_y += glyph_width + glyph_gap

    vertices = np.empty((len(line_pairs), 2), dtype=gymapi.Vec3.dtype)
    colors = np.empty(len(line_pairs), dtype=gymapi.Vec3.dtype)
    for line_index, (points, color) in enumerate(zip(line_pairs, color_values)):
        vertices[line_index][0] = points[0]
        vertices[line_index][1] = points[1]
        colors[line_index] = color
    env.gym.add_lines(
        env.viewer, env.envs[0], len(line_pairs), vertices, colors
    )


def reset_to_course_start(env):
    """Reset the robot at the beginning of the ordered platform course."""
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    env.terrain_levels[0] = 0
    terrain_type = env.terrain_types[0]
    env.env_origins[0] = env.terrain_origins[0, terrain_type]
    env.reset_idx(env_ids)

    env.obs_buf[env_ids] = 0.0
    if env.privileged_obs_buf is not None:
        env.privileged_obs_buf[env_ids] = 0.0
    env.compute_observations()
    return env.get_observations()


def follow_camera(env):
    position = env.root_states[0, :3].detach().cpu().tolist()
    camera = [position[0] - 3.0, position[1] - 3.0, position[2] + 1.8]
    target = [position[0] + 0.8, position[1], position[2] + 0.25]
    env.set_camera(camera, target)


def print_status(env, vx, vy, wz, prefix="状态"):
    world_x = float(env.root_states[0, 0].item())
    terrain_length = env.cfg.terrain.terrain_length
    course_index = max(
        0,
        min(int(world_x // terrain_length), env.cfg.terrain.num_rows - 1),
    )
    tile_centre_x = (course_index + 0.5) * terrain_length
    local_x = world_x - tile_centre_x
    print(
        f"[{prefix}] 路段={course_index + 1}/{env.cfg.terrain.num_rows} "
        f"| 高台={platform_height(env, course_index):.3f} m "
        f"| local_x={local_x:.2f} m | cmd=({vx:.2f}, {vy:.2f}, {wz:.2f})"
    )


def play(args, model_path, demo):
    if args.task != TASK_NAME:
        raise SystemExit(f"该脚本只支持 --task={TASK_NAME}")
    if args.headless and demo["auto_vx"] is None:
        raise SystemExit("无窗口模式需要提供 --auto-vx，例如 --auto-vx 0.8")

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_demo(env_cfg)

    env, _ = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    if demo["auto_vx"] is not None:
        body_props = env.gym.get_actor_rigid_body_properties(
            env.envs[0], env.actor_handles[0]
        )
        base = body_props[0]
        print(
            "[ISAAC BASE] "
            f"mass={base.mass:.9g} "
            f"com=({base.com.x:.6g},{base.com.y:.6g},{base.com.z:.6g}) "
            f"inertia_rows=({base.inertia.x}, {base.inertia.y}, "
            f"{base.inertia.z})"
        )

    # ``terrain.curriculum=True`` is retained only to generate eight ordered
    # descending heights. Runtime promotion/demotion is disabled: all rows are
    # already physically arranged in order along +X.
    def disable_runtime_curriculum(self, env_ids):
        return None

    env._update_terrain_curriculum = MethodType(
        disable_runtime_curriculum, env
    )

    # Never let a random training command (especially the standing mode)
    # overwrite keyboard input after a manual or automatic reset.
    def reset_demo_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, :3] = 0.0
        self.commands[env_ids, 4] = self.cfg.rewards.base_height_target
        if not hasattr(self, "standing_env_mask"):
            self.standing_env_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        self.standing_env_mask[env_ids] = False

    env._resample_commands = MethodType(reset_demo_commands, env)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    print(f"[INFO] 加载模型: {model_path}")
    runner.load(model_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    obs = reset_to_course_start(env)
    if not args.headless:
        follow_camera(env)

    controls = ViewerControls()
    if not args.headless:
        install_viewer_controls(env, controls)
        draw_platform_height_labels(env)

    print("\n=== YQZ 高台课程演示 ===")
    print("请先点击 Isaac Gym Viewer 窗口，再使用键盘控制。")
    print("W/S: 前进/后退    A/D: 左右平移    Q/E: 左右转向")
    print("空格或 R: 停止    P: 回到整条路线起点")
    print("C: 开关相机跟随    ESC: 退出")
    print("相机默认不跟随，可直接用鼠标自由旋转、平移和缩放。")
    heights = ", ".join(
        f"{int(round(platform_height(env, index) * 100))} cm"
        for index in range(env.cfg.terrain.num_rows)
    )
    print(
        f"{env.cfg.terrain.num_rows} 个课程高台已沿 +X 方向依次排列："
        f"{heights}。"
    )
    print("提示：该策略训练时主要使用前进指令，倒退/侧移/转向效果取决于模型。\n")
    print_status(env, 0.0, 0.0, 0.0, prefix="开始")

    step_count = 0
    next_debug_time = demo["debug_interval"]
    try:
        while True:
            if demo["auto_vx"] is None:
                vx, vy, wz, reset_requested, exit_requested = controls.consume()
            else:
                vx, vy, wz = demo["auto_vx"], 0.0, 0.0
                reset_requested = exit_requested = False
            if exit_requested:
                break

            if reset_requested:
                obs = reset_to_course_start(env)
                follow_camera(env)
                print_status(env, vx, vy, wz, prefix="复位")

            env.commands[0, 0] = vx
            env.commands[0, 1] = vy
            env.commands[0, 2] = wz
            if env.cfg.commands.num_commands >= 5:
                env.commands[0, 4] = env.cfg.rewards.base_height_target

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _, _, _, _ = env.step(actions)

            step_count += 1
            sim_time = step_count * env.dt
            if demo["debug_interval"] > 0 and sim_time >= next_debug_time:
                print_status(env, vx, vy, wz)
                position = env.root_states[0, :3].detach().cpu().numpy()
                quaternion = env.root_states[0, 3:7].detach().cpu().numpy()
                print(
                    f"[DEBUG t={sim_time:5.2f}] "
                    f"xyz=({position[0]:+.3f},{position[1]:+.3f},"
                    f"{position[2]:+.3f}) quat_xyzw="
                    f"{np.array2string(quaternion, precision=3)} "
                    f"action_w={np.array2string(actions[0, [3, 7, 11, 15]].detach().cpu().numpy(), precision=2)}"
                )
                next_debug_time += demo["debug_interval"]
            if demo["duration"] > 0 and sim_time >= demo["duration"]:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
        print("[INFO] 演示已退出")


if __name__ == "__main__":
    checkpoint_path, demo_options = pop_demo_args()
    cli_args = get_args()
    play(cli_args, checkpoint_path, demo_options)
