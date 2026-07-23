"""Compact keyboard validation scene for the dog_high platform policy."""

import os
import sys
from types import MethodType


def extract_model_path(argv):
    """Read --model_path before Isaac Gym parses and rejects custom arguments."""
    model_path = None
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--model_path="):
            model_path = arg.split("=", 1)[1]
        elif arg == "--model_path":
            if i + 1 >= len(argv):
                raise ValueError("--model_path requires a checkpoint path")
            model_path = argv[i + 1]
            i += 1
        else:
            cleaned.append(arg)
        i += 1
    argv[:] = cleaned
    return model_path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import isaacgym  # noqa: F401,E402 - must be imported before torch
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaacgym import terrain_utils  # noqa: E402
from pynput import keyboard  # noqa: E402

from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.terrain import Terrain  # noqa: E402


NUM_TEST_LEVELS = 4
PLATFORM_HEIGHT_MIN = 0.05
PLATFORM_HEIGHT_MAX = 0.30
PLATFORM_X_OFFSET = 2.0
PLATFORM_LENGTH = 1.0


def platform_terrain(terrain, horizontal_scale, vertical_scale, difficulty=0.0,
                     platform_x_offset=PLATFORM_X_OFFSET,
                     platform_length=PLATFORM_LENGTH):
    """Create one full-width, flat-topped platform in a sub-terrain."""
    terrain.height_field_raw[:] = 0
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    height = PLATFORM_HEIGHT_MIN + (
        PLATFORM_HEIGHT_MAX - PLATFORM_HEIGHT_MIN
    ) * difficulty

    # Isaac Gym SubTerrain stores height fields as [x_pixels, y_pixels].
    # Use the array shape directly because its width/length attribute names are
    # counter-intuitive for non-square terrain cells.
    x_pixels = terrain.height_field_raw.shape[0]
    center_x = x_pixels // 2 + int(round(platform_x_offset / horizontal_scale))
    length_px = max(1, int(round(platform_length / horizontal_scale)))
    start_x = max(0, center_x - length_px // 2)
    end_x = min(x_pixels, start_x + length_px)
    height_raw = int(round(height / vertical_scale))
    terrain.height_field_raw[start_x:end_x, :] = height_raw


def compact_platform_selected(self):
    """Build four compact cells containing only the platform task."""
    print(
        f"\n[Platform validation] {self.cfg.num_rows} levels, "
        f"cell={self.env_length:.1f}x{self.env_width:.1f}m"
    )
    for row in range(self.cfg.num_rows):
        difficulty = row / max(1, self.cfg.num_rows - 1)
        height = PLATFORM_HEIGHT_MIN + (
            PLATFORM_HEIGHT_MAX - PLATFORM_HEIGHT_MIN
        ) * difficulty
        print(f"  level {row}: platform height={height * 100:.1f}cm")

        sub_terrain = terrain_utils.SubTerrain(
            "platform_validation",
            # SubTerrain allocates [width, length], while Terrain.add_terrain_to_map
            # expects [length_per_env_pixels, width_per_env_pixels].
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        platform_terrain(
            sub_terrain,
            horizontal_scale=self.cfg.horizontal_scale,
            vertical_scale=self.cfg.vertical_scale,
            difficulty=difficulty,
        )
        self.add_terrain_to_map(sub_terrain, row, 0)


Terrain.selected_terrain = compact_platform_selected

vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
height_cmd = 0.25
current_level = 0
reset_requested = False
exit_requested = False


def on_press(key):
    global vx_cmd, vy_cmd, wz_cmd, height_cmd
    global current_level, reset_requested, exit_requested
    try:
        if key.char == "w":
            vx_cmd = 1.0
        elif key.char == "s":
            vx_cmd = -1.0
        elif key.char == "a":
            vy_cmd = 1.0
        elif key.char == "d":
            vy_cmd = -1.0
        elif key.char == "q":
            wz_cmd = -1.0
        elif key.char == "e":
            wz_cmd = 1.0
        elif key.char == "r":
            height_cmd = max(0.20, height_cmd - 0.02)
        elif key.char == "f":
            height_cmd = min(0.35, height_cmd + 0.02)
        elif key.char == "z":
            height_cmd = 0.25
        elif key.char == "[":
            current_level = max(0, current_level - 1)
            reset_requested = True
        elif key.char == "]":
            current_level = min(NUM_TEST_LEVELS - 1, current_level + 1)
            reset_requested = True
        elif key.char == "p":
            reset_requested = True
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_requested = True


def on_release(key):
    global vx_cmd, vy_cmd, wz_cmd
    try:
        if key.char in "ws":
            vx_cmd = 0.0
        elif key.char in "ad":
            vy_cmd = 0.0
        elif key.char in "qe":
            wz_cmd = 0.0
    except AttributeError:
        pass


def place_robot_at_level(env, level):
    """Move the single validation robot to a selected platform-height cell."""
    level = int(np.clip(level, 0, NUM_TEST_LEVELS - 1))
    env.terrain_levels[0] = level
    env.terrain_types[0] = 0
    env.env_origins[0] = env.terrain_origins[level, 0]
    env.reset_idx(torch.tensor([0], device=env.device))
    height = PLATFORM_HEIGHT_MIN + (
        PLATFORM_HEIGHT_MAX - PLATFORM_HEIGHT_MIN
    ) * level / max(1, NUM_TEST_LEVELS - 1)
    print(f"[Level] {level}/{NUM_TEST_LEVELS - 1}, platform={height * 100:.1f}cm")


def play(args):
    global reset_requested

    if args.headless:
        raise ValueError("Keyboard validation requires a viewer; remove --headless")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Small 4x1 validation map: only four platform heights, one robot.
    env_cfg.env.num_envs = 1
    env_cfg.terrain.selected = True
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = NUM_TEST_LEVELS
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_length = 6.0
    env_cfg.terrain.terrain_width = 4.0
    env_cfg.terrain.border_size = 2.0
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.measure_heights = True
    env_cfg.terrain.wall_x_offsets = [PLATFORM_X_OFFSET]
    env_cfg.terrain.platform_length = PLATFORM_LENGTH

    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.commands.platform_command_ratio = 1.0
    env_cfg.commands.num_commands = 5
    env_cfg.env.episode_length_s = 1e6
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.env.test = True

    env_cfg.env.num_one_step_observations = 46
    env_cfg.env.num_observations = 46 * 6
    env_cfg.env.num_one_step_privileged_obs = 46 + 3 + 3 + 187
    env_cfg.env.num_privileged_obs = env_cfg.env.num_one_step_privileged_obs

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera([2.5, 3.0, 1.8], [1.0, 0.0, 0.35])

    def validation_termination(self):
        fallen = self.root_states[:, 2] < self.env_origins[:, 2] + 0.12
        upside_down = self.projected_gravity[:, 2] > 0.0
        # Do not reset at a sub-terrain cell boundary. The platform far edge is
        # at x=2.5m, so the old x=2.8m guard reset a successful robot almost
        # immediately after it cleared the obstacle.
        self.reset_buf = fallen | upside_down | self.time_out_buf

    env.check_termination = MethodType(validation_termination, env)
    place_robot_at_level(env, current_level)

    if not args.model_path:
        raise ValueError("Provide --model_path=/absolute/path/to/model_xxx.pt")
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    print(f"[Model] {model_path}")
    runner.load(model_path)
    policy = runner.get_inference_policy(device=env.device)
    if not callable(policy):
        raise TypeError(f"Inference policy is not callable: {type(policy).__name__}")

    print("\nControls: W/S forward/back, A/D strafe, Q/E turn")
    print("          [/] platform height, P reset, F/R body height, ESC quit")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        obs = env.get_observations()
        step_count = 0
        while not exit_requested:
            env.commands[0, 0] = vx_cmd
            env.commands[0, 1] = vy_cmd
            env.commands[0, 2] = wz_cmd
            env.commands[0, 4] = height_cmd

            with torch.no_grad():
                actions = policy(obs)
            obs, _, rewards, _, _, _, _ = env.step(actions)

            if reset_requested:
                place_robot_at_level(env, current_level)
                obs = env.get_observations()
                reset_requested = False

            step_count += 1
            if step_count % 100 == 0:
                mounted = int(env.platform_mounted[0].sum().item())
                passed = int(env.wall_crossed[0].sum().item())
                print(
                    f"[Step {step_count}] reward={rewards[0].item():.3f}, "
                    f"mounted={mounted}, passed={passed}, "
                    f"cmd=({vx_cmd:.1f},{vy_cmd:.1f},{wz_cmd:.1f})"
                )
            env.render()
    finally:
        listener.stop()
        if getattr(env, "viewer", None):
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    direct_model_path = extract_model_path(sys.argv)
    parsed_args = get_args()
    parsed_args.model_path = direct_model_path
    play(parsed_args)
