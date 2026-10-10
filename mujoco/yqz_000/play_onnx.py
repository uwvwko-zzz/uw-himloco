#!/usr/bin/env python3
"""Play YQZ-000 ONNX policies with the runtime-control MuJoCo plugin."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PLUGIN_SRC = SCRIPT_DIR.parent / "mujoco-gui" / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PLUGIN_SRC))

import mujoco
import onnxruntime as ort
from runtime_control import (
    MapSpec,
    ParameterSpec,
    RuntimeScene,
    bundled_map_specs,
    make_runtime_config,
    make_standard_robot_cameras,
    setup_tracking_camera,
    standard_camera_options,
)
from yqz_adapter import YQZ000Adapter

ROBOT_XML = PROJECT_ROOT / "resources/robots/YQZ-000/xml/YQZ-000.xml"
MAP_SPECS = {
    "flat": MapSpec(ROBOT_XML, exclude_bodies=("base_link",)),
    **bundled_map_specs(),
}
MAP_LABELS = {
    "flat": "纯平地", "rc26_track": "26RC 综合赛道",
    "race_track": "竞速赛道", "stairs": "长楼梯",
    "cross_stairs": "四向课程楼梯", "cross_slope": "四向课程斜坡",
    "google_barkour": "Google Barkour", "gap_jump": "沟槽跳跃",
    "hurdles": "跨栏", "suspended_steps": "悬空踏板",
    "perlin_rough": "Perlin 粗糙地形",
    "dynamic_obstacles": "动态随机障碍",
}
ROBOT_CAMERAS = make_standard_robot_cameras(
    prefix="yqz", front_position=(0.34, 0.0, 0.18),
    rear_position=(-1.0, 0.0, 0.65), top_position=(0.0, 0.0, 1.2),
)
CAMERAS = standard_camera_options(prefix="yqz")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--xml", default=ROBOT_XML, type=Path)
    parser.add_argument("--map", choices=tuple(MAP_SPECS), default="flat")
    parser.add_argument("--gui", action="store_true", help="仅使用浏览器 GUI")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gui-port", type=int, default=8765)
    parser.add_argument("--preset", type=Path, default=SCRIPT_DIR / "runtime_preset.json")
    parser.add_argument("--vx", type=float, default=1.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--key-vx", type=float, default=1.0,
                        help="W/S 按键的前后速度幅值 (m/s)")
    parser.add_argument("--key-vy", type=float, default=0.6,
                        help="A/D 按键的横向速度幅值 (m/s)")
    parser.add_argument("--key-wz", type=float, default=1.0,
                        help="Q/E 按键的转向角速度幅值 (rad/s)")
    parser.add_argument("--height", type=float, default=0.47)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.gui and args.headless:
        parser.error("--gui and --headless cannot be combined")
    if args.headless and args.duration <= 0:
        parser.error("--headless requires --duration > 0")
    for name in ("onnx", "xml", "preset"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.onnx.is_file(): parser.error(f"ONNX not found: {args.onnx}")
    if not args.xml.is_file(): parser.error(f"XML not found: {args.xml}")
    return args


def load_policy(path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    policy_input, policy_output = session.get_inputs()[0], session.get_outputs()[0]
    if policy_input.shape[-1] != 348 or policy_output.shape[-1] != 16:
        raise ValueError(f"Bad ONNX dimensions: {policy_input.shape} -> {policy_output.shape}")
    return session, policy_input.name, policy_output.name


def map_spawns():
    spawns = {name: {"position": [0, 0, 0.54], "quaternion": [1, 0, 0, 0]}
              for name in MAP_SPECS}
    spawns["flat"]["position"] = [0, 0, 0.50]
    spawns["rc26_track"] = {
        "position": [3.7, -9.0, 0.54],
        "quaternion": [-0.7071068, 0, 0, 0.7071068],
    }
    spawns["google_barkour"]["position"] = [1.0, -1.5, 0.54]
    return spawns


def runtime_config(args):
    spawns = map_spawns()
    config = make_runtime_config(
        gui=args.gui, title="YQZ-000 MuJoCo Sim2Sim 实时调参",
        maps=MAP_LABELS, map_spawns=spawns,
        kp=80, kd=2, torque_limit=120,
        initial_position=spawns[args.map]["position"],
        initial_quaternion=spawns[args.map]["quaternion"],
        # These values are keyboard magnitudes in the browser panel.  They
        # must not come from the initial --vy/--wz command, whose defaults are
        # intentionally zero.
        command=(abs(args.key_vx), abs(args.key_vy),
                 abs(args.key_wz), args.height),
        height_range=(0.40, 0.62), cameras=CAMERAS, port=args.gui_port,
        render={"width": 960, "height": 720, "fps": 60, "jpeg_quality": 92},
        tracking_camera={"camera_distance": 2, "camera_azimuth": 135,
                         "camera_elevation": -20},
        randomization={
            "kp": [60, 100], "kd": [1.5, 3], "torque_limit": [80, 130],
            "motor_strength": [0.85, 1.15], "motor_delay_ms": [0, 20],
            "mass_scale": [0.85, 1.15], "payload_mass": [0, 3],
            "friction_scale": [0.5, 1.5], "gravity_z": [-10.3, -9.3],
            "wheel_velocity_kp": [0.2, 1.0],
        },
        push={"force_range": [40, 120], "duration_range": [0.1, 0.25]},
        parameters=(ParameterSpec("wheel_velocity_kp", "轮子速度 Kp",
                                  0, 2, 0.05, 0.5),),
        random_seed=args.seed, snapshot_path=args.preset,
        stop_on_panel_close=True,
    )
    config["runtime_ui"]["default_map"] = args.map
    return config


class NativeKeys:
    def __init__(self, command, height, magnitudes):
        self.command, self.height, self.magnitudes = command, height, magnitudes
        self.reset_requested = False
        self.runtime = None

    def key_callback(self, code):
        key = chr(code).lower() if 0 <= code < 256 else ""
        if key == "w": self.command[0] = self.magnitudes[0]
        elif key == "s": self.command[0] = -self.magnitudes[0]
        elif key == "a": self.command[1] = self.magnitudes[1]
        elif key == "d": self.command[1] = -self.magnitudes[1]
        elif key == "q": self.command[2] = self.magnitudes[2]
        elif key == "e": self.command[2] = -self.magnitudes[2]
        elif key == "x": self.command[:] = 0
        elif key == "r": self.height = max(0.40, self.height - 0.02)
        elif key == "f": self.height = min(0.62, self.height + 0.02)
        elif key == "z": self.height = 0.47
        elif key == "t": self.reset_requested = True
        elif key == " " and self.runtime is not None: self.runtime.request_push()

    def consume_reset(self):
        result, self.reset_requested = self.reset_requested, False
        return result


def run():
    args = parse_args()
    np.random.seed(args.seed)
    config = runtime_config(args)
    policy, input_name, output_name = load_policy(args.onnx)
    adapter = YQZ000Adapter()
    map_specs = dict(MAP_SPECS)
    map_specs["flat"] = MapSpec(args.xml, exclude_bodies=("base_link",))
    scene = RuntimeScene.for_adapter(
        adapter, robot_xml=args.xml, map_specs=map_specs,
        runtime_config=config, robot_cameras=ROBOT_CAMERAS,
        dimension_context="YQZ-000 map composition",
        dynamic_obstacle_map="dynamic_obstacles",
        output_name="yqz_000_all_maps.xml",
    )
    try:
        scene.open()
        model, data, runtime = scene.model, scene.data, scene.runtime
        command = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
        keys = NativeKeys(command, args.height,
                          (abs(args.key_vx), abs(args.key_vy),
                           abs(args.key_wz)))
        keys.runtime = runtime
        runtime.runtime_control(model, data)
        runtime.consume_reset()
        adapter.reset(model, data, config)
        runtime.reset_simulation_state()
        action = np.zeros(16)
        control_steps, step, start = adapter.control_steps(model), 0, time.monotonic()
        print(f"[ONNX] {args.onnx}\n[MAP] {args.map} / {len(map_specs)} maps")
        print("[KEYS] W/S A/D Q/E | X stop | R/F/Z height | T reset | SPACE push")
        if args.gui: print(f"[GUI] http://127.0.0.1:{args.gui_port}")
        with scene.viewer(args.gui or args.headless,
                          key_callback=keys.key_callback) as viewer:
            if not args.gui and not args.headless:
                setup_tracking_camera(viewer, model, "base_link", 2, 135, -20)
            while viewer.is_running():
                tick = time.monotonic()
                if args.duration > 0 and tick - start >= args.duration: break
                if args.gui and runtime.update_command(()):
                    command[:] = [config["command"][k]
                                  for k in ("linear_x", "linear_y", "yaw")]
                    keys.height = config["command"]["height"]
                state = runtime.runtime_control(model, data)
                if runtime.consume_reset() or keys.consume_reset():
                    adapter.reset(model, data, config)
                    action.fill(0); runtime.reset_simulation_state()
                if step % control_steps == 0:
                    obs = adapter.build_observation(model, data, command,
                                                    height=keys.height)
                    action[:] = np.clip(policy.run(
                        [output_name], {input_name: obs})[0][0], -100, 100)
                adapter.apply_control(model, data, action, state)
                runtime.apply_external_forces(model, data)
                mujoco.mj_step(model, data); step += 1
                if not args.gui and not args.headless: viewer.sync()
                delay = model.opt.timestep - (time.monotonic() - tick)
                if delay > 0: time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        scene.close()


if __name__ == "__main__":
    run()
