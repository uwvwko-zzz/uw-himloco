#!/usr/bin/env python3
"""Run a W1W ONNX policy with the reusable MuJoCo browser GUI plugin."""

import argparse
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PLUGIN_SRC = SCRIPT_DIR.parent / "mujoco-gui" / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import mujoco
except (ImportError, AttributeError) as exc:
    raise SystemExit(
        "MuJoCo Python package is unavailable. Install it with: pip install mujoco"
    ) from exc

from play_onnx import (
    KeyboardController,
    load_policy,
)


try:
    from runtime_control import (
        MAP_GEOM_PREFIX,
        ActionSpec,
        MapSpec,
        ParameterSpec,
        RuntimeScene,
        bundled_map_specs,
        make_runtime_config,
        make_standard_robot_cameras,
        setup_tracking_camera,
        standard_camera_options,
    )
except ModuleNotFoundError as exc:
    if exc.name != "runtime_control" or not PLUGIN_SRC.is_dir():
        raise
    sys.path.insert(0, str(PLUGIN_SRC))
    from runtime_control import (
        MAP_GEOM_PREFIX,
        ActionSpec,
        MapSpec,
        ParameterSpec,
        RuntimeScene,
        bundled_map_specs,
        make_runtime_config,
        make_standard_robot_cameras,
        setup_tracking_camera,
        standard_camera_options,
    )

from w1w_adapter import W1W16Adapter


ROBOT_XML = SCRIPT_DIR / "w1w.xml"
MAP_SPECS = {
    # Reuse the robot XML's own world floor/material for the exact native look.
    "flat": MapSpec(ROBOT_XML, exclude_bodies=("base",)),
    **bundled_map_specs(),
}
MAP_LABELS = {
    "flat": "纯平地",
    "rc26_track": "26RC 综合赛道",
    "race_track": "竞速赛道",
    "stairs": "长楼梯",
    "cross_stairs": "四向课程楼梯",
    "cross_slope": "四向课程斜坡",
    "google_barkour": "Google Barkour",
    "gap_jump": "沟槽跳跃",
    "hurdles": "跨栏",
    "suspended_steps": "悬空踏板",
    "perlin_rough": "Perlin 粗糙地形",
    "dynamic_obstacles": "动态随机障碍",
}

ROBOT_CAMERAS = make_standard_robot_cameras(
    prefix="w1w",
    front_position=(0.24, 0.0, 0.14),
    rear_position=(-0.85, 0.0, 0.55),
    top_position=(0.0, 0.0, 1.0),
)
CAMERA_OPTIONS = standard_camera_options(prefix="w1w")

WHEEL_VELOCITY_KP = 1.5
KP_REFERENCE = 80.0
KD_REFERENCE = 2.5
TORQUE_LIMIT_REFERENCE = 175.38
NOMINAL_BASE_HEIGHT = 0.48
FALL_CLEARANCE = 0.01
CONTACT_PARAMS = {
    "condim": "3",
    "friction": "0.6 0.005 0.0001",
    "solref": "0.03 2.0",
    "solimp": "0.8 0.95 0.01 0.5 2",
    "margin": "0.005",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--xml", type=Path, default=ROBOT_XML)
    parser.add_argument(
        "--map", default="flat",
        help="Initial map; use 'custom' together with --custom-map",
    )
    parser.add_argument(
        "--custom-map", type=Path,
        help="Terrain-only MJCF exported by the visual scene editor",
    )
    parser.add_argument("--gui", action="store_true", help="Open browser GUI only")
    parser.add_argument(
        "--no-open-browser", action="store_true",
        help="Start the browser GUI server without opening another window",
    )
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--gui-port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--preset", type=Path, default=SCRIPT_DIR / "runtime_preset.json",
        help="Runtime parameters saved/loaded by the browser panel",
    )
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--vy", type=float, default=1.0)
    parser.add_argument("--wz", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=0.45)
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Seconds to run; 0 means until closed or interrupted",
    )
    args = parser.parse_args()
    if args.gui and args.headless:
        parser.error("--gui and --headless cannot be used together")
    if args.duration < 0:
        parser.error("--duration must be >= 0")
    if args.headless and args.duration == 0:
        parser.error("--headless requires a positive --duration")
    args.onnx = args.onnx.expanduser().resolve()
    args.xml = args.xml.expanduser().resolve()
    args.preset = args.preset.expanduser().resolve()
    if args.custom_map is not None:
        args.custom_map = args.custom_map.expanduser().resolve()
    if not args.onnx.is_file():
        parser.error(f"ONNX policy not found: {args.onnx}")
    if not args.xml.is_file():
        parser.error(f"robot XML not found: {args.xml}")
    available_maps = set(MAP_SPECS)
    if args.custom_map is not None:
        if not args.custom_map.is_file():
            parser.error(f"custom map not found: {args.custom_map}")
        available_maps.add("custom")
    if args.map not in available_maps:
        parser.error("unknown map %r; choose from %s" % (
            args.map, ", ".join(sorted(available_maps))
        ))
    return args


def map_spawns(map_names=None):
    map_names = tuple(map_names or MAP_SPECS)
    spawns = {
        name: {
            "position": [0.0, 0.0, 0.52],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        }
        for name in map_names
    }
    spawns["flat"]["position"] = [0.0, 0.0, 0.48]
    spawns["rc26_track"] = {
        "position": [3.7, -9.0, 0.48],
        "quaternion": [-0.7071068, 0.0, 0.0, 0.7071068],
    }
    spawns["google_barkour"]["position"] = [1.0, -1.5, 0.52]
    if "custom" in spawns:
        spawns["custom"]["position"] = [0.0, 0.0, 0.48]
    return spawns


def build_runtime_config(args, map_labels=None, spawns=None):
    map_labels = dict(map_labels or MAP_LABELS)
    spawns = dict(spawns or map_spawns(map_labels))
    config = make_runtime_config(
        gui=args.gui,
        title="W1W MuJoCo 实时调参与 Sim2Sim 验证",
        maps=map_labels,
        map_spawns=spawns,
        kp=KP_REFERENCE,
        kd=KD_REFERENCE,
        torque_limit=TORQUE_LIMIT_REFERENCE,
        initial_position=spawns[args.map]["position"],
        initial_quaternion=spawns[args.map]["quaternion"],
        command=(abs(args.vx), abs(args.vy), abs(args.wz), args.height),
        height_range=(0.40, 0.62),
        cameras=CAMERA_OPTIONS,
        port=args.gui_port,
        render={"width": 960, "height": 720, "fps": 60, "jpeg_quality": 92},
        tracking_camera={
            "camera_distance": 1.7,
            "camera_azimuth": 135.0,
            "camera_elevation": -20.0,
        },
        randomization={
            "kp": [48.0, 96.0],
            "kd": [1.5, 3.5],
            "torque_limit": [120.0, 190.0],
            "motor_strength": [0.85, 1.15],
            "motor_delay_ms": [0.0, 20.0],
            "mass_scale": [0.85, 1.15],
            "payload_mass": [0.0, 4.0],
            "friction_scale": [0.5, 1.5],
            "gravity_z": [-10.3, -9.3],
            "wheel_velocity_kp": [1.0, 2.2],
        },
        push={"force_range": [80.0, 180.0], "duration_range": [0.10, 0.25]},
        parameters=(
            ParameterSpec(
                "wheel_velocity_kp", "轮子速度 Kp",
                0.0, 5.0, 0.05, WHEEL_VELOCITY_KP,
            ),
        ),
        actions=(
            ActionSpec("fall_prone", "趴倒测试", "m", "warn"),
            ActionSpec("fall_side", "侧翻测试", "n", "warn"),
            ActionSpec("fall_random", "随机倾倒", "b", "warn"),
        ),
        random_seed=args.seed,
        snapshot_path=args.preset,
        stop_on_panel_close=True,
        open_browser=not args.no_open_browser,
    )
    config["runtime_ui"]["default_map"] = args.map
    return config


def configure_contacts(xml_path):
    """Use the same over-damped wheel contact settings as W1W terrain play."""
    root = ET.parse(xml_path).getroot()
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        parent_is_wheel = name == "" and geom.get("type") == "cylinder"
        is_map_collision = (
            name.startswith(MAP_GEOM_PREFIX)
            and geom.get("contype", "1") != "0"
            and geom.get("conaffinity", "1") != "0"
        )
        if not parent_is_wheel and not is_map_collision:
            continue
        for attribute, value in CONTACT_PARAMS.items():
            geom.set(attribute, value)
    ET.ElementTree(root).write(
        xml_path, encoding="utf-8", xml_declaration=True
    )


def quat_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray([
        lw*rw - lx*rx - ly*ry - lz*rz,
        lw*rx + lx*rw + ly*rz - lz*ry,
        lw*ry - lx*rz + ly*rw + lz*rx,
        lw*rz + lx*ry - ly*rx + lz*rw,
    ])


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = angle / 2.0
    return np.concatenate(([np.cos(half)], axis * np.sin(half)))


def yaw_only_quat(quat):
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    return axis_angle_quat((0.0, 0.0, 1.0), yaw)


class RuntimeKeyboardController(KeyboardController):
    GLFW_KEY_SPACE = 32
    GLFW_KEY_M = 77
    GLFW_KEY_N = 78
    GLFW_KEY_B = 66
    GLFW_KEY_O = 79
    GLFW_KEY_I = 73
    GLFW_KEY_P = 80

    def __init__(self, command, height):
        super().__init__(command, height)
        self.runtime = None
        self.runtime_config = None
        self.fall_requested = None
        self.camera_requested = None

    def attach_runtime(self, runtime, config):
        self.runtime = runtime
        self.runtime_config = config

    def key_callback(self, keycode):
        if keycode == self.GLFW_KEY_SPACE and self.runtime is not None:
            self.runtime.request_push()
        elif keycode == self.GLFW_KEY_M:
            self.fall_requested = "prone"
        elif keycode == self.GLFW_KEY_N:
            self.fall_requested = "side"
        elif keycode == self.GLFW_KEY_B:
            self.fall_requested = "random"
        elif keycode == self.GLFW_KEY_O:
            self.camera_requested = "w1w_front_camera"
        elif keycode == self.GLFW_KEY_I:
            self.camera_requested = "w1w_rear_overhead_camera"
        elif keycode == self.GLFW_KEY_P:
            self.camera_requested = "tracking"
        else:
            super().key_callback(keycode)

        if self.runtime is None:
            return
        if keycode in (self.GLFW_KEY_R, self.GLFW_KEY_F, self.GLFW_KEY_Z):
            self.runtime_config["command"]["height"] = self.height
            self.runtime.sync_height_to_panel()
        elif keycode == self.GLFW_KEY_X:
            self.runtime.sync_stop_to_panel()

    def consume_fall(self):
        requested = self.fall_requested
        self.fall_requested = None
        return requested


def is_robot_geom(model, base_id, geom_id):
    body_id = model.geom_bodyid[geom_id]
    return body_id == base_id or model.body_rootid[body_id] == base_id


def support_height(model, data, base_id):
    heights = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first = is_robot_geom(model, base_id, contact.geom1)
        second = is_robot_geom(model, base_id, contact.geom2)
        if first == second or abs(contact.frame[2]) < 0.5:
            continue
        heights.append(float(contact.pos[2]))
    return max(heights) if heights else float(data.qpos[2] - NOMINAL_BASE_HEIGHT)


def lowest_robot_collision_point(model, data, base_id):
    lowest = np.inf
    for geom_id in range(model.ngeom):
        if not is_robot_geom(model, base_id, geom_id):
            continue
        if model.geom_contype[geom_id] == 0:
            continue
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        size = model.geom_size[geom_id]
        geom_type = model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.dot(np.abs(rotation[2]), size)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis_z = abs(rotation[2, 2])
            extent = size[0] * np.sqrt(max(0.0, 1.0-axis_z*axis_z)) + size[1]*axis_z
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[1] * abs(rotation[2, 2]) + size[0]
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = size[0]
        else:
            extent = model.geom_rbound[geom_id]
        lowest = min(lowest, float(data.geom_xpos[geom_id, 2] - extent))
    return lowest


def apply_fall_pose(model, data, base_id, mode):
    ground = support_height(model, data, base_id)
    heading = yaw_only_quat(data.qpos[3:7])
    if mode == "prone":
        tilt = axis_angle_quat((0.0, 1.0, 0.0), np.pi / 2.0)
        label = "趴倒"
    elif mode == "side":
        tilt = axis_angle_quat((1.0, 0.0, 0.0), np.pi / 2.0)
        label = "侧翻"
    else:
        angle = np.random.uniform(0.0, 2.0*np.pi)
        axis = (np.cos(angle), np.sin(angle), 0.0)
        tilt_angle = np.random.uniform(np.deg2rad(90.0), np.deg2rad(170.0))
        tilt = axis_angle_quat(axis, tilt_angle)
        label = f"随机倾倒 {np.degrees(tilt_angle):.1f}°"

    data.qpos[3:7] = quat_multiply(heading, tilt)
    data.qpos[2] = ground + NOMINAL_BASE_HEIGHT
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    lowest = lowest_robot_collision_point(model, data, base_id)
    if np.isfinite(lowest):
        data.qpos[2] += ground + FALL_CLEARANCE - lowest
    mujoco.mj_forward(model, data)
    print(f"[FALL TEST] {label} | surface z={ground:.3f} m")


def update_native_camera(viewer, model, keyboard):
    requested = keyboard.camera_requested
    if requested is None:
        return
    keyboard.camera_requested = None
    if requested == "tracking":
        setup_tracking_camera(viewer, model, "base", 1.7, 135.0, -20.0)
        return
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, requested
    )
    if camera_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id


def run():
    args = parse_args()
    np.random.seed(args.seed)
    map_specs = dict(MAP_SPECS)
    map_specs["flat"] = MapSpec(args.xml, exclude_bodies=("base",))
    map_labels = dict(MAP_LABELS)
    if args.custom_map is not None:
        map_specs["custom"] = MapSpec(args.custom_map)
        map_labels["custom"] = "可视化编辑器场景"
    spawns = map_spawns(map_specs)
    runtime_config = build_runtime_config(args, map_labels, spawns)
    policy, input_name, output_name = load_policy(str(args.onnx))
    adapter = W1W16Adapter(
        kp_reference=KP_REFERENCE,
        kd_reference=KD_REFERENCE,
        torque_limit_reference=TORQUE_LIMIT_REFERENCE,
    )
    scene = RuntimeScene.for_adapter(
        adapter,
        robot_xml=args.xml,
        map_specs=map_specs,
        runtime_config=runtime_config,
        robot_cameras=ROBOT_CAMERAS,
        dimension_context="W1W map composition",
        dynamic_obstacle_map="dynamic_obstacles",
        xml_transform=configure_contacts,
        output_name="w1w_all_maps.xml",
    )
    try:
        scene.open()
        model = scene.model
        data = scene.data
        runtime = scene.runtime
        base_id = adapter.base_body_id
        command = np.zeros(3, dtype=np.float32)
        keyboard = RuntimeKeyboardController(command, args.height)
        keyboard.attach_runtime(runtime, runtime_config)

        # Activate the requested initial map before the first reset.
        runtime.runtime_control(model, data)
        runtime.consume_reset()
        adapter.reset(model, data, runtime_config)
        runtime.reset_simulation_state()

        action = np.zeros(adapter.action_size, dtype=np.float64)
        control_steps = adapter.control_steps(model)
        step = 0
        start = time.monotonic()

        print(f"[ONNX] {args.onnx}")
        print(f"[MODEL] nq={model.nq}, nv={model.nv}, nu={model.nu}")
        print(f"[MAP] initial={args.map}, available={len(map_specs)}")
        print("[CONTROL] W/S A/D Q/E | R/F height | Z height reset | T reset | X stop")
        print("[TEST] SPACE push | M prone | N side | B random fall | O/I/P cameras")
        if args.gui:
            print(f"[GUI] http://127.0.0.1:{args.gui_port}")

        display = scene.viewer(
            args.gui or args.headless,
            key_callback=keyboard.key_callback,
        )
        with display as viewer:
            if not args.gui and not args.headless:
                setup_tracking_camera(
                    viewer, model, "base", 1.7, 135.0, -20.0
                )

            while viewer.is_running():
                loop_start = time.monotonic()
                if args.duration > 0 and loop_start - start >= args.duration:
                    break

                with keyboard.lock:
                    physical_pressed = keyboard.pressed.copy()
                if runtime.update_command(physical_pressed):
                    command[:] = [
                        runtime_config["command"]["linear_x"],
                        runtime_config["command"]["linear_y"],
                        runtime_config["command"]["yaw"],
                    ]
                    keyboard.height = runtime_config["command"]["height"]
                else:
                    keyboard.update_command()
                height = float(np.clip(keyboard.height, 0.40, 0.62))

                runtime_state = runtime.runtime_control(model, data)
                reset_requested = runtime.consume_reset() or keyboard.consume_reset()
                if reset_requested:
                    adapter.reset(model, data, runtime_config)
                    action.fill(0.0)
                    runtime.reset_simulation_state()
                    print("[RESET] robot and policy history reset")

                fall_mode = keyboard.consume_fall()
                browser_falls = {
                    "fall_prone": "prone",
                    "fall_side": "side",
                    "fall_random": "random",
                }
                for browser_action in runtime.consume_actions():
                    fall_mode = browser_falls.get(browser_action, fall_mode)
                if fall_mode is not None:
                    apply_fall_pose(model, data, base_id, fall_mode)
                    adapter.reset_policy_state()
                    action.fill(0.0)
                    runtime.reset_simulation_state()

                if step % control_steps == 0:
                    observation = adapter.build_observation(
                        model, data, command, height=height
                    )
                    action[:] = np.clip(
                        policy.run(
                            [output_name], {input_name: observation}
                        )[0][0],
                        -100.0,
                        100.0,
                    )
                adapter.apply_control(model, data, action, runtime_state)

                runtime.apply_external_forces(model, data)
                mujoco.mj_step(model, data)
                step += 1
                if not args.gui and not args.headless:
                    update_native_camera(viewer, model, keyboard)
                    viewer.sync()

                remaining = model.opt.timestep - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        scene.close()


if __name__ == "__main__":
    run()
