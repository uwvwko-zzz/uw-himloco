#!/usr/bin/env python3
"""Run a YQZ high-platform ONNX policy on 10--80 cm MuJoCo ledges."""

import argparse
from pathlib import Path
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PLUGIN_SRC = SCRIPT_DIR.parent / "mujoco-gui" / "src"
YQZ000_DIR = SCRIPT_DIR.parent / "yqz_000"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(YQZ000_DIR))
sys.path.insert(0, str(PLUGIN_SRC))

import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402
from runtime_control import (  # noqa: E402
    MapSpec,
    ParameterSpec,
    RuntimeScene,
    make_runtime_config,
    make_standard_robot_cameras,
    setup_tracking_camera,
    standard_camera_options,
)
from yqz_high_adapter import YQZHighAdapter  # noqa: E402


ROBOT_XML = PROJECT_ROOT / "resources/robots/YQZ-000/xml/YQZ-000.xml"
PLATFORM_HEIGHTS = tuple(round(value / 10.0, 1) for value in range(8, 0, -1))
# Match legged_gym/scripts/yqz_high/play.py.  The interactive PT demo uses a
# compact course whose ledge is 0.8 m ahead of each tile centre; using the
# training-generation value 1.5 m changes the policy phase at first contact.
PLATFORM_EDGE_X = 0.8
PLATFORM_LENGTH = 1.2
PLATFORM_WIDTH = 2.0
COURSE_SPACING = 3.0
# Isaac Gym's deterministic seed-1 demo reset starts about 8 cm behind and
# 8 cm to the right of the tile centre (LeggedRobot reset XY randomization).
# Reproduce that approach distance/phase for an apples-to-apples rollout.
SPAWN_OFFSET_X = -0.08
SPAWN_OFFSET_Y = -0.08
NOMINAL_PAYLOAD_MASS = 20.0
COLORS = (
    "0.80 0.15 0.20 1", "0.90 0.28 0.18 1", "0.95 0.45 0.18 1",
    "0.95 0.65 0.18 1", "0.75 0.78 0.20 1", "0.40 0.80 0.35 1",
    "0.25 0.72 0.75 1", "0.25 0.55 0.90 1",
)
CONTACT = {
    # Isaac demo: friction randomization disabled -> nominal coefficient 1.0,
    # PhysX contact_offset=0.01 and restitution=0.
    "condim": "3", "friction": "1.0 0.005 0.0001",
    "solref": "0.03 2.0", "solimp": "0.8 0.95 0.01 0.5 2",
    "margin": "0.01",
}
ROBOT_CAMERAS = make_standard_robot_cameras(
    prefix="yqz_high", front_position=(0.34, 0.0, 0.18),
    rear_position=(-1.0, 0.0, 0.65), top_position=(0.0, 0.0, 1.2),
)
CAMERAS = standard_camera_options(prefix="yqz_high")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument(
        "--platform-height", type=float, default=0.8,
        choices=PLATFORM_HEIGHTS,
        help="从指定高度的路段开始，默认从 0.8 m 开始",
    )
    parser.add_argument("--vx", type=float, default=0.0,
                        help="初始前向速度；默认静止，使用方向键控制")
    parser.add_argument("--key-vx", type=float, default=1.0,
                        help="↑/↓ 前后速度幅值 (m/s)")
    parser.add_argument("--key-vy", type=float, default=0.5,
                        help="←/→ 左右横移速度幅值 (m/s)")
    parser.add_argument("--key-wz", type=float, default=0.8,
                        help="Q/E 转向角速度幅值 (rad/s)")
    parser.add_argument("--gui", action="store_true", help="使用浏览器 GUI")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--debug-interval", type=float, default=0.0,
        help="每隔多少秒打印坐标/速度/轮速；0 表示关闭",
    )
    parser.add_argument("--gui-port", type=int, default=8766)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.gui and args.headless:
        parser.error("--gui and --headless cannot be combined")
    if args.headless and args.duration <= 0:
        parser.error("--headless requires --duration > 0")
    args.onnx = args.onnx.expanduser().resolve()
    if not args.onnx.is_file():
        parser.error(f"ONNX not found: {args.onnx}")
    return args


def course_edge_x(index):
    return index * COURSE_SPACING + PLATFORM_EDGE_X


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


def add_height_label(worldbody, edge_x, height):
    """Add a non-colliding yellow ``80cm``-style label on the platform top."""
    label = f"{int(round(height * 100)):02d}cm"
    glyph_width, glyph_gap, glyph_height = 0.16, 0.05, 0.42
    total_width = len(label) * glyph_width + (len(label) - 1) * glyph_gap
    cursor_y = -total_width / 2.0
    base_x = edge_x + 0.20
    z = height + 0.012
    for character in label:
        if character.isdigit():
            strokes = [_LABEL_SEGMENTS[name]
                       for name in _DIGIT_SEGMENTS[character]]
        elif character == "c":
            strokes = [_LABEL_SEGMENTS[name] for name in "afed"]
        else:
            strokes = (
                ((0, 0), (0, 1)), ((0, 1), (0.5, 0.55)),
                ((0.5, 0.55), (1, 1)), ((1, 1), (1, 0)),
            )
        for (u0, v0), (u1, v1) in strokes:
            ET.SubElement(worldbody, "geom", {
                "type": "capsule", "size": "0.012",
                "fromto": (
                    f"{base_x + v0 * glyph_height:.4f} "
                    f"{cursor_y + u0 * glyph_width:.4f} {z:.4f} "
                    f"{base_x + v1 * glyph_height:.4f} "
                    f"{cursor_y + u1 * glyph_width:.4f} {z:.4f}"
                ),
                "rgba": "1 0.9 0.05 1", "contype": "0", "conaffinity": "0",
            })
        cursor_y += glyph_width + glyph_gap


def build_platform_scene(output_path):
    """Add a compact 80-to-10 cm high-platform course along world +X."""
    root = ET.parse(ROBOT_XML).getroot()
    compiler = root.find("compiler")
    compiler.set(
        "meshdir", str(PROJECT_ROOT / "resources/robots/YQZ-000/meshes_mujoco")
    )
    option = root.find("option")
    option.set("iterations", "100")
    option.set("integrator", "implicitfast")
    worldbody = root.find("worldbody")
    for index, (height, color) in enumerate(zip(PLATFORM_HEIGHTS, COLORS)):
        edge_x = course_edge_x(index)
        center_x = edge_x + PLATFORM_LENGTH / 2.0
        geom = ET.SubElement(worldbody, "geom", {
            "name": f"high_platform_{int(height * 100):02d}cm",
            "type": "box",
            "pos": f"{center_x:.3f} 0 {height / 2:.3f}",
            "size": f"{PLATFORM_LENGTH / 2:.3f} {PLATFORM_WIDTH / 2:.3f} {height / 2:.3f}",
            "rgba": color,
        })
        for key, value in CONTACT.items():
            geom.set(key, value)
        add_height_label(worldbody, edge_x, height)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def configure_loaded_base(path):
    """Match the collision shapes and loaded inertia used by Isaac Gym."""
    root = ET.parse(path).getroot()

    # LeggedRobotCfg.asset.replace_cylinder_with_capsule is enabled during
    # training.  This matters most when a wheel hits the sharp platform lip:
    # cylinder-edge contacts in MuJoCo are qualitatively different from the
    # capsule contacts seen by the policy in PhysX.
    for body_name in ("FL_foot_Link", "FR_foot_Link",
                      "RL_foot_Link", "RR_foot_Link"):
        wheel = root.find(f".//body[@name='{body_name}']/geom[@type='cylinder']")
        if wheel is None:
            raise ValueError(f"Missing wheel collision geom on {body_name}")
        wheel.set("type", "capsule")

    inertial = root.find(".//body[@name='base_link']/inertial")
    if inertial is None:
        raise ValueError("Missing base_link inertial")
    original_mass = float(inertial.get("mass"))
    loaded_mass = original_mass + NOMINAL_PAYLOAD_MASS
    inertial.set("mass", f"{loaded_mass:.9g}")
    # Measured directly from Isaac Gym after its
    # set_actor_rigid_body_properties(..., recomputeInertia=True) call with
    # the deterministic 20 kg demo payload. MuJoCo fullinertia order is
    # Ixx Iyy Izz Ixy Ixz Iyz.
    inertial.attrib.pop("diaginertia", None)
    inertial.set(
        "fullinertia",
        "0.256713 1.564523 1.649631 0.010674 -0.003208 0.001070",
    )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def load_policy(path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    policy_input, policy_output = session.get_inputs()[0], session.get_outputs()[0]
    if policy_input.shape[-1] != 348 or policy_output.shape[-1] != 16:
        raise ValueError(
            f"Bad ONNX dimensions: {policy_input.shape} -> {policy_output.shape}"
        )
    return session, policy_input.name, policy_output.name


def make_config(args, spawn_x):
    config = make_runtime_config(
        gui=args.gui, title="YQZ High Platform MuJoCo Sim2Sim",
        maps={"high_platform": "80–10 cm 高台路线"},
        map_spawns={"high_platform": {
            "position": [spawn_x + SPAWN_OFFSET_X, SPAWN_OFFSET_Y, 0.50],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        }},
        kp=80, kd=2, torque_limit=120,
        initial_position=[spawn_x + SPAWN_OFFSET_X, SPAWN_OFFSET_Y, 0.50],
        initial_quaternion=[1.0, 0.0, 0.0, 0.0],
        command=(abs(args.key_vx), 0.0, abs(args.key_wz), 0.47),
        height_range=(0.47, 0.47), cameras=CAMERAS, port=args.gui_port,
        render={"width": 960, "height": 720, "fps": 60, "jpeg_quality": 92},
        tracking_camera={"camera_distance": 3.0, "camera_azimuth": 135,
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
        random_seed=args.seed, snapshot_path=None, stop_on_panel_close=True,
    )
    config["runtime_ui"]["default_map"] = "high_platform"
    # Training randomizes an added base payload in [18, 22] kg.  Starting the
    # MuJoCo rollout at zero payload is a large dynamics mismatch.
    # configure_loaded_base embeds both the nominal payload and its matching
    # rotational inertia. Runtime payload is additional payload only.
    config["runtime_parameters"]["payload_mass"] = 0.0
    return config


class HighPlatformKeys:
    """Hold-to-move controls using Linux key press/release events."""

    GLFW_KEY_RIGHT = 262
    GLFW_KEY_LEFT = 263
    GLFW_KEY_DOWN = 264
    GLFW_KEY_UP = 265

    def __init__(self, command, height, magnitudes, enable_evdev=True):
        self.command = command
        self.height = height
        self.magnitudes = magnitudes
        self.pressed = set()
        self.lock = threading.Lock()
        self.devices = []
        self.evdev = None
        self.last_reported_command = None
        self.reset_requested = False
        self.runtime = None
        if enable_evdev:
            self._start_evdev()

    def _start_evdev(self):
        """Use evdev because MuJoCo's callback does not report key releases."""
        try:
            import evdev
        except ImportError:
            print("[KEYBOARD] 未安装 evdev，无法使用按住/松开控制")
            print("           可执行: pip install evdev")
            return

        self.evdev = evdev
        self.scancodes = {
            evdev.ecodes.KEY_W: "forward",
            evdev.ecodes.KEY_S: "backward",
            evdev.ecodes.KEY_A: "left",
            evdev.ecodes.KEY_D: "right",
            evdev.ecodes.KEY_Q: "turn_left",
            evdev.ecodes.KEY_E: "turn_right",
            evdev.ecodes.KEY_UP: "forward",
            evdev.ecodes.KEY_DOWN: "backward",
            evdev.ecodes.KEY_LEFT: "left",
            evdev.ecodes.KEY_RIGHT: "right",
        }
        permission_denied = False
        for path in evdev.list_devices():
            try:
                device = evdev.InputDevice(path)
                supported = device.capabilities().get(evdev.ecodes.EV_KEY, ())
                if evdev.ecodes.KEY_W in supported or evdev.ecodes.KEY_UP in supported:
                    self.devices.append(device)
            except PermissionError:
                permission_denied = True
            except OSError:
                pass

        if not self.devices:
            print("[KEYBOARD] 未找到可读键盘，方向键控制不可用")
            if permission_denied:
                print("           请将当前用户加入 input 组后重新登录:")
                print("           sudo usermod -aG input $USER")
            return

        for device in self.devices:
            threading.Thread(
                target=self._read_device, args=(device,), daemon=True
            ).start()
        print(f"[KEYBOARD] evdev 正在监听 {len(self.devices)} 个键盘设备")
        for device in self.devices:
            print(f"           {device.path}: {device.name}")

    def _read_device(self, device):
        try:
            for event in device.read_loop():
                if event.type != self.evdev.ecodes.EV_KEY:
                    continue
                key = self.scancodes.get(event.code)
                if key is None:
                    continue
                with self.lock:
                    if event.value == 1:
                        self.pressed.add(key)
                    elif event.value == 0:
                        self.pressed.discard(key)
        except (OSError, PermissionError) as exc:
            print(f"[KEYBOARD] {device.path} 读取中止: {exc}")

    def update_command(self):
        """Recompute commands every frame; released keys therefore mean zero."""
        with self.lock:
            pressed = self.pressed.copy()
        forward = int("forward" in pressed) - int("backward" in pressed)
        lateral = int("left" in pressed) - int("right" in pressed)
        turn = int("turn_left" in pressed) - int("turn_right" in pressed)
        self.command[:] = (
            self.magnitudes[0] * forward,
            self.magnitudes[1] * lateral,
            self.magnitudes[2] * turn,
        )
        current = tuple(float(value) for value in self.command)
        if current != self.last_reported_command:
            print(
                f"[COMMAND] vx={current[0]:+.2f} "
                f"vy={current[1]:+.2f} wz={current[2]:+.2f}"
            )
            self.last_reported_command = current

    def key_callback(self, code):
        if code in (ord("x"), ord("X")):
            with self.lock:
                self.pressed.clear()
            self.command[:] = 0.0
            print("[COMMAND] exact zero command")
        elif code in (ord("t"), ord("T")):
            self.reset_requested = True
            print("[COMMAND] reset requested")
        elif code == 32 and self.runtime is not None:
            self.runtime.request_push()

    def consume_reset(self):
        result = self.reset_requested
        self.reset_requested = False
        return result


def run():
    args = parse_args()
    np.random.seed(args.seed)
    start_index = PLATFORM_HEIGHTS.index(round(args.platform_height, 1))
    spawn_x = start_index * COURSE_SPACING
    with tempfile.TemporaryDirectory(prefix="yqz_high_mujoco_") as directory:
        scene_xml = Path(directory) / "yqz_high_platforms.xml"
        build_platform_scene(scene_xml)
        config = make_config(args, spawn_x)
        policy, input_name, output_name = load_policy(args.onnx)
        adapter = YQZHighAdapter()
        scene = RuntimeScene.for_adapter(
            adapter, robot_xml=ROBOT_XML,
            map_specs={"high_platform": MapSpec(
                scene_xml, exclude_bodies=("base_link",),
                contact_params=CONTACT,
            )},
            runtime_config=config, robot_cameras=ROBOT_CAMERAS,
            dimension_context="YQZ high-platform scene",
            xml_transform=configure_loaded_base,
            output_name="yqz_high_platform.xml",
        )
        try:
            scene.open()
            model, data, runtime = scene.model, scene.data, scene.runtime
            command = np.array([args.vx, 0.0, 0.0], dtype=np.float32)
            keys = HighPlatformKeys(
                command, 0.47,
                (abs(args.key_vx), abs(args.key_vy), abs(args.key_wz)),
                enable_evdev=not args.gui and not args.headless,
            )
            keys.runtime = runtime
            runtime.runtime_control(model, data)
            runtime.consume_reset()
            adapter.reset(model, data, config)
            runtime.reset_simulation_state()
            action = np.zeros(16)
            control_steps = adapter.control_steps(model)
            step, start = 0, time.monotonic()
            next_debug_time = args.debug_interval
            print(
                f"[ONNX] {args.onnx}\n[COURSE] 0.8 -> 0.1 m, "
                f"start={args.platform_height:.1f} m "
                f"(edge x={course_edge_x(start_index):.1f} m)"
            )
            print("[KEYS] ↑/↓ or W/S forward/back | ←/→ or A/D strafe")
            print("[KEYS] Q/E yaw")
            print("[KEYS] X zero command | T reset | SPACE push (click the MuJoCo window first)")
            if args.gui:
                print(f"[GUI] http://127.0.0.1:{args.gui_port}")
            with scene.viewer(
                args.gui or args.headless, key_callback=keys.key_callback
            ) as viewer:
                if not args.gui and not args.headless:
                    setup_tracking_camera(viewer, model, "base_link", 3, 135, -20)
                pacing_wall_time = time.monotonic()
                pacing_sim_time = float(data.time)
                while viewer.is_running():
                    tick = time.monotonic()
                    if args.duration > 0 and tick - start >= args.duration:
                        break
                    if args.gui:
                        if runtime.update_command(()):
                            command[:] = [config["command"][key] for key in (
                                "linear_x", "linear_y", "yaw"
                            )]
                    elif not args.headless:
                        keys.update_command()
                    state = runtime.runtime_control(model, data)
                    if runtime.consume_reset() or keys.consume_reset():
                        adapter.reset(model, data, config)
                        action.fill(0.0)
                        runtime.reset_simulation_state()
                        pacing_wall_time = time.monotonic()
                        pacing_sim_time = float(data.time)
                    if step % control_steps == 0:
                        obs = adapter.build_observation(
                            model, data, command, height=0.47
                        )
                        action[:] = np.clip(policy.run(
                            [output_name], {input_name: obs}
                        )[0][0], -100, 100)
                    adapter.apply_control(model, data, action, state)
                    runtime.apply_external_forces(model, data)
                    mujoco.mj_step(model, data)
                    step += 1
                    sim_time = float(data.time)
                    if args.debug_interval > 0 and sim_time >= next_debug_time:
                        wheel_ids = np.array([3, 7, 11, 15])
                        wheel_vel = data.qvel[adapter.qvel_adr][wheel_ids]
                        qw, qx, qy, qz = data.qpos[3:7]
                        roll = np.arctan2(
                            2.0 * (qw * qx + qy * qz),
                            1.0 - 2.0 * (qx * qx + qy * qy),
                        )
                        pitch = np.arcsin(np.clip(
                            2.0 * (qw * qy - qz * qx), -1.0, 1.0
                        ))
                        yaw = np.arctan2(
                            2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz),
                        )
                        print(
                            f"[DEBUG t={sim_time:5.2f}] "
                            f"xyz=({data.qpos[0]:+.3f},{data.qpos[1]:+.3f},"
                            f"{data.qpos[2]:+.3f}) "
                            f"vxy=({data.qvel[0]:+.3f},{data.qvel[1]:+.3f}) "
                            f"rpy=({roll:+.2f},{pitch:+.2f},{yaw:+.2f}) "
                            f"action_w="
                            f"{np.array2string(action[wheel_ids], precision=2)} "
                            f"qvel_w={np.array2string(wheel_vel, precision=2)}"
                        )
                        next_debug_time += args.debug_interval
                    # The policy runs at 50 Hz, so rendering at the 200 Hz
                    # physics rate only wastes wall time and slows simulation.
                    if (
                        not args.gui and not args.headless
                        and step % control_steps == 0
                    ):
                        viewer.sync()
                    target_wall_time = pacing_wall_time + (
                        float(data.time) - pacing_sim_time
                    )
                    delay = target_wall_time - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            wall_elapsed = max(time.monotonic() - start, 1.0e-9)
            realtime_factor = float(data.time) / wall_elapsed
            print(
                f"[RESULT] base=({data.qpos[0]:.3f}, {data.qpos[1]:.3f}, "
                f"{data.qpos[2]:.3f}) m, command=({command[0]:.2f}, "
                f"{command[1]:.2f}, {command[2]:.2f}), "
                f"realtime={realtime_factor:.2f}x"
            )
        except KeyboardInterrupt:
            pass
        finally:
            scene.close()


if __name__ == "__main__":
    run()
