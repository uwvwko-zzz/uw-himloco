#!/usr/bin/env python3
"""Run a trained W1W HIMLoco ONNX policy in MuJoCo."""

import argparse
import os
import sys
import threading
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import mujoco
    import mujoco.viewer
except (ImportError, AttributeError) as exc:
    raise SystemExit(
        "MuJoCo Python package is unavailable. Install it with: pip install mujoco"
    ) from exc

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit(
        "ONNX Runtime is unavailable. Install it with: pip install onnxruntime"
    ) from exc


JOINT_NAMES = [
    "fl_hip_joint", "fl_thigh_joint", "fl_calf_joint", "fl_foot_joint",
    "fr_hip_joint", "fr_thigh_joint", "fr_calf_joint", "ft_foot_joint",
    "rl_hip_joint", "rl_thigh_joint", "rl_calf_joint", "rl_foot_joint",
    "rr_hip_joint", "rr_thigh_joint", "rr_calf_joint", "rr_foot_joint",
]
WHEEL_INDICES = np.array([3, 7, 11, 15])
LEG_INDICES = np.array([i for i in range(16) if i not in WHEEL_INDICES])
HIP_INDICES = np.array([0, 4, 8, 12])
WHEEL_DIRECTIONS = np.array([-1.0, 1.0, -1.0, 1.0])
DEFAULT_POS = np.array([
    0.0, -0.8, 1.55, 0.0,
    0.0, 0.8, -1.55, 0.0,
    0.0, -0.8, 1.55, 0.0,
    0.0, 0.8, -1.55, 0.0,
])
KP = np.array([60.0, 60.0, 80.0, 0.0] * 4)
KD = np.array([2.0, 2.0, 2.5, 0.0] * 4)


class KeyboardController:
    """Match the evdev + MuJoCo callback controls used by jqg/play_onnx.py."""

    GLFW_KEY_R = 82
    GLFW_KEY_F = 70
    GLFW_KEY_Z = 90
    GLFW_KEY_T = 84
    GLFW_KEY_X = 88

    def __init__(self, command, height):
        self.command = command
        self.height = height
        self.pressed = set()
        self.lock = threading.Lock()
        self.reset_requested = False
        self.devices = []
        self.evdev = None
        self._start_evdev()

    def _start_evdev(self):
        try:
            import evdev
        except ImportError:
            print("[KEYBOARD] 未安装 evdev，方向键控制不可用")
            print("           可执行: pip install evdev")
            return

        self.evdev = evdev
        self.scancodes = {
            evdev.ecodes.KEY_W: "w", evdev.ecodes.KEY_S: "s",
            evdev.ecodes.KEY_A: "a", evdev.ecodes.KEY_D: "d",
            evdev.ecodes.KEY_Q: "q", evdev.ecodes.KEY_E: "e",
            evdev.ecodes.KEY_UP: "w", evdev.ecodes.KEY_DOWN: "s",
            evdev.ecodes.KEY_LEFT: "a", evdev.ecodes.KEY_RIGHT: "d",
        }
        permission_denied = False
        for path in evdev.list_devices():
            try:
                device = evdev.InputDevice(path)
                keys = device.capabilities().get(evdev.ecodes.EV_KEY, ())
                if evdev.ecodes.KEY_W in keys or evdev.ecodes.KEY_UP in keys:
                    self.devices.append(device)
            except PermissionError:
                permission_denied = True
            except OSError:
                pass

        if not self.devices:
            print("[KEYBOARD] 未找到可读取的键盘设备，方向键控制不可用")
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
        with self.lock:
            pressed = self.pressed.copy()
        self.command[0] = 1.5 if "w" in pressed else -1.5 if "s" in pressed else 0.0
        self.command[1] = 1.0 if "a" in pressed else -1.0 if "d" in pressed else 0.0
        self.command[2] = 1.0 if "q" in pressed else -1.0 if "e" in pressed else 0.0

    def key_callback(self, keycode):
        if keycode == self.GLFW_KEY_R:
            self.height = max(0.4, self.height - 0.02)
        elif keycode == self.GLFW_KEY_F:
            self.height = min(0.62, self.height + 0.02)
        elif keycode == self.GLFW_KEY_Z:
            self.height = 0.45
        elif keycode == self.GLFW_KEY_T:
            self.reset_requested = True
        elif keycode == self.GLFW_KEY_X:
            with self.lock:
                self.pressed.clear()
            self.command[:] = 0.0

    def consume_reset(self):
        requested = self.reset_requested
        self.reset_requested = False
        return requested


def load_policy(path):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("The ONNX policy must have exactly one input and one output")
    policy_input = session.get_inputs()[0]
    policy_output = session.get_outputs()[0]
    if policy_input.shape[-1] != 348 or policy_output.shape[-1] != 16:
        raise ValueError(
            "Expected W1W ONNX dimensions [batch, 348] -> [batch, 16], got "
            f"{policy_input.shape} -> {policy_output.shape}"
        )
    return session, policy_input.name, policy_output.name


def quat_to_rotation(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("l", nargs="?", default="l", choices=["l"])
    parser.add_argument("--onnx", required=True, help="Path to the exported .onnx policy")
    parser.add_argument("--xml", default=os.path.join(SCRIPT_DIR, "w1w.xml"))
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=0.45)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Seconds to run; 0 means until the viewer closes")
    return parser.parse_args()


def main(
    xml_override=None,
    keyboard_cls=KeyboardController,
    initial_position=(0.0, 0.0, 0.48),
    initial_quaternion=(1.0, 0.0, 0.0, 0.0),
    camera_update=None,
    setup_viewer=None,
):
    args = parse_args()
    onnx_path = os.path.abspath(os.path.expanduser(args.onnx))
    if not os.path.isfile(onnx_path):
        raise SystemExit(f"ONNX policy not found: {onnx_path}")

    xml_path = xml_override or os.path.abspath(args.xml)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    policy, input_name, output_name = load_policy(onnx_path)

    qpos_adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES])
    dof_adr = np.array([model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES])
    actuator_ids = np.array([mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace("_joint", ""))
        for name in JOINT_NAMES])

    def reset_robot():
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = initial_position
        data.qpos[3:7] = initial_quaternion
        data.qpos[qpos_adr] = DEFAULT_POS
        mujoco.mj_forward(model, data)

    reset_robot()

    history = np.zeros(348, dtype=np.float32)
    last_action = np.zeros(16, dtype=np.float32)
    command = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    height = float(np.clip(args.height, 0.4, 0.62))
    keyboard = keyboard_cls(command, height)
    control_steps = max(1, round(0.02 / model.opt.timestep))
    start = time.monotonic()

    print(f"Loaded ONNX policy: {onnx_path}")
    print(f"Command: vx={args.vx:.2f}, vy={args.vy:.2f}, "
          f"wz={args.wz:.2f}, height={height:.2f}")
    print("\n=== 键盘控制 ===")
    print("方向键/W/S: 前后，方向键/A/D: 左右，Q/E: 转向")
    print("R/F: 降低/升高目标高度，Z: 重置高度")
    print("T: 重置机器人，X: 紧急停止；移动键松开即停\n")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=keyboard.key_callback
    ) as viewer:
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        if base_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_id
        if setup_viewer is not None:
            setup_viewer(viewer, model, data, keyboard)
        step = 0
        while viewer.is_running():
            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break
            loop_start = time.monotonic()
            keyboard.update_command()
            height = keyboard.height
            if keyboard.consume_reset():
                reset_robot()
                history.fill(0.0)
                last_action.fill(0.0)
                reset_extra = getattr(keyboard, "reset_simulation_state", None)
                if reset_extra is not None:
                    reset_extra()
                print("[RESET] 机器人状态已重置")
            apply_forces = getattr(keyboard, "apply_external_forces", None)
            if apply_forces is not None:
                state_changed = bool(apply_forces(model, data))
                if state_changed:
                    # A teleported fall pose invalidates all stacked frames.
                    history.fill(0.0)
                    last_action.fill(0.0)

            joint_pos = data.qpos[qpos_adr].copy()
            joint_vel = data.qvel[dof_adr].copy()
            pos_obs = joint_pos - DEFAULT_POS
            pos_obs[WHEEL_INDICES] = 0.0
            rotation = quat_to_rotation(data.qpos[3:7])
            gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
            # MuJoCo free-joint angular velocity is already body-local.
            gyro_body = data.qvel[3:6].copy()

            one_step = np.concatenate((
                command * np.array([2.0, 2.0, 0.25]),
                gyro_body * 0.25,
                gravity_body,
                pos_obs,
                joint_vel * 0.05,
                last_action,
                np.array([(height - 0.51) / 0.11]),
            )).astype(np.float32)
            history = np.concatenate((one_step, history[:-58])).astype(np.float32)

            obs = history[np.newaxis, :]
            action = policy.run([output_name], {input_name: obs})[0][0]
            action = np.clip(action, -100.0, 100.0)

            target = DEFAULT_POS + 0.25 * action
            target[HIP_INDICES] = DEFAULT_POS[HIP_INDICES] + 0.25 * action[HIP_INDICES]
            torque = np.zeros(16)
            torque[LEG_INDICES] = (
                KP[LEG_INDICES] * (target[LEG_INDICES] - joint_pos[LEG_INDICES])
                - KD[LEG_INDICES] * joint_vel[LEG_INDICES]
            )
            wheel_target = action[WHEEL_INDICES] * 1.5 * WHEEL_DIRECTIONS
            torque[WHEEL_INDICES] = 1.5 * (wheel_target - joint_vel[WHEEL_INDICES])
            torque = np.clip(torque, model.actuator_ctrlrange[actuator_ids, 0],
                             model.actuator_ctrlrange[actuator_ids, 1])
            data.ctrl[actuator_ids] = torque
            last_action = action.astype(np.float32)

            for _ in range(control_steps):
                mujoco.mj_step(model, data)
            if camera_update is not None:
                camera_update(viewer, model, data, keyboard)
            viewer.sync()
            step += 1

            remaining = 0.02 - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
