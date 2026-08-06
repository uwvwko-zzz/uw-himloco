#!/usr/bin/env python3
"""00000JQG MuJoCo 模型加载、关节映射和动力学稳定性验证。"""

import argparse
import math
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_JOINTS = (
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
)


class KeyboardController:
    """使用 evdev 获取按住/松开状态，功能键由 MuJoCo 回调处理。"""

    GLFW_KEY_R = 82
    GLFW_KEY_F = 70
    GLFW_KEY_Z = 90
    GLFW_KEY_T = 84
    GLFW_KEY_X = 88

    def __init__(self, config):
        self.config = config
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
            evdev.ecodes.KEY_W: "w",
            evdev.ecodes.KEY_S: "s",
            evdev.ecodes.KEY_A: "a",
            evdev.ecodes.KEY_D: "d",
            evdev.ecodes.KEY_Q: "q",
            evdev.ecodes.KEY_E: "e",
            evdev.ecodes.KEY_UP: "w",
            evdev.ecodes.KEY_DOWN: "s",
            evdev.ecodes.KEY_LEFT: "a",
            evdev.ecodes.KEY_RIGHT: "d",
        }

        permission_denied = False
        for path in evdev.list_devices():
            try:
                device = evdev.InputDevice(path)
                keys = device.capabilities().get(evdev.ecodes.EV_KEY, ())
                if (
                    evdev.ecodes.KEY_W in keys
                    or evdev.ecodes.KEY_UP in keys
                ):
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
            thread = threading.Thread(
                target=self._read_device, args=(device,), daemon=True
            )
            thread.start()
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
        """方向键/WASD/QE：按住运动，松开归零。"""
        with self.lock:
            pressed = self.pressed.copy()
        command = self.config["command"]
        command["linear_x"] = (
            1.0 if "w" in pressed else -1.0 if "s" in pressed else 0.0
        )
        command["linear_y"] = (
            1.0 if "a" in pressed else -1.0 if "d" in pressed else 0.0
        )
        command["yaw"] = (
            1.0 if "q" in pressed else -1.0 if "e" in pressed else 0.0
        )

    def key_callback(self, keycode):
        height_min, height_max = self.config["observation"]["height_range"]
        command = self.config["command"]
        if keycode == self.GLFW_KEY_R:
            command["height"] = max(
                float(height_min), float(command["height"]) - 0.02
            )
        elif keycode == self.GLFW_KEY_F:
            command["height"] = min(
                float(height_max), float(command["height"]) + 0.02
            )
        elif keycode == self.GLFW_KEY_Z:
            command["height"] = float(
                self.config["simulation"]["initial_height"]
            )
        elif keycode == self.GLFW_KEY_T:
            self.reset_requested = True
        elif keycode == self.GLFW_KEY_X:
            with self.lock:
                self.pressed.clear()
            command["linear_x"] = 0.0
            command["linear_y"] = 0.0
            command["yaw"] = 0.0

    def consume_reset(self):
        requested = self.reset_requested
        self.reset_requested = False
        return requested


def load_config(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def joint_addresses(model):
    qpos_adr = []
    dof_adr = []
    for name in EXPECTED_JOINTS:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"MJCF 缺少关节: {name}")
        qpos_adr.append(model.jnt_qposadr[joint_id])
        dof_adr.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_adr), np.asarray(dof_adr)


def actuator_joint_names(model):
    names = []
    for actuator_id in range(model.nu):
        transmission_id = model.actuator_trnid[actuator_id, 0]
        names.append(
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, transmission_id
            )
        )
    return tuple(names)


def set_initial_state(model, data, config, qpos_adr):
    mujoco.mj_resetData(model, data)
    initial_position = config["simulation"].get(
        "initial_position",
        [0.0, 0.0, config["simulation"]["initial_height"]],
    )
    initial_quaternion = config["simulation"].get(
        "initial_quaternion", [1.0, 0.0, 0.0, 0.0]
    )
    data.qpos[0:3] = np.asarray(initial_position, dtype=np.float64)
    data.qpos[3:7] = np.asarray(initial_quaternion, dtype=np.float64)
    targets = np.asarray(
        [
            config["default_joint_angles"][name]
            for name in EXPECTED_JOINTS
        ],
        dtype=np.float64,
    )
    data.qpos[qpos_adr] = targets
    mujoco.mj_forward(model, data)
    return targets


def validate_structure(model):
    if (model.nq, model.nv, model.nu) != (19, 18, 12):
        raise ValueError(
            "模型维度不符合 6-DoF 浮动基座 + 12 个关节："
            f"nq={model.nq}, nv={model.nv}, nu={model.nu}"
        )
    actuator_joints = actuator_joint_names(model)
    if actuator_joints != EXPECTED_JOINTS:
        raise ValueError(
            "执行器顺序与预期关节顺序不一致：\n"
            f"实际: {actuator_joints}\n预期: {EXPECTED_JOINTS}"
        )


def quat_to_rotation_matrix(quat_wxyz):
    """MuJoCo 的 [w, x, y, z] 四元数转换为旋转矩阵。"""
    w, x, y, z = quat_wxyz
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class ONNXPolicy:
    """按 Isaac Gym 的 46D×6 历史观测运行导出的 HIMLoco 策略。"""

    def __init__(self, model_path, config, default_positions):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "使用 --onnx 需要安装 onnxruntime 或 onnxruntime-gpu"
            ) from exc

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.config = config
        self.default_positions = default_positions
        self.action = np.zeros(12, dtype=np.float32)

        obs_cfg = config["observation"]
        self.one_step_size = 46
        self.history_length = int(obs_cfg["history_length"])
        self.history = np.zeros(
            self.one_step_size * self.history_length, dtype=np.float32
        )
        self.initialized = False

        input_size = self.input.shape[-1]
        expected_size = self.history.size
        if isinstance(input_size, int) and input_size != expected_size:
            raise ValueError(
                f"ONNX 输入维度为 {input_size}，JQG 配置需要 {expected_size}"
            )
        output_size = self.output.shape[-1]
        if isinstance(output_size, int) and output_size != 12:
            raise ValueError(f"ONNX 输出维度应为 12，实际为 {output_size}")

    def reset(self):
        self.action.fill(0.0)
        self.history.fill(0.0)
        self.initialized = False

    def build_observation(self, data, qpos_adr, dof_adr):
        obs_cfg = self.config["observation"]
        command_cfg = self.config["command"]
        rotation = quat_to_rotation_matrix(data.qpos[3:7])

        command = np.asarray(
            [
                command_cfg["linear_x"],
                command_cfg["linear_y"],
                command_cfg["yaw"],
            ],
            dtype=np.float64,
        )
        command *= np.asarray(obs_cfg["command_scale"], dtype=np.float64)

        # MuJoCo free joint 的角速度 qvel[3:6] 已经位于机身坐标系，
        # 无需像世界坐标系重力一样再做一次逆旋转。
        body_ang_vel = data.qvel[3:6].copy()
        body_ang_vel *= float(obs_cfg["angular_velocity_scale"])
        projected_gravity = rotation.T @ np.asarray([0.0, 0.0, -1.0])

        joint_position = (
            data.qpos[qpos_adr] - self.default_positions
        ) * float(obs_cfg["joint_position_scale"])
        joint_velocity = data.qvel[dof_adr] * float(
            obs_cfg["joint_velocity_scale"]
        )

        height_min, height_max = obs_cfg["height_range"]
        height_center = 0.5 * (height_min + height_max)
        height_half_range = 0.5 * (height_max - height_min)
        if height_half_range <= 0:
            raise ValueError("observation.height_range 必须满足 max > min")
        height_obs = (
            float(command_cfg["height"]) - height_center
        ) / height_half_range

        observation = np.concatenate(
            (
                command,
                body_ang_vel,
                projected_gravity,
                joint_position,
                joint_velocity,
                self.action,
                np.asarray([height_obs]),
            )
        ).astype(np.float32)
        if observation.size != self.one_step_size:
            raise RuntimeError(
                f"单步观测维度错误: {observation.size} != {self.one_step_size}"
            )
        return observation

    def infer(self, data, qpos_adr, dof_adr):
        observation = self.build_observation(data, qpos_adr, dof_adr)
        if not self.initialized:
            self.history = np.tile(observation, self.history_length)
            self.initialized = True
        else:
            self.history[self.one_step_size :] = self.history[
                : -self.one_step_size
            ]
            self.history[: self.one_step_size] = observation

        output = self.session.run(
            [self.output.name],
            {self.input.name: self.history.reshape(1, -1)},
        )[0]
        self.action = np.asarray(output[0], dtype=np.float32)
        if self.action.shape != (12,) or not np.isfinite(self.action).all():
            raise FloatingPointError(
                f"ONNX 策略输出无效，shape={self.action.shape}"
            )
        action_scale = float(self.config["control"]["action_scale"])
        return self.default_positions + action_scale * self.action


def simulate(
    model,
    data,
    config,
    targets,
    qpos_adr,
    dof_adr,
    duration,
    viewer,
    policy=None,
    keyboard=None,
):
    kp = float(config["control"]["kp"])
    kd = float(config["control"]["kd"])
    torque_limit = float(config["control"]["torque_limit"])
    velocity_limit = float(config["control"]["velocity_limit"])
    steps = max(1, int(duration / model.opt.timestep))

    min_height = math.inf
    max_height = -math.inf
    max_torque = 0.0
    max_velocity = 0.0
    max_joint_error = 0.0

    decimation = int(config["control"]["decimation"])
    step = 0
    while viewer is not None or step < steps:
        if viewer is not None and not viewer.is_running():
            break
        step_start = time.perf_counter()
        if keyboard is not None:
            keyboard.update_command()
            if keyboard.consume_reset():
                targets = set_initial_state(
                    model, data, config, qpos_adr
                )
                if policy is not None:
                    policy.reset()
        if policy is not None and step % decimation == 0:
            targets = policy.infer(data, qpos_adr, dof_adr)
        positions = data.qpos[qpos_adr]
        velocities = data.qvel[dof_adr]
        torques = kp * (targets - positions) - kd * velocities
        data.ctrl[:] = np.clip(torques, -torque_limit, torque_limit)
        mujoco.mj_step(model, data)

        if not (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        ):
            raise FloatingPointError(
                f"MuJoCo 状态在 t={data.time:.4f}s 出现 NaN/Inf"
            )

        min_height = min(min_height, float(data.qpos[2]))
        max_height = max(max_height, float(data.qpos[2]))
        max_torque = max(max_torque, float(np.max(np.abs(data.ctrl))))
        max_velocity = max(
            max_velocity, float(np.max(np.abs(data.qvel[dof_adr])))
        )
        max_joint_error = max(
            max_joint_error,
            float(np.max(np.abs(targets - data.qpos[qpos_adr]))),
        )

        if viewer is not None:
            viewer.sync()
            remaining = model.opt.timestep - (
                time.perf_counter() - step_start
            )
            if remaining > 0:
                time.sleep(remaining)
        step += 1

    return {
        "simulated_time": data.time,
        "base_height_min": min_height,
        "base_height_max": max_height,
        "max_abs_torque": max_torque,
        "max_abs_joint_velocity": max_velocity,
        "max_abs_joint_error": max_joint_error,
        "velocity_limit": velocity_limit,
        "contacts": data.ncon,
        "mode": "ONNX policy" if policy is not None else "fixed-pose PD",
    }


def print_report(model, report):
    total_mass = float(np.sum(model.body_mass))
    print("\n=== 00000JQG MuJoCo 验证结果 ===")
    print(f"MuJoCo:              {mujoco.__version__}")
    print(f"body / geom / sensor:{model.nbody} / {model.ngeom} / {model.nsensor}")
    print(f"nq / nv / nu:        {model.nq} / {model.nv} / {model.nu}")
    print(f"总质量:              {total_mass:.6f} kg")
    print(f"控制模式:            {report['mode']}")
    print(f"仿真时长:            {report['simulated_time']:.3f} s")
    print(
        "机身高度范围:        "
        f"{report['base_height_min']:.4f} .. "
        f"{report['base_height_max']:.4f} m"
    )
    print(f"最大关节误差:        {report['max_abs_joint_error']:.4f} rad")
    print(f"最大绝对力矩:        {report['max_abs_torque']:.3f} N·m")
    print(
        "最大关节速度:        "
        f"{report['max_abs_joint_velocity']:.3f} rad/s "
        f"(电机限制 {report['velocity_limit']:.2f})"
    )
    print(f"结束时接触数:        {report['contacts']}")
    print("状态有限性:          PASS（无 NaN/Inf）")


def main(default_config=None, model_override=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config or SCRIPT_DIR / "config.yaml",
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="可选 ONNX 策略路径；不指定时执行固定姿态 PD 验证",
    )
    parser.add_argument("--cmd-x", type=float, default=None)
    parser.add_argument("--cmd-y", type=float, default=None)
    parser.add_argument("--cmd-yaw", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="不打开 MuJoCo viewer"
    )
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    model_path = (
        Path(model_override).resolve()
        if model_override is not None
        else (args.config.resolve().parent / config["model"]).resolve()
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = float(config["simulation"]["timestep"])
    validate_structure(model)

    qpos_adr, dof_adr = joint_addresses(model)
    data = mujoco.MjData(model)
    targets = set_initial_state(model, data, config, qpos_adr)
    command_overrides = {
        "linear_x": args.cmd_x,
        "linear_y": args.cmd_y,
        "yaw": args.cmd_yaw,
        "height": args.height,
    }
    for name, value in command_overrides.items():
        if value is not None:
            config["command"][name] = float(value)

    policy = None
    if args.onnx is not None:
        onnx_path = args.onnx.expanduser().resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"找不到 ONNX 模型: {onnx_path}")
        policy = ONNXPolicy(onnx_path, config, targets.copy())
        print(f"ONNX 模型: {onnx_path}")
        print(
            "运动命令: "
            f"vx={config['command']['linear_x']}, "
            f"vy={config['command']['linear_y']}, "
            f"yaw={config['command']['yaw']}, "
            f"height={config['command']['height']}"
        )
    duration = (
        float(args.duration)
        if args.duration is not None
        else float(config["simulation"]["duration"])
    )

    if args.headless:
        report = simulate(
            model,
            data,
            config,
            targets,
            qpos_adr,
            dof_adr,
            duration,
            None,
            policy,
            None,
        )
    else:
        from mujoco import viewer as mj_viewer

        keyboard = KeyboardController(config)
        print("\n=== 键盘控制 ===")
        print("方向键/W/S: 前后，方向键/A/D: 左右，Q/E: 转向")
        print("R/F: 降低/升高目标高度，Z: 重置高度")
        print("T: 重置机器人，X: 紧急停止；移动键松开即停\n")
        with mj_viewer.launch_passive(
            model, data, key_callback=keyboard.key_callback
        ) as viewer:
            viewer.cam.distance = 1.7
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            base_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
            )
            if base_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = base_id
            report = simulate(
                model,
                data,
                config,
                targets,
                qpos_adr,
                dof_adr,
                duration,
                viewer,
                policy,
                keyboard,
            )

    print_report(model, report)


if __name__ == "__main__":
    main()
