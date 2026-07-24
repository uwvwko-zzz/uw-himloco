"""
Dog Sim2Sim — IsaacGym → MuJoCo + 模型热切换

基于 dog_policy_test_12.cpp 的模型切换逻辑:
  - YAML 配置模型列表 (models 列表, 每项含 name / path / obs_dim)
  - Xbox 手柄: Back=上一个模型, Start=下一个模型
  - 切换时自动关闭 RL, 按 Y 重新启动
  - 环形切换 (wrap around)

关节顺序:
  MuJoCo XML 顺序: FL(0-2), FR(3-5), RR(6-8), RL(9-11)
  IsaacGym dof 顺序: FL(0-2), FR(3-5), RL(6-8), RR(9-11)

运行命令:
  python3 mujoco/dog/play_xbox_onnx_46_switch.py dog_switch.yaml
  python3 mujoco/dog/play_xbox_onnx_46_switch.py dog_switch.yaml --no-policy
"""
import time
import os
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml
import threading


# ============================================================
#  RL/RR 映射 — IsaacGym 内部重排关节顺序
# ============================================================
MUJOCO_TO_ISAAC = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]
ISAAC_TO_MUJOCO = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]

# 默认关节角度 — IsaacGym dof 顺序: FL, FR, RL, RR
DEFAULT_ANGLES_ISAAC = np.array([
    -0.1, -0.8, -1.5,    # FL
     0.1,  0.8,  1.5,    # FR
     0.1, -1.0, -1.5,    # RL
    -0.1,  1.0,  1.5,    # RR
], dtype=np.float64)

DEFAULT_ANGLES_MUJOCO = DEFAULT_ANGLES_ISAAC[ISAAC_TO_MUJOCO]

TAU_LIMIT_HIP_THIGH = 23.7
TAU_LIMIT_CALF = 35.55


def quat_rotate_inverse(q, v):
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


class ObsHistoryBuffer:
    def __init__(self, history_len, single_obs_dim):
        self.history_len = history_len
        self.single_obs_dim = single_obs_dim
        self.total_dim = history_len * single_obs_dim
        self.buffer = np.zeros(self.total_dim, dtype=np.float32)

    def push(self, new_obs):
        self.buffer[self.single_obs_dim:] = self.buffer[:-self.single_obs_dim].copy()
        self.buffer[:self.single_obs_dim] = new_obs

    def get(self):
        return self.buffer.reshape(1, -1).copy()

    def reset(self):
        self.buffer[:] = 0.0

    def resize(self, new_single_obs_dim, history_len):
        """切换 obs_dim 时重建 buffer"""
        self.single_obs_dim = new_single_obs_dim
        self.history_len = history_len
        self.total_dim = history_len * new_single_obs_dim
        self.buffer = np.zeros(self.total_dim, dtype=np.float32)


# ============================================================
#  ★ 模型管理器 (对应 C++ DogPolicyTestNode 的模型管理部分)
# ============================================================
class ModelManager:
    """
    管理模型列表、加载、热切换
    对应 C++ 的 LoadModel / SwitchModel / ResetInferenceState
    """

    def __init__(self, model_list, default_index=0):
        self.model_list = model_list  # list of dict: {name, path, obs_dim}
        self.current_idx = 0
        self.policy = None
        self.input_name = None
        self.output_name = None
        self.current_obs_dim = 46

        if not model_list:
            raise ValueError("模型列表为空!")

        # 加载默认模型
        if default_index < 0 or default_index >= len(model_list):
            print(f"[MODEL] default_model_index={default_index} 越界, 使用 0")
            default_index = 0

        self.load_model(default_index)

    def load_model(self, index):
        """加载指定索引的模型 (对应 C++ LoadModel)"""
        if index < 0 or index >= len(self.model_list):
            print(f"[MODEL] index={index} 越界")
            return False

        entry = self.model_list[index]
        path = entry['path']
        obs_dim = entry['obs_dim']

        # 检查文件是否存在
        if not os.path.isfile(path):
            print(f"[MODEL] ✗ 模型文件不存在: {path}")
            return False

        # 创建 ONNX Session
        try:
            policy = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            input_name = policy.get_inputs()[0].name
            output_name = policy.get_outputs()[0].name
            shape = policy.get_inputs()[0].shape
            print(f"[MODEL] ONNX loaded: {shape} → {policy.get_outputs()[0].shape}")
        except Exception as e:
            print(f"[MODEL] ✗ ONNX 加载失败: {path} — {e}")
            return False

        # 替换当前模型
        self.policy = policy
        self.input_name = input_name
        self.output_name = output_name
        self.current_idx = index
        self.current_obs_dim = obs_dim

        print(f"[MODEL] ★ 模型已加载: [{index}] '{entry['name']}' obs={obs_dim} path={path}")
        return True

    def switch_model(self, direction):
        """
        切换模型 (对应 C++ SwitchModel)
        direction: -1=上一个, +1=下一个, 环形
        返回 True 表示切换成功
        """
        if len(self.model_list) <= 1:
            print("[MODEL] 只有一个模型, 无法切换")
            return False

        new_idx = self.current_idx + direction
        if new_idx < 0:
            new_idx = len(self.model_list) - 1
        if new_idx >= len(self.model_list):
            new_idx = 0

        old_name = self.model_list[self.current_idx]['name']
        new_name = self.model_list[new_idx]['name']

        print(f"\n{'='*60}")
        print(f"  ★ 切换模型: [{self.current_idx}] {old_name}  →  [{new_idx}] {new_name}")
        print(f"{'='*60}\n")

        if not self.load_model(new_idx):
            print(f"[MODEL] ★ 模型切换失败! 保持当前模型.")
            return False

        return True

    def print_model_list(self):
        """打印模型列表 (对应 C++ PrintModelList)"""
        print(f"\n{'='*60}")
        print(f"  ★ 模型列表 (共 {len(self.model_list)} 个)")
        print(f"{'='*60}")
        for i, m in enumerate(self.model_list):
            marker = " ◀ current" if i == self.current_idx else ""
            print(f"  [{i}] {m['name']:<24s} obs={m['obs_dim']}{marker}")
        print(f"{'='*60}")
        print(f"  手柄: Back=上一个  Start=下一个")
        print(f"  切换时自动关闭 RL, 按 Y 重新启动")
        print(f"{'='*60}\n")


def build_single_obs(quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                     last_action_isaac, default_angles_isaac,
                     cmd, cmd_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale,
                     clip_obs, height_cmd=0.25, obs_dim=46):
    """
    构建46维单步观测 — 所有关节量使用 IsaacGym dof 顺序: FL, FR, RL, RR
    第46维: 高度指令 (height_cmd - 0.25) / 0.1
    """
    obs = np.zeros(obs_dim, dtype=np.float32)
    idx = 0

    obs[idx:idx+3] = cmd * cmd_scale[:3]
    idx += 3

    omega_body = omega  # MuJoCo free-joint angular qvel is already in the local body frame
    obs[idx:idx+3] = omega_body.astype(np.float32) * ang_vel_scale
    idx += 3

    gravity_world = np.array([0., 0., -1.], dtype=np.float64)
    proj_gravity = quat_rotate_inverse(quat_xyzw, gravity_world)
    obs[idx:idx+3] = proj_gravity.astype(np.float32)
    idx += 3

    obs[idx:idx+12] = ((joint_q_isaac - default_angles_isaac) * dof_pos_scale).astype(np.float32)
    idx += 12

    obs[idx:idx+12] = (joint_dq_isaac * dof_vel_scale).astype(np.float32)
    idx += 12

    obs[idx:idx+12] = last_action_isaac
    idx += 12

    # 第46维: 高度指令 (obs_dim=46 时才有)
    if obs_dim >= 46:
        obs[45] = np.float32((height_cmd - 0.25) / 0.1)

    obs = np.clip(obs, -clip_obs, clip_obs)
    return obs


def reset_robot(model, data, default_angles_mujoco):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.42
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = default_angles_mujoco
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


# ============================================================
#  Xbox 手柄控制 (带模型切换按钮)
# ============================================================
import inputs

BTN_A = 0; BTN_B = 1; BTN_X = 2; BTN_Y = 3
BTN_LB = 4; BTN_RB = 5; BTN_BACK = 6; BTN_START = 7

# 全局状态
_gamepad_vx = 0.0
_gamepad_vy = 0.0
_gamepad_wz = 0.0
_gamepad_time = 0.0
_height_cmd = 0.25
_reset_flag = False
_rl_enabled = False
_stop_flag = False

# ★ 模型切换标志 (对应 C++ SwitchModel)
_switch_model_flag = 0   # 0=无, -1=上一个, +1=下一个

_raw_lx = 0.0
_raw_ly = 0.0
_raw_rx = 0.0

JOY_DEADBAND = 0.1
JOY_TIMEOUT_SEC = 0.5
JOY_VX_MAX = 1.0
JOY_VY_MAX = 1.0
JOY_WZ_MAX = 1.0


def _apply_deadband(v):
    return 0.0 if abs(v) < JOY_DEADBAND else v


def _detect_gamepads():
    try:
        devices = inputs.devices.gamepads
    except Exception:
        return []
    if not devices:
        print("[GAMEPAD] 未检测到手柄!")
        return []
    print(f"[GAMEPAD] 检测到 {len(devices)} 个手柄:")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev.name}")
    return devices


_gp_devices = _detect_gamepads()
if _gp_devices:
    print(f"[GAMEPAD] 使用: {_gp_devices[0].name}")
else:
    print("[GAMEPAD] 警告: 没有手柄, 等待中...")


def _gamepad_thread():
    global _gamepad_vx, _gamepad_vy, _gamepad_wz, _gamepad_time
    global _height_cmd, _reset_flag, _rl_enabled, _stop_flag, _switch_model_flag
    global _raw_lx, _raw_ly, _raw_rx

    _gamepad_time = time.time()
    _event_count = 0
    _last_warn = time.time()

    while not _stop_flag:
        try:
            events = inputs.get_gamepad()
        except inputs.UnpluggedError:
            if time.time() - _last_warn > 5.0:
                print("[GAMEPAD] 设备未连接, 等待中...")
                _last_warn = time.time()
            time.sleep(0.5)
            continue
        except Exception:
            time.sleep(0.1)
            continue

        for event in events:
            _event_count += 1
            if _event_count == 1:
                print(f"[GAMEPAD] 收到第一个事件! type={event.ev_type} code={event.code}")

            if event.ev_type == 'Absolute':
                val = event.state / 32768.0 if event.state is not None else 0.0
                if event.code == 'ABS_X':
                    _raw_lx = val
                elif event.code == 'ABS_Y':
                    _raw_ly = val
                elif event.code == 'ABS_RX':
                    _raw_rx = val

                vx = -_apply_deadband(_raw_ly) * JOY_VX_MAX
                vy = -_apply_deadband(_raw_lx) * JOY_VY_MAX
                wz = -_apply_deadband(_raw_rx) * JOY_WZ_MAX
                _gamepad_vx = vx
                _gamepad_vy = vy
                _gamepad_wz = wz
                _gamepad_time = time.time()

            elif event.ev_type == 'Key':
                if event.code == 'BTN_SOUTH' and event.state == 1:       # A → 重置
                    _reset_flag = True
                elif event.code == 'BTN_EAST' and event.state == 1:      # B → 切换模型
                    _switch_model_flag = 1
                    print("[GAMEPAD] ★ B → 切换下一个模型")
                elif event.code == 'BTN_NORTH' and event.state == 1:     # X → 关闭RL
                    _rl_enabled = False
                    print("[GAMEPAD] ★ RL 已关闭")
                elif event.code == 'BTN_WEST' and event.state == 1:      # Y → 开启RL
                    _rl_enabled = True
                    print("[GAMEPAD] ★ RL 已开启")
                elif event.code == 'BTN_TR' and event.state == 1:        # RB → 站起
                    _height_cmd = min(0.35, _height_cmd + 0.02)
                elif event.code == 'BTN_TL' and event.state == 1:        # LB → 蹲下
                    _height_cmd = max(0.20, _height_cmd - 0.02)
                # ★ 模型切换按钮 (对应 C++ N/M 键)
                elif event.code == 'BTN_SELECT' and event.state == 1:    # Back → 上一个模型
                    _switch_model_flag = -1
                    print("[GAMEPAD] ★ Back → 切换上一个模型")
                elif event.code == 'BTN_START' and event.state == 1:     # Start → 下一个模型
                    _switch_model_flag = 1
                    print("[GAMEPAD] ★ Start → 切换下一个模型")


_gp_thread = threading.Thread(target=_gamepad_thread, daemon=True)
_gp_thread.start()


def get_commands():
    # 摇杆值本身就是状态 (已过 deadband), 不需要超时
    # 只在断连时用长超时归零 (安全机制)
    if _gamepad_time > 0 and time.time() - _gamepad_time > 5.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([_gamepad_vx, _gamepad_vy, _gamepad_wz], dtype=np.float32)


# ============================================================
#  MuJoCo 键盘回调 — 功能键 + 模型切换
# ============================================================
GLFW_KEY_R = 82;  GLFW_KEY_F = 70
GLFW_KEY_Z = 90;  GLFW_KEY_T = 84
GLFW_KEY_Y = 89;  GLFW_KEY_X = 88
GLFW_KEY_B = 66   # ★ 模型切换键 (B → 下一个模型)


def key_callback(keycode):
    global _height_cmd, _reset_flag, _rl_enabled, _switch_model_flag
    print(f"[DEBUG] keycode={keycode}")  # ★ 调试: 查看接收到的键码
    if keycode == GLFW_KEY_R:
        _height_cmd = max(0.20, _height_cmd - 0.02)
    elif keycode == GLFW_KEY_F:
        _height_cmd = min(0.35, _height_cmd + 0.02)
    elif keycode == GLFW_KEY_Z:
        _height_cmd = 0.25
    elif keycode == GLFW_KEY_T:
        _reset_flag = True
    elif keycode == GLFW_KEY_Y:
        _rl_enabled = not _rl_enabled
        print(f"[KEY] ★ RL {'已开启' if _rl_enabled else '已关闭'}")
    elif keycode == GLFW_KEY_X:
        _rl_enabled = False
        print("[KEY] ★ 紧急停止, RL 已关闭")
    # ★ 模型切换 (B → 下一个模型, 环形)
    elif keycode == GLFW_KEY_B:
        _switch_model_flag = 1
        print("[KEY] ★ B → 切换下一个模型")


# ============================================================
#  主程序
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dog Sim2Sim with model hot-switching")
    parser.add_argument("config_file", type=str, help="YAML 配置文件名 (在 config/ 目录下)")
    parser.add_argument("--no-policy", action="store_true", help="不加载策略, 仅保持默认站姿")
    args = parser.parse_args()

    base = "/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym"
    config_path = f"{base}/mujoco/dog/config/{args.config_file}"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    simulation_duration = config["simulation_duration"]
    simulation_dt       = config["simulation_dt"]
    control_decimation  = config["control_decimation"]
    kps            = np.array(config["kps"], dtype=np.float64)
    kds            = np.array(config["kds"], dtype=np.float64)
    action_scale   = config["action_scale"]
    ang_vel_scale  = config["ang_vel_scale"]
    dof_pos_scale  = config["dof_pos_scale"]
    dof_vel_scale  = config["dof_vel_scale"]
    cmd_scale      = np.array(config["cmd_scale"], dtype=np.float32)

    # 读取手柄参数 (覆盖默认值)
    if "joy_vx_max" in config:
        JOY_VX_MAX = config["joy_vx_max"]
    if "joy_vy_max" in config:
        JOY_VY_MAX = config["joy_vy_max"]
    if "joy_wz_max" in config:
        JOY_WZ_MAX = config["joy_wz_max"]
    if "joy_deadband" in config:
        JOY_DEADBAND = config["joy_deadband"]
    if "joy_timeout_sec" in config:
        JOY_TIMEOUT_SEC = config["joy_timeout_sec"]

    # 力矩限制
    tau_limits_mujoco = np.array([
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
    ])

    HISTORY_LEN = 6
    NUM_ACTIONS = 12

    # ==========================
    # ★ 加载模型列表
    # ==========================
    model_manager = None
    if not args.no_policy:
        model_list = config.get("models", [])
        if not model_list:
            print("[ERROR] YAML 中没有 models 列表! 请检查配置文件.")
            exit(1)

        default_idx = config.get("default_model_index", 0)

        # 打印模型列表
        print(f"\n{'='*60}")
        print(f"  ★ 模型列表 (共 {len(model_list)} 个)")
        print(f"{'='*60}")
        for i, m in enumerate(model_list):
            marker = " ◀ default" if i == default_idx else ""
            print(f"  [{i}] {m['name']:<24s} obs={m['obs_dim']}{marker}")
            print(f"      {m['path']}")
        print(f"{'='*60}")
        print(f"  手柄: Back=上一个  Start=下一个  键盘: B")
        print(f"  切换时自动关闭 RL, 按 Y 重新启动")
        print(f"{'='*60}\n")

        model_manager = ModelManager(model_list, default_idx)

    # MuJoCo
    xml_path = config.get("xml_path", f"{base}/resources/robots/dog/xml/dog_terrain.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = simulation_dt
    mj_data = mujoco.MjData(mj_model)

    print(f"\n[验证] MuJoCo joint 顺序:")
    for i in range(mj_model.njnt):
        jn = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jn: print(f"  [{i}] {jn}")

    reset_robot(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)

    q_isaac = mj_data.qpos[7:19][MUJOCO_TO_ISAAC]
    print(f"\n[验证] qpos(MuJoCo→Isaac): {q_isaac}")
    print(f"[验证] 期望(Isaac):         {DEFAULT_ANGLES_ISAAC}")
    print(f"[验证] {'✓ 一致' if np.allclose(q_isaac, DEFAULT_ANGLES_ISAAC) else '✗ 不一致!'}")

    # 推理状态
    current_obs_dim = model_manager.current_obs_dim if model_manager else 46
    obs_history = ObsHistoryBuffer(HISTORY_LEN, current_obs_dim)
    target_q_mujoco  = DEFAULT_ANGLES_MUJOCO.copy()
    action_isaac     = np.zeros(NUM_ACTIONS, dtype=np.float64)
    last_action_isaac = np.zeros(NUM_ACTIONS, dtype=np.float32)
    count = 0

    # 预热
    for _ in range(HISTORY_LEN):
        quat_wxyz = mj_data.qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        omega = mj_data.qvel[3:6].astype(np.float64)
        joint_q_isaac = mj_data.qpos[7:19].astype(np.float64)[MUJOCO_TO_ISAAC]
        joint_dq_isaac = mj_data.qvel[6:18].astype(np.float64)[MUJOCO_TO_ISAAC]
        obs = build_single_obs(
            quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
            last_action_isaac, DEFAULT_ANGLES_ISAAC,
            np.zeros(3, dtype=np.float32), cmd_scale,
            ang_vel_scale, dof_pos_scale, dof_vel_scale, 100.0,
            height_cmd=0.25, obs_dim=current_obs_dim
        )
        obs_history.push(obs)

    print(f"[INFO] 预热完成, norm={np.linalg.norm(obs_history.buffer):.4f}")
    print(f"\n  ★ Xbox 手柄控制:")
    print(f"    左摇杆: 前后(vx) 左右(vy)   右摇杆 X: 转向(wz)")
    print(f"    A: 重置  B: 切换模型  X: 关闭RL  Y: 开启RL")
    print(f"    LB: 蹲下  RB: 站起")
    print(f"    Back: 上一个模型  Start: 下一个模型  (★ 新增)")
    print(f"  ★ MuJoCo 窗口键盘:")
    print(f"    Y: 切换RL  T: 重置  R/F: 高度  Z: 默认高度")
    print(f"    B: 切换下一个模型  (★ 新增)")
    print(f"    超时 {JOY_TIMEOUT_SEC}s 无输入 → 自动归零\n")

    # 打印当前模型列表
    if model_manager:
        model_manager.print_model_list()

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        trunk_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        if trunk_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = trunk_id
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 135

        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            cmd = get_commands()

            # ★ 处理模型切换 (对应 C++ SwitchModel)
            if _switch_model_flag != 0 and model_manager:
                direction = _switch_model_flag
                _switch_model_flag = 0

                if model_manager.switch_model(direction):
                    # 切换成功 → 重置推理状态 (对应 C++ ResetInferenceState)
                    new_dim = model_manager.current_obs_dim
                    obs_history.resize(new_dim, HISTORY_LEN)
                    action_isaac[:] = 0
                    last_action_isaac[:] = 0
                    count = 0
                    current_obs_dim = new_dim

                    # 自动关闭 RL (对应 C++: 切换时自动关闭 RL)
                    _rl_enabled = False
                    print(f"[MODEL] ★ RL 已关闭, 按 Y 重新启动")

                    # 重新预热
                    for _ in range(HISTORY_LEN):
                        quat_wxyz = mj_data.qpos[3:7]
                        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                        omega = mj_data.qvel[3:6].astype(np.float64)
                        joint_q_isaac = mj_data.qpos[7:19].astype(np.float64)[MUJOCO_TO_ISAAC]
                        joint_dq_isaac = mj_data.qvel[6:18].astype(np.float64)[MUJOCO_TO_ISAAC]
                        obs = build_single_obs(
                            quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                            last_action_isaac, DEFAULT_ANGLES_ISAAC,
                            np.zeros(3, dtype=np.float32), cmd_scale,
                            ang_vel_scale, dof_pos_scale, dof_vel_scale, 100.0,
                            height_cmd=_height_cmd, obs_dim=current_obs_dim
                        )
                        obs_history.push(obs)

                    model_manager.print_model_list()

            # 处理重置
            if _reset_flag:
                reset_robot(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)
                obs_history.reset()
                last_action_isaac[:] = 0
                action_isaac[:] = 0
                count = 0
                _reset_flag = False
                print("[RESET] 机器人已重置")

            # 读取 MuJoCo 状态
            joint_q_mujoco  = mj_data.qpos[7:19].astype(np.float64)
            joint_dq_mujoco = mj_data.qvel[6:18].astype(np.float64)
            joint_q_isaac   = joint_q_mujoco[MUJOCO_TO_ISAAC]
            joint_dq_isaac  = joint_dq_mujoco[MUJOCO_TO_ISAAC]

            quat_wxyz = mj_data.qpos[3:7]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            omega = mj_data.qvel[3:6].astype(np.float64)

            if count % control_decimation == 0:
                if not args.no_policy and _rl_enabled and model_manager:
                    single_obs = build_single_obs(
                        quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                        last_action_isaac, DEFAULT_ANGLES_ISAAC,
                        cmd, cmd_scale,
                        ang_vel_scale, dof_pos_scale, dof_vel_scale, 100.0,
                        height_cmd=_height_cmd, obs_dim=current_obs_dim
                    )
                    obs_history.push(single_obs)
                    obs_input = obs_history.get()

                    action_raw = model_manager.policy.run(
                        [model_manager.output_name],
                        {model_manager.input_name: obs_input}
                    )[0][0]
                    action_isaac[:] = np.clip(action_raw, -10.0, 10.0)
                    last_action_isaac = action_isaac.astype(np.float32)

                    target_q_isaac = action_isaac * action_scale + DEFAULT_ANGLES_ISAAC
                    target_q_mujoco = target_q_isaac[ISAAC_TO_MUJOCO]
                else:
                    target_q_mujoco = DEFAULT_ANGLES_MUJOCO.copy()

            # PD 控制
            tau = pd_control(target_q_mujoco, joint_q_mujoco, kps,
                           np.zeros(NUM_ACTIONS), joint_dq_mujoco, kds)
            tau = np.clip(tau, -tau_limits_mujoco, tau_limits_mujoco)
            mj_data.ctrl[:NUM_ACTIONS] = tau

            mujoco.mj_step(mj_model, mj_data)
            count += 1

            if count % (control_decimation * 50) == 0:
                grav = quat_rotate_inverse(quat_xyzw, np.array([0., 0., -1.]))
                h = mj_data.qpos[2]
                model_name = model_manager.model_list[model_manager.current_idx]['name'] if model_manager else "none"
                rl_str = "ON" if _rl_enabled else "OFF"
                print(f"[{time.time()-start:.1f}s] Step {count} H={h:.3f}(cmd={_height_cmd:.2f}) "
                      f"vx={mj_data.qvel[0]:.2f} vy={mj_data.qvel[1]:.2f} wz={mj_data.qvel[5]:.2f} "
                      f"grav_z={grav[2]:.3f} "
                      f"act=[{action_isaac.min():.2f},{action_isaac.max():.2f}] "
                      f"model={model_name} RL={rl_str}")

            viewer.sync()
            elapsed = time.time() - step_start
            if simulation_dt - elapsed > 0:
                time.sleep(simulation_dt - elapsed)

    print("\n[INFO] 仿真结束")