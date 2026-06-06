"""
Dog Sim2Sim — IsaacGym → MuJoCo

关节顺序:
  MuJoCo XML 顺序 (即 URDF 顺序): FL(0-2), FR(3-5), RR(6-8), RL(9-11)
  IsaacGym dof 顺序 (内部重排):    FL(0-2), FR(3-5), RL(6-8), RR(9-11)
  → 后两条腿反了! 需要 RL/RR 映射!
"""
import time
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


# ============================================================
#  RL/RR 映射 — IsaacGym 内部重排关节顺序
# ============================================================
# MuJoCo qpos[7:19] 顺序: FL(0-2), FR(3-5), RR(6-8), RL(9-11)
# IsaacGym dof 顺序:      FL(0-2), FR(3-5), RL(6-8), RR(9-11)
#
# MuJoCo[i] -> Isaac[j]:
#   MuJoCo 0-5  (FL,FR) -> Isaac 0-5  (FL,FR)  不变
#   MuJoCo 6-8  (RR)    -> Isaac 9-11 (RR)
#   MuJoCo 9-11 (RL)    -> Isaac 6-8  (RL)

MUJOCO_TO_ISAAC = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]
ISAAC_TO_MUJOCO = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]

# 默认关节角度 — IsaacGym dof 顺序: FL, FR, RL, RR
# 来自 dog_config.py default_joint_angles
# FL: hip=-0.1, thigh=-0.8, calf=-1.5
# FR: hip=0.1,  thigh=0.8,  calf=1.5
# RL: hip=0.1,  thigh=-1.0, calf=-1.5
# RR: hip=-0.1, thigh=1.0,  calf=1.5
DEFAULT_ANGLES_ISAAC = np.array([
    -0.1, -0.8, -1.5,    # FL (Isaac dof 0-2)
     0.1,  0.8,  1.5,    # FR (Isaac dof 3-5)
     0.1, -1.0, -1.5,    # RL (Isaac dof 6-8)
    -0.1,  1.0,  1.5,    # RR (Isaac dof 9-11)
], dtype=np.float64)

# MuJoCo 顺序: FL, FR, RR, RL — 通过映射从 Isaac 顺序得到
DEFAULT_ANGLES_MUJOCO = DEFAULT_ANGLES_ISAAC[ISAAC_TO_MUJOCO]

# 力矩限制 (来自 URDF actuatorfrcrange / dog.xml)
TAU_LIMIT_HIP_THIGH = 23.7   # hip, thigh 关节力矩限制 [Nm]
TAU_LIMIT_CALF = 35.55        # calf 关节力矩限制 [Nm]
OUTPUT_PRINT_SCALE = 0.25


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


# ============================================================
#  Xbox 手柄控制 (与 dog_policy_test_11.cpp JoyCallback 一致)
#  左摇杆: 前后/左右 (vy, vx)  右摇杆 X: 转向 (wz)
#  A: 启动RL  B: 重置  X: 紧急停止  Y: 打印action
#  RB: 站起  LB: 蹲下  Back: 重置高度  Start: 不用
#  ============================================================

import inputs
import threading

# 手柄参数 (与 test_11 的 YAML 参数对应)
JOY_VX_MAX = 1.0      # 左摇杆 Y → 前进最大速度
JOY_VY_MAX = 1.0      # 左摇杆 X → 侧移最大速度
JOY_WZ_MAX = 1.0      # 右摇杆 X → 转向最大速度
JOY_DEADBAND = 0.1    # 死区
JOY_TIMEOUT_SEC = 0.5 # 超时归零

# 手柄轴映射 (Xbox 标准布局)
# inputs 库 event.code 直接是 'ABS_X', 'ABS_Y', 'ABS_RX' 等字符串

# 手柄按钮
BTN_A = 0; BTN_B = 1; BTN_X = 2; BTN_Y = 3
BTN_LB = 4; BTN_RB = 5; BTN_BACK = 6; BTN_START = 7

# 全局状态
_gamepad_vx = 0.0
_gamepad_vy = 0.0
_gamepad_wz = 0.0
_gamepad_time = 0.0
_height_cmd = 0.25
_reset_flag = False
_print_action_flag = False
_start_rl_flag = False
_stop_flag = False

# 摇杆原始值 (用于平滑更新)
_raw_lx = 0.0
_raw_ly = 0.0
_raw_rx = 0.0


def _apply_deadband(v):
    return 0.0 if abs(v) < JOY_DEADBAND else v


def _detect_gamepads():
    """检测并列出已连接的手柄设备"""
    try:
        devices = inputs.devices.gamepads
    except Exception as e:
        print(f"[GAMEPAD] 检测设备失败: {e}")
        return []

    if not devices:
        print("[GAMEPAD] 未检测到任何手柄设备!")
        print("  请检查:")
        print("    1. 手柄是否已连接 (USB / 蓝牙)")
        print("    2. ls /dev/input/js* 确认设备节点")
        print("    3. sudo jstest /dev/input/js0 测试手柄")
        print("    4. 当前用户是否在 input 组: groups")
        return []

    print(f"[GAMEPAD] 检测到 {len(devices)} 个手柄设备:")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev.name}")
    return devices


# 启动时检测
_gp_devices = _detect_gamepads()
if _gp_devices:
    print(f"[GAMEPAD] 使用设备: {_gp_devices[0].name}")
else:
    print("[GAMEPAD] 警告: 没有手柄，将在等待状态运行...")
print()


def _gamepad_thread():
    """后台线程: 持续读取手柄事件"""
    global _gamepad_vx, _gamepad_vy, _gamepad_wz, _gamepad_time
    global _height_cmd, _reset_flag, _print_action_flag, _start_rl_flag, _stop_flag
    global _raw_lx, _raw_ly, _raw_rx

    _gamepad_time = time.time()
    _event_count = 0
    _last_warn = time.time()

    print("[GAMEPAD] 读取线程已启动, 等待手柄输入...")

    while not _stop_flag:
        try:
            events = inputs.get_gamepad()
        except inputs.UnpluggedError:
            if time.time() - _last_warn > 5.0:
                print("[GAMEPAD] 设备未连接, 等待中... (连接后自动恢复)")
                _last_warn = time.time()
            time.sleep(0.5)
            continue
        except Exception as e:
            if time.time() - _last_warn > 5.0:
                print(f"[GAMEPAD] 读取异常: {e}")
                _last_warn = time.time()
            time.sleep(0.1)
            continue

        for event in events:
            _event_count += 1
            if _event_count == 1:
                print(f"[GAMEPAD] 已收到第一个事件! type={event.ev_type} code={event.code} state={event.state}")

            if event.ev_type == 'Absolute':
                # Xbox 手柄摇杆范围: -32768 ~ 32767，中心为 0
                val = event.state / 32768.0 if event.state is not None else 0.0

                if event.code == 'ABS_X':
                    _raw_lx = val
                elif event.code == 'ABS_Y':
                    _raw_ly = val
                elif event.code == 'ABS_RX':
                    _raw_rx = val

                # 映射：与 test_11 JoyCallback 一致
                # 左摇杆 Y 前推为负 → 取反使前推=正速度
                vx = -_apply_deadband(_raw_ly) * JOY_VX_MAX
                vy = -_apply_deadband(_raw_lx) * JOY_VY_MAX
                wz = _apply_deadband(_raw_rx) * JOY_WZ_MAX

                _gamepad_vx = vx
                _gamepad_vy = vy
                _gamepad_wz = wz
                _gamepad_time = time.time()

            elif event.ev_type == 'Key':
                if event.code == 'BTN_SOUTH' and event.state == 1:     # A
                    _start_rl_flag = True
                elif event.code == 'BTN_EAST' and event.state == 1:    # B
                    _reset_flag = True
                elif event.code == 'BTN_WEST' and event.state == 1:    # X
                    pass  # 紧急停止可扩展
                elif event.code == 'BTN_NORTH' and event.state == 1:   # Y
                    _print_action_flag = not _print_action_flag
                elif event.code == 'BTN_RL' and event.state == 1:      # RB
                    _height_cmd = min(0.35, _height_cmd + 0.02)
                elif event.code == 'BTN_TL' and event.state == 1:      # LB
                    _height_cmd = max(0.20, _height_cmd - 0.02)


# 启动手柄线程
_gp_thread = threading.Thread(target=_gamepad_thread, daemon=True)
_gp_thread.start()


def get_commands():
    """返回当前手柄速度指令，超时自动归零"""
    if time.time() - _gamepad_time > JOY_TIMEOUT_SEC:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([_gamepad_vx, _gamepad_vy, _gamepad_wz], dtype=np.float32)


def build_single_obs(quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                     last_action_isaac, default_angles_isaac,
                     cmd, cmd_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale,
                     clip_obs, height_cmd=0.25):
    """
    构建46维单步观测 — 所有关节量使用 IsaacGym dof 顺序: FL, FR, RL, RR
    第46维: 高度指令 (height_cmd - 0.25) / 0.1
    """
    obs = np.zeros(46, dtype=np.float32)
    obs[0:3] = cmd * cmd_scale[:3]

    omega_body = quat_rotate_inverse(quat_xyzw, omega)
    obs[3:6] = omega_body.astype(np.float32) * ang_vel_scale

    gravity_world = np.array([0., 0., -1.], dtype=np.float64)
    proj_gravity = quat_rotate_inverse(quat_xyzw, gravity_world)
    obs[6:9] = proj_gravity.astype(np.float32)

    obs[9:21] = ((joint_q_isaac - default_angles_isaac) * dof_pos_scale).astype(np.float32)
    obs[21:33] = (joint_dq_isaac * dof_vel_scale).astype(np.float32)
    obs[33:45] = last_action_isaac
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--no-policy", action="store_true")
    args = parser.parse_args()

    base = "/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym"
    config_path = f"{base}/mujoco/dog/config/{args.config_file}"
    policy_path = f"/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/dog_rough/model_4500.onnx"
    xml_path    = f"/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/resources/robots/dog/xml/dog_terrain.xml"

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
    clip_obs       = config.get("clip_obs", 100.0)

    # 力矩限制: hip/thigh=23.7, calf=35.55 (MuJoCo 顺序)
    tau_limits_mujoco = np.array([
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
    ])

    NUM_ONE_STEP_OBS = 46
    HISTORY_LEN      = 6
    NUM_ACTIONS      = 12

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = simulation_dt
    mj_data = mujoco.MjData(mj_model)

    reset_robot(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)

    if not args.no_policy:
        policy = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
        input_name  = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name

    obs_history      = ObsHistoryBuffer(HISTORY_LEN, NUM_ONE_STEP_OBS)
    target_q_mujoco  = DEFAULT_ANGLES_MUJOCO.copy()
    action_isaac     = np.zeros(NUM_ACTIONS, dtype=np.float64)
    last_action_isaac = np.zeros(NUM_ACTIONS, dtype=np.float32)
    labels = ["FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf",
              "RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf"]
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
            ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs,
            height_cmd=0.25
        )
        obs_history.push(obs)

    print(f"[INFO] 预热完成\n")
    print(f"  ★ Xbox 手柄控制:")
    print(f"    左摇杆: 前后(vx) 左右(vy)   右摇杆 X: 转向(wz)")
    print(f"    A: 重置  B: ---  Y: 打印action  X: ---")
    print(f"    LB: 蹲下↓  RB: 站起↑")
    print(f"    超时 {JOY_TIMEOUT_SEC}s 无输入 → 自动归零\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # 设置第三人称跟踪相机
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

            # 获取当前指令 (按住运动，松开停止)
            cmd = get_commands()

            if _reset_flag:
                reset_robot(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)
                obs_history.reset()
                last_action_isaac[:] = 0; action_isaac[:] = 0; count = 0
                _reset_flag = False

            # 读取 MuJoCo 状态
            joint_q_mujoco  = mj_data.qpos[7:19].astype(np.float64)
            joint_dq_mujoco = mj_data.qvel[6:18].astype(np.float64)

            # ★ 映射到 IsaacGym 顺序 (送给策略)
            joint_q_isaac  = joint_q_mujoco[MUJOCO_TO_ISAAC]
            joint_dq_isaac = joint_dq_mujoco[MUJOCO_TO_ISAAC]

            quat_wxyz = mj_data.qpos[3:7]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            omega = mj_data.qvel[3:6].astype(np.float64)

            if count % control_decimation == 0:
                if not args.no_policy:
                    single_obs = build_single_obs(
                        quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                        last_action_isaac, DEFAULT_ANGLES_ISAAC,
                        cmd, cmd_scale,
                        ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs,
                        height_cmd=_height_cmd
                    )
                    obs_history.push(single_obs)
                    obs_input = obs_history.get()

                    action_raw = policy.run([output_name], {input_name: obs_input})[0][0]
                    action_isaac[:] = np.clip(action_raw, -10.0, 10.0)
                    last_action_isaac = action_isaac.astype(np.float32)

                    # ★ 目标角: IsaacGym顺序 → MuJoCo顺序
                    target_q_isaac = action_isaac * action_scale + DEFAULT_ANGLES_ISAAC
                    target_q_mujoco = target_q_isaac[ISAAC_TO_MUJOCO]
                else:
                    target_q_mujoco = DEFAULT_ANGLES_MUJOCO.copy()

            # PD 控制 (MuJoCo 顺序)
            tau = pd_control(target_q_mujoco, joint_q_mujoco, kps,
                           np.zeros(NUM_ACTIONS), joint_dq_mujoco, kds)
            tau = np.clip(tau, -tau_limits_mujoco, tau_limits_mujoco)
            mj_data.ctrl[:NUM_ACTIONS] = tau

            mujoco.mj_step(mj_model, mj_data)
            count += 1

            if count % (control_decimation * 50) == 0:
                grav = quat_rotate_inverse(quat_xyzw, np.array([0., 0., -1.]))
                h = mj_data.qpos[2]
                print(f"[{time.time()-start:.1f}s] Step {count} H={h:.3f}(cmd={_height_cmd:.2f}) "
                      f"vx={mj_data.qvel[0]:.2f} vy={mj_data.qvel[1]:.2f} wz={mj_data.qvel[5]:.2f} "
                      f"grav_z={grav[2]:.3f} "
                      f"act=[{action_isaac.min():.2f},{action_isaac.max():.2f}]")

            viewer.sync()
            elapsed = time.time() - step_start
            if simulation_dt - elapsed > 0:
                time.sleep(simulation_dt - elapsed)

    print("\n[INFO] 仿真结束")
