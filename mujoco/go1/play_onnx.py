"""
HIMLoco ONNX + MuJoCo 部署脚本 (修正版)

关键修正:
  1. obs 拼接顺序严格对齐 compute_observations()
  2. MuJoCo关节顺序(FR,FL,RR,RL) ↔ Isaac Gym关节顺序(FL,FR,RL,RR) 映射
  3. commands 只用 3D (不含 heading)
  4. 270D = 45 × 6 历史队列, 最新帧在前
"""

import time
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml
from pynput import keyboard  # 新增: 用于键盘按下/松开检测


# ============================================================
#  关节顺序映射
# ============================================================
# MuJoCo XML body顺序: FR, FL, RR, RL
# MuJoCo qpos[7:19] / qvel[6:18] 顺序:
#   [0]  FR_hip    [1]  FR_thigh  [2]  FR_calf
#   [3]  FL_hip    [4]  FL_thigh  [5]  FL_calf
#   [6]  RR_hip    [7]  RR_thigh  [8]  RR_calf
#   [9]  RL_hip    [10] RL_thigh  [11] RL_calf
#
# Isaac Gym URDF顺序: FL, FR, RL, RR
# Isaac dof_pos 顺序:
#   [0]  FL_hip    [1]  FL_thigh  [2]  FL_calf
#   [3]  FR_hip    [4]  FR_thigh  [5]  FR_calf
#   [6]  RL_hip    [7]  RL_thigh  [8]  RL_calf
#   [9]  RR_hip    [10] RR_thigh  [11] RR_calf

# MuJoCo索引 -> Isaac索引 (读取qpos后重排为Isaac顺序送给策略)
MUJOCO_TO_ISAAC = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# Isaac索引 -> MuJoCo索引 (策略输出action后重排为MuJoCo顺序写入ctrl)
ISAAC_TO_MUJOCO = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Isaac Gym 顺序的 default_angles:
# FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
# RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
DEFAULT_ANGLES_ISAAC = np.array([
    0.1,  0.8, -1.5,   # FL
   -0.1,  0.8, -1.5,   # FR
    0.1,  1.0, -1.5,   # RL
   -0.1,  1.0, -1.5,   # RR
], dtype=np.float64)

# MuJoCo 顺序的 default_angles (用于初始化 qpos 和 PD 控制):
# FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
# RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
DEFAULT_ANGLES_MUJOCO = DEFAULT_ANGLES_ISAAC[ISAAC_TO_MUJOCO]


# ============================================================
#  工具函数
# ============================================================

def quat_rotate_inverse(q, v):
    """将世界系向量 v 旋转到体坐标系。q: [x,y,z,w], v: [3]"""
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# ============================================================
#  观测历史管理器
# ============================================================

class ObsHistoryBuffer:
    """维护固定长度的观测历史队列, 最新帧在前"""
    def __init__(self, history_len: int, single_obs_dim: int):
        self.history_len = history_len
        self.single_obs_dim = single_obs_dim
        self.total_dim = history_len * single_obs_dim
        self.buffer = np.zeros(self.total_dim, dtype=np.float32)

    def push(self, new_obs: np.ndarray):
        self.buffer[self.single_obs_dim:] = self.buffer[:-self.single_obs_dim].copy()
        self.buffer[:self.single_obs_dim] = new_obs

    def get(self) -> np.ndarray:
        return self.buffer.reshape(1, -1).copy()

    def reset(self):
        self.buffer[:] = 0.0


# ============================================================
#  键盘控制 (新增: 使用 pynput 监听按下/松开)
# ============================================================

vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
reset_flag = False  # 新增: 重置标志

def on_press(key):
    global vx_cmd, vy_cmd, wz_cmd, reset_flag
    try:
        if key.char == 'w' or key == keyboard.Key.up:
            vx_cmd = 1.0
        elif key.char == 's' or key == keyboard.Key.down:
            vx_cmd = -1.0
        elif key.char == 'a' or key == keyboard.Key.left:
            vy_cmd = 1.0
        elif key.char == 'd' or key == keyboard.Key.right:
            vy_cmd = -1.0
        elif key.char == 'q':
            wz_cmd = 1.0
        elif key.char == 'e':
            wz_cmd = -1.0
        elif key.char == 'r':
            reset_flag = True  # 按 R 触发重置
        elif key == keyboard.Key.space:
            vx_cmd = vy_cmd = wz_cmd = 0.0  # 空格停止
    except AttributeError:
        pass

def on_release(key):
    global vx_cmd, vy_cmd, wz_cmd
    try:
        if key.char in 'ws' or key in [keyboard.Key.up, keyboard.Key.down]:
            vx_cmd = 0.0
        elif key.char in 'ad' or key in [keyboard.Key.left, keyboard.Key.right]:
            vy_cmd = 0.0
        elif key.char in 'qe':
            wz_cmd = 0.0
    except AttributeError:
        pass


# ============================================================
#  构建单步观测 (45D) — 严格对齐 compute_observations()
# ============================================================

def build_single_obs(
    quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
    last_action_isaac, default_angles_isaac,
    cmd, cmd_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale,
    clip_obs
):
    """
    严格对齐 HIMLoco compute_observations():
      [0:3]   commands[:, :3] * commands_scale   (vx, vy, wz) × (2, 2, 0.25)
      [3:6]   base_ang_vel * ang_vel_scale       (体坐标系)
      [6:9]   projected_gravity                  (体坐标系)
      [9:21]  (dof_pos - default) * dof_pos_scale
      [21:33] dof_vel * dof_vel_scale
      [33:45] actions                            (当前动作, 即上一步输出)

    所有关节相关量均为 Isaac Gym 顺序 (FL, FR, RL, RR)
    """
    obs = np.zeros(45, dtype=np.float32)

    # [0:3] commands * scale (只取前3维)
    obs[0:3] = cmd * cmd_scale[:3]

    # [3:6] body-frame angular velocity * scale
    omega_body = omega  # MuJoCo free-joint angular qvel is already in the local body frame
    obs[3:6] = omega_body.astype(np.float32) * ang_vel_scale

    # [6:9] projected gravity (body frame)
    gravity_world = np.array([0., 0., -1.], dtype=np.float64)
    proj_gravity = quat_rotate_inverse(quat_xyzw, gravity_world)
    obs[6:9] = proj_gravity.astype(np.float32)

    # [9:21] (dof_pos - default) * scale (Isaac顺序)
    obs[9:21] = ((joint_q_isaac - default_angles_isaac) * dof_pos_scale).astype(np.float32)

    # [21:33] dof_vel * scale (Isaac顺序)
    obs[21:33] = (joint_dq_isaac * dof_vel_scale).astype(np.float32)

    # [33:45] actions (Isaac顺序, 当前动作=上一步策略输出)
    obs[33:45] = last_action_isaac

    obs = np.clip(obs, -clip_obs, clip_obs)
    return obs


# ============================================================
#  重置机器人状态
# ============================================================

def reset_robot(model, data, default_angles_mujoco):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.35
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = default_angles_mujoco
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


# ============================================================
#  主程序
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HIMLoco ONNX + MuJoCo Deploy")
    parser.add_argument("config_file", type=str, help="YAML 配置文件名")
    parser.add_argument("--no-policy", action="store_true")
    args = parser.parse_args()

    # ==================== 路径 ====================
    config_path = f"/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/mujoco/go1/config/{args.config_file}"
    policy_path = "/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/rough_go1/good/model_10000.onnx"
    xml_path    = "/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/resources/robots/go1/xml/go1.xml"

    # ==================== 加载配置 ====================
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
    tau_limit      = config.get("tau_limit", 33.5)
    clip_obs       = config.get("clip_obs", 100.0)

    NUM_ONE_STEP_OBS = 45
    HISTORY_LEN      = 6
    NUM_ACTIONS      = 12
    TOTAL_OBS_DIM    = NUM_ONE_STEP_OBS * HISTORY_LEN  # 270

    # ==================== 打印关键配置 ====================
    print(f"\n{'='*60}")
    print(f"[CONFIG] HIMLoco MuJoCo 部署")
    print(f"{'='*60}")
    print(f"  ONNX 输入: {TOTAL_OBS_DIM}D = {NUM_ONE_STEP_OBS} × {HISTORY_LEN}")
    print(f"  kps={kps[0]}, kds={kds[0]}, action_scale={action_scale}")
    print(f"  ang_vel_scale={ang_vel_scale}, dof_pos_scale={dof_pos_scale}, dof_vel_scale={dof_vel_scale}")
    print(f"  cmd_scale={cmd_scale[:3]} (只用前3维)")
    print(f"  tau_limit={tau_limit}, clip_obs={clip_obs}")
    print(f"\n  Default angles (Isaac顺序 FL,FR,RL,RR):")
    print(f"    {DEFAULT_ANGLES_ISAAC}")
    print(f"  Default angles (MuJoCo顺序 FR,FL,RR,RL):")
    print(f"    {DEFAULT_ANGLES_MUJOCO}")
    print(f"\n  关节映射 MuJoCo→Isaac: {MUJOCO_TO_ISAAC}")
    print(f"  关节映射 Isaac→MuJoCo: {ISAAC_TO_MUJOCO}")
    print(f"{'='*60}")

    # ==================== MuJoCo 初始化 ====================
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = simulation_dt
    mj_data = mujoco.MjData(mj_model)

    # 打印 MuJoCo 关节名以验证顺序
    print(f"\n[DEBUG] MuJoCo 关节名顺序:")
    for i in range(mj_model.njnt):
        jnt_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  joint[{i}]: {jnt_name}")

    print(f"\n[DEBUG] MuJoCo 执行器名顺序:")
    for i in range(mj_model.nu):
        act_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  actuator[{i}]: {act_name}")

    reset_robot(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)

    # 验证初始化后的关节位置
    print(f"\n[DEBUG] 初始化后 qpos[7:19] (MuJoCo顺序):")
    print(f"  {mj_data.qpos[7:19]}")
    print(f"[DEBUG] 转为Isaac顺序:")
    print(f"  {mj_data.qpos[7:19][MUJOCO_TO_ISAAC]}")
    print(f"[DEBUG] 期望Isaac顺序:")
    print(f"  {DEFAULT_ANGLES_ISAAC}")

    # ==================== ONNX 加载 ====================
    if not args.no_policy:
        policy = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
        input_name  = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name
        input_shape = policy.get_inputs()[0].shape
        print(f"\n[INFO] ONNX 已加载")
        print(f"  输入: {input_name}, shape={input_shape}")
        print(f"  输出: {output_name}, shape={policy.get_outputs()[0].shape}")

    # ==================== 变量初始化 ====================
    obs_history = ObsHistoryBuffer(HISTORY_LEN, NUM_ONE_STEP_OBS)
    target_q_mujoco = DEFAULT_ANGLES_MUJOCO.copy()
    action_isaac    = np.zeros(NUM_ACTIONS, dtype=np.float64)   # Isaac顺序
    last_action_isaac = np.zeros(NUM_ACTIONS, dtype=np.float32) # Isaac顺序
    count = 0

    # ==================== 预热历史缓冲 ====================
    print("\n[INFO] 预热历史缓冲区...")
    for i in range(HISTORY_LEN):
        quat_wxyz = mj_data.qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        omega     = mj_data.qvel[3:6].astype(np.float64)

        # MuJoCo顺序 → Isaac顺序
        joint_q_isaac  = mj_data.qpos[7:19].astype(np.float64)[MUJOCO_TO_ISAAC]
        joint_dq_isaac = mj_data.qvel[6:18].astype(np.float64)[MUJOCO_TO_ISAAC]

        init_obs = build_single_obs(
            quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
            last_action_isaac, DEFAULT_ANGLES_ISAAC,
            np.zeros(3, dtype=np.float32), cmd_scale,
            ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs
        )
        obs_history.push(init_obs)

        if i == 0:
            print(f"  [DEBUG] 预热帧0 obs (45D):")
            print(f"    cmd_scaled   [0:3]  = {init_obs[0:3]}")
            print(f"    angvel_scaled[3:6]  = {init_obs[3:6]}")
            print(f"    proj_gravity [6:9]  = {init_obs[6:9]}")
            print(f"    dof_pos_delta[9:21] = {init_obs[9:21]}")
            print(f"    dof_vel      [21:33]= {init_obs[21:33]}")
            print(f"    last_action  [33:45]= {init_obs[33:45]}")

    print("[INFO] 预热完成")
    print(f"  obs_history norm = {np.linalg.norm(obs_history.buffer):.4f}")

    # ==================== 控制说明 ====================
    print(f"\n{'='*50}")
    print("        HIMLoco MuJoCo 键盘控制")
    print('='*50)
    print("  ↑/W: 前进   ↓/S: 后退   ←/A: 左移   →/D: 右移")
    print("  Q: 左转     E: 右转     空格: 停止   R: 重置")
    print('='*50)
    print(f"\n[INFO] 开始仿真...")

    # ==================== 启动键盘监听 ====================
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # ==================== 主循环 ====================
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:  # 移除 key_callback
        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # --- 更新 cmd (从全局变量) ---
            cmd = np.array([vx_cmd, vy_cmd, wz_cmd], dtype=np.float32)

            # --- 手动重置检查 ---
            if reset_flag:
                print("[CMD] 重置")
                reset_robot(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)
                obs_history.reset()  # 可选: 重置历史缓冲
                reset_flag = False

            # --- 读取状态 (MuJoCo顺序) ---
            joint_q_mujoco  = mj_data.qpos[7:19].astype(np.float64)
            joint_dq_mujoco = mj_data.qvel[6:18].astype(np.float64)

            # 转为 Isaac 顺序 (送给策略)
            joint_q_isaac  = joint_q_mujoco[MUJOCO_TO_ISAAC]
            joint_dq_isaac = joint_dq_mujoco[MUJOCO_TO_ISAAC]

            # 四元数: MuJoCo [w,x,y,z] -> Isaac [x,y,z,w]
            quat_wxyz = mj_data.qpos[3:7]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

            # 角速度 (世界系)
            omega = mj_data.qvel[3:6].astype(np.float64)

            # --- 策略推理 ---
            if count % control_decimation == 0:
                if not args.no_policy:
                    # 1) 构建单步观测 (Isaac顺序)
                    single_obs = build_single_obs(
                        quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                        last_action_isaac, DEFAULT_ANGLES_ISAAC,
                        cmd, cmd_scale,
                        ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs
                    )

                    # 2) 压入历史
                    obs_history.push(single_obs)

                    # 3) ONNX 推理 (输入270D, 输出12D, Isaac顺序)
                    obs_input = obs_history.get()
                    action_raw = policy.run(
                        [output_name], {input_name: obs_input}
                    )[0][0]  # (12,) Isaac顺序

                    action_isaac = np.clip(action_raw, -10.0, 10.0).astype(np.float64)
                    last_action_isaac = action_isaac.astype(np.float32)

                    # 4) 目标关节角 (Isaac顺序 → MuJoCo顺序)
                    target_q_isaac = action_isaac * action_scale + DEFAULT_ANGLES_ISAAC
                    target_q_mujoco = target_q_isaac[ISAAC_TO_MUJOCO]
                else:
                    target_q_mujoco = DEFAULT_ANGLES_MUJOCO.copy()

            # --- PD 控制 (MuJoCo顺序) ---
            target_dq = np.zeros(NUM_ACTIONS, dtype=np.float64)
            tau = pd_control(target_q_mujoco, joint_q_mujoco, kps, target_dq, joint_dq_mujoco, kds)
            tau = np.clip(tau, -tau_limit, tau_limit)
            mj_data.ctrl[:NUM_ACTIONS] = tau

            # --- 步进 ---
            mujoco.mj_step(mj_model, mj_data)
            count += 1

            # --- 调试输出 ---
            if count % (control_decimation * 50) == 0:
                grav = quat_rotate_inverse(quat_xyzw, np.array([0., 0., -1.]))
                h  = mj_data.qpos[2]
                vx = mj_data.qvel[0]
                vy = mj_data.qvel[1]
                wz = mj_data.qvel[5]

                print(f"\n{'─'*60}")
                print(f"[{time.time()-start:.1f}s] Step {count}")
                print(f"  Height: {h:.3f}  vx={vx:.2f}  vy={vy:.2f}  wz={wz:.2f}")
                print(f"  Cmd: vx={cmd[0]:.2f} vy={cmd[1]:.2f} yaw={cmd[2]:.2f}")
                print(f"  Gravity(body): [{grav[0]:.3f}, {grav[1]:.3f}, {grav[2]:.3f}]")
                print(f"  Quat(xyzw): [{quat_xyzw[0]:.4f}, {quat_xyzw[1]:.4f}, {quat_xyzw[2]:.4f}, {quat_xyzw[3]:.4f}]")
                print(f"  Joint Q (Isaac): {np.array2string(joint_q_isaac, precision=3, suppress_small=True)}")
                print(f"  Joint Q (MuJoCo): {np.array2string(joint_q_mujoco, precision=3, suppress_small=True)}")
                print(f"  Action (Isaac): {np.array2string(action_isaac, precision=3, suppress_small=True)}")
                print(f"  Target Q (MuJoCo): {np.array2string(target_q_mujoco, precision=3, suppress_small=True)}")
                print(f"  Tau: {np.array2string(tau, precision=2, suppress_small=True)}")
                print(f"  Obs norm: {np.linalg.norm(obs_history.buffer):.3f}")
                print(f"{'─'*60}")

            viewer.sync()
            elapsed = time.time() - step_start
            if simulation_dt - elapsed > 0:
                time.sleep(simulation_dt - elapsed)

    # ==================== 清理键盘监听 ====================
    listener.stop()

    print("\n[INFO] 仿真结束")