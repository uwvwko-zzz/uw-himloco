"""
Dog 倒地恢复测试 — MuJoCo Sim2Sim (IsaacGym → MuJoCo)

基于 dog_recovery ONNX 策略，测试机器人从倒地姿态恢复站立。
启动即随机倒地，观察策略能否自行站起。

关节顺序:
  MuJoCo XML 顺序 (即 URDF 顺序): FL(0-2), FR(3-5), RR(6-8), RL(9-11)
  IsaacGym dof 顺序 (内部重排):    FL(0-2), FR(3-5), RL(6-8), RR(9-11)

操作:
  键盘:
    T:      重置到站立姿态
    F:      重置到随机倒地姿态（测试恢复）
    B:      重置到背部朝下（四脚朝天，翻 180°）
    C:      重置到侧翻 90°（侧躺）
    空格:   随机推一下（施加冲击速度）
    R:      降低目标高度
    G:      升高目标高度
    Z:      重置目标高度到 0.25m
  鼠标 (MuJoCo viewer 内置):
    双击身体部位:     选中
    Ctrl+左键拖拽:    施加力（拖拽狗子）
    Ctrl+右键拖拽:    施加力矩
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
MUJOCO_TO_ISAAC = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]
ISAAC_TO_MUJOCO = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]

# 默认关节角度 — IsaacGym dof 顺序: FL, FR, RL, RR
DEFAULT_ANGLES_ISAAC = np.array([
    -0.1, -0.8, -1.5,    # FL (Isaac dof 0-2)
     0.1,  0.8,  1.5,    # FR (Isaac dof 3-5)
     0.1, -1.0, -1.5,    # RL (Isaac dof 6-8)
    -0.1,  1.0,  1.5,    # RR (Isaac dof 9-11)
], dtype=np.float64)

DEFAULT_ANGLES_MUJOCO = DEFAULT_ANGLES_ISAAC[ISAAC_TO_MUJOCO]

# 力矩限制 (来自 dog.xml actuator ctrlrange)
TAU_LIMIT_HIP_THIGH = 23.7   # hip, thigh [Nm]
TAU_LIMIT_CALF = 35.55       # calf [Nm]


# ============================================================
#  数学工具
# ============================================================
def quat_rotate_inverse(q_xyzw, v):
    """四元数逆旋转 (q 为 [x,y,z,w])"""
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def euler_to_quat_wxyz(roll, pitch, yaw):
    """Euler (XYZ intrinsic) → quaternion [w, x, y, z] (MuJoCo 顺序)"""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float64)


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
#  观测构建 — 与 IsaacGym compute_observations 完全一致
# ============================================================
def build_single_obs(quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                     last_action_isaac, default_angles_isaac,
                     cmd, cmd_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale,
                     clip_obs, height_cmd=0.25):
    """
    构建 46 维单步观测 — 关节量使用 IsaacGym dof 顺序: FL, FR, RL, RR
    恢复任务 cmd = [0, 0, 0]（速度命令为零），第 46 维 = (height_cmd - 0.25) / 0.1
    """
    obs = np.zeros(46, dtype=np.float32)
    # [0:3]  速度命令 × scale（恢复任务恒为 0）
    obs[0:3] = cmd * cmd_scale[:3]
    # [3:6]  身体角速度 × scale
    omega_body = quat_rotate_inverse(quat_xyzw, omega)
    obs[3:6] = omega_body.astype(np.float32) * ang_vel_scale
    # [6:9]  重力投影
    gravity_world = np.array([0., 0., -1.], dtype=np.float64)
    proj_gravity = quat_rotate_inverse(quat_xyzw, gravity_world)
    obs[6:9] = proj_gravity.astype(np.float32)
    # [9:21]  关节位置偏差 × scale
    obs[9:21] = ((joint_q_isaac - default_angles_isaac) * dof_pos_scale).astype(np.float32)
    # [21:33] 关节速度 × scale
    obs[21:33] = (joint_dq_isaac * dof_vel_scale).astype(np.float32)
    # [33:45] 上一步动作
    obs[33:45] = last_action_isaac
    # [45]    高度指令
    obs[45] = np.float32((height_cmd - 0.25) / 0.1)
    obs = np.clip(obs, -clip_obs, clip_obs)
    return obs


# ============================================================
#  重置 / 推倒
# ============================================================
def reset_to_pose(model, data, default_angles_mujoco,
                  roll=0.0, pitch=0.0, yaw=0.0, height=0.42):
    """重置到指定姿态（roll/pitch/yaw + z 高度 + 默认关节角）"""
    mujoco.mj_resetData(model, data)
    data.qpos[2] = height
    data.qpos[3:7] = euler_to_quat_wxyz(roll, pitch, yaw)
    data.qpos[7:19] = default_angles_mujoco
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def reset_standing(model, data, default_angles_mujoco):
    """重置到标准站立姿态"""
    reset_to_pose(model, data, default_angles_mujoco,
                  roll=0.0, pitch=0.0, yaw=0.0, height=0.42)


def reset_fallen(model, data, default_angles_mujoco):
    """重置到随机倒地姿态（模拟训练初始化）"""
    # 随机 roll/pitch 倒地（全范围 π），yaw 小范围
    roll = np.random.uniform(-np.pi, np.pi)
    pitch = np.random.uniform(-np.pi, np.pi)
    yaw = np.random.uniform(-0.5, 0.5)
    reset_to_pose(model, data, default_angles_mujoco,
                  roll=roll, pitch=pitch, yaw=yaw,
                  height=np.random.uniform(0.10, 0.20))


def reset_back_down(model, data, default_angles_mujoco):
    """重置到背部朝下（四脚朝天，翻 180°，grav_z≈+1）"""
    reset_to_pose(model, data, default_angles_mujoco,
                  roll=np.pi, pitch=0.0, yaw=0.0, height=0.18)


def reset_side(model, data, default_angles_mujoco):
    """重置到侧翻 90°（侧躺，grav_z≈0）"""
    reset_to_pose(model, data, default_angles_mujoco,
                  roll=np.pi / 2, pitch=0.0, yaw=0.0, height=0.16)


def apply_push(data):
    """施加随机冲击速度（模拟被踹一下）"""
    data.qvel[0] += np.random.uniform(-2.0, 2.0)   # vx
    data.qvel[1] += np.random.uniform(-2.0, 2.0)   # vy
    data.qvel[3] += np.random.uniform(-5.0, 5.0)   # roll rate
    data.qvel[4] += np.random.uniform(-5.0, 5.0)   # pitch rate


# ============================================================
#  键盘控制
# ============================================================
# GLFW keycodes
GLFW_KEY_T = 84;   GLFW_KEY_F = 70
GLFW_KEY_R = 82;   GLFW_KEY_G = 71
GLFW_KEY_B = 66;   GLFW_KEY_C = 67
GLFW_KEY_Z = 90;   GLFW_KEY_SPACE = 32

# 全局状态
height_cmd = 0.25
action_request = None   # 'standing' | 'fallen' | 'push' | 'back_down' | 'side' | None


def key_callback(keycode):
    global height_cmd, action_request
    if keycode == GLFW_KEY_T:
        action_request = 'standing'
    elif keycode == GLFW_KEY_F:
        action_request = 'fallen'
    elif keycode == GLFW_KEY_B:
        action_request = 'back_down'
    elif keycode == GLFW_KEY_C:
        action_request = 'side'
    elif keycode == GLFW_KEY_SPACE:
        action_request = 'push'
    elif keycode == GLFW_KEY_R:
        height_cmd = max(0.20, height_cmd - 0.02)
    elif keycode == GLFW_KEY_G:
        height_cmd = min(0.35, height_cmd + 0.02)
    elif keycode == GLFW_KEY_Z:
        height_cmd = 0.25


# ============================================================
#  主程序
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dog 倒地恢复测试")
    parser.add_argument("--no-policy", action="store_true", help="不加载策略（纯物理）")
    args = parser.parse_args()

    # --- 路径 ---
    base = "/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_hop"
    config_path = f"/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_hop/mujoco/dog_r/config/dog_r.yaml"
    policy_path = f"/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_hop/logs/dog_recovery/model_4500.onnx"
    xml_path     = f"/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_hop/resources/robots/dog/xml/dog.xml"

    # --- 加载配置 ---
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

    tau_limits_mujoco = np.array([
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
        TAU_LIMIT_HIP_THIGH, TAU_LIMIT_HIP_THIGH, TAU_LIMIT_CALF,
    ])

    NUM_ONE_STEP_OBS = 46
    HISTORY_LEN      = 6
    NUM_ACTIONS      = 12

    # --- 打印配置信息 ---
    print(f"\n{'='*60}")
    print(f"[Dog 倒地恢复测试]")
    print(f"{'='*60}")
    print(f"  action_scale: {action_scale}")
    print(f"  kp={kps[0]}, kd={kds[0]}, decimation={control_decimation}")
    print(f"  tau_limits: hip/thigh={TAU_LIMIT_HIP_THIGH}, calf={TAU_LIMIT_CALF}")
    print(f"  obs: {NUM_ONE_STEP_OBS} x {HISTORY_LEN} = {NUM_ONE_STEP_OBS * HISTORY_LEN}")
    print(f"{'='*60}")

    # --- 加载 MuJoCo 模型 ---
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = simulation_dt
    mj_data = mujoco.MjData(mj_model)

    # --- 加载策略 ---
    if not args.no_policy:
        policy = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
        input_name  = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name
        print(f"[ONNX] {policy.get_inputs()[0].shape} -> {policy.get_outputs()[0].shape}")
    else:
        print("[INFO] --no-policy 模式：不加载策略，关节保持默认角度")

    # --- 启动即倒地 ---
    reset_fallen(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)
    print(f"[INFO] 初始倒地姿态: z={mj_data.qpos[2]:.3f}, "
          f"quat={mj_data.qpos[3:7]}")

    # --- 初始化历史缓冲区 ---
    obs_history      = ObsHistoryBuffer(HISTORY_LEN, NUM_ONE_STEP_OBS)
    target_q_mujoco  = DEFAULT_ANGLES_MUJOCO.copy()
    action_isaac     = np.zeros(NUM_ACTIONS, dtype=np.float64)
    last_action_isaac = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # 预热历史缓冲
    cmd_zero = np.zeros(3, dtype=np.float32)
    for _ in range(HISTORY_LEN):
        quat_wxyz = mj_data.qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        omega = mj_data.qvel[3:6].astype(np.float64)
        joint_q_isaac = mj_data.qpos[7:19].astype(np.float64)[MUJOCO_TO_ISAAC]
        joint_dq_isaac = mj_data.qvel[6:18].astype(np.float64)[MUJOCO_TO_ISAAC]
        obs = build_single_obs(
            quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
            last_action_isaac, DEFAULT_ANGLES_ISAAC,
            cmd_zero, cmd_scale,
            ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs,
            height_cmd=height_cmd
        )
        obs_history.push(obs)

    # --- 帮助信息 ---
    print(f"\n{'='*60}")
    print(f"  键盘控制:")
    print(f"    T:      重置到站立姿态")
    print(f"    F:      重置到随机倒地姿态")
    print(f"    B:      重置到背部朝下(四脚朝天)")
    print(f"    C:      重置到侧翻90°(侧躺)")
    print(f"    空格:   随机推一下")
    print(f"    R/G:    降低/升高目标高度 (当前 {height_cmd:.2f}m)")
    print(f"    Z:      重置目标高度到 0.25m")
    print(f"  鼠标 (viewer 内置):")
    print(f"    双击身体选中, Ctrl+左键拖拽 = 施加力（拖拽狗子）")
    print(f"{'='*60}\n")

    # --- 主循环 ---
    count = 0
    trunk_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        # 第三人称跟踪相机
        if trunk_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = trunk_id
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 135

        start = time.time()
        next_step_time = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            # 处理键盘请求
            if action_request is not None:
                if action_request == 'standing':
                    reset_standing(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)
                    print("[RESET] -> 站立姿态")
                elif action_request == 'fallen':
                    reset_fallen(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)
                    q = mj_data.qpos[3:7]
                    quat_xyzw = np.array([q[1], q[2], q[3], q[0]])
                    grav = quat_rotate_inverse(quat_xyzw, np.array([0., 0., -1.]))
                    print(f"[RESET] -> 随机倒地  z={mj_data.qpos[2]:.3f}  grav_z={grav[2]:+.3f}")
                elif action_request == 'back_down':
                    reset_back_down(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)
                    print(f"[RESET] -> 背部朝下(四脚朝天)  z={mj_data.qpos[2]:.3f}")
                elif action_request == 'side':
                    reset_side(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)
                    print(f"[RESET] -> 侧翻90°  z={mj_data.qpos[2]:.3f}")
                elif action_request == 'push':
                    apply_push(mj_data)
                    print("[PUSH] 施加随机冲击")
                # 重置历史和动作
                obs_history.reset()
                last_action_isaac[:] = 0
                action_isaac[:] = 0
                action_request = None

            # 读取 MuJoCo 状态
            joint_q_mujoco  = mj_data.qpos[7:19].astype(np.float64)
            joint_dq_mujoco = mj_data.qvel[6:18].astype(np.float64)

            # 映射到 IsaacGym 顺序
            joint_q_isaac  = joint_q_mujoco[MUJOCO_TO_ISAAC]
            joint_dq_isaac = joint_dq_mujoco[MUJOCO_TO_ISAAC]

            quat_wxyz = mj_data.qpos[3:7]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            omega = mj_data.qvel[3:6].astype(np.float64)

            # 策略推理（每 control_decimation 步）
            if count % control_decimation == 0:
                if not args.no_policy:
                    single_obs = build_single_obs(
                        quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                        last_action_isaac, DEFAULT_ANGLES_ISAAC,
                        cmd_zero, cmd_scale,
                        ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs,
                        height_cmd=height_cmd
                    )
                    obs_history.push(single_obs)
                    obs_input = obs_history.get()

                    action_raw = policy.run([output_name], {input_name: obs_input})[0][0]
                    action_isaac[:] = np.clip(action_raw, -10.0, 10.0)
                    last_action_isaac = action_isaac.astype(np.float32)

                    # 目标角: IsaacGym 顺序 -> MuJoCo 顺序
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

            # 状态打印 + viewer 同步
            if count % (control_decimation * 50) == 0:
                grav = quat_rotate_inverse(quat_xyzw, np.array([0., 0., -1.]))
                h = mj_data.qpos[2]
                upright_str = "正立" if grav[2] < -0.9 else ("侧躺" if grav[2] > -0.3 else "恢复中")
                print(f"[{time.time()-start:.1f}s] H={h:.3f}(cmd={height_cmd:.2f}) "
                      f"grav_z={grav[2]:+.3f} [{upright_str}] "
                      f"act=[{action_isaac.min():.2f},{action_isaac.max():.2f}]")

            if count % control_decimation == 0:
                viewer.sync()

            # 墙钟对齐
            next_step_time += simulation_dt
            _sleep = next_step_time - time.time()
            if _sleep > 0:
                time.sleep(_sleep)

    print("\n[INFO] 仿真结束")
