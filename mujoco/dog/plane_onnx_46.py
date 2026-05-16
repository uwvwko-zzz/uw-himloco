"""
Dog 46维观测版 — MuJoCo sim2sim

基于 plane_onnx_45.py 修改，仅在45维基础上末尾增加1维高度指令

46维观测结构 (与 Isaac Gym legged_robot.py compute_observations 一致):
  [0:3]   commands × scale       (vx, vy, wz)          ← 和45维完全一样
  [3:6]   base_ang_vel × scale                           ← 和45维完全一样
  [6:9]   projected_gravity                                 ← 和45维完全一样
  [9:21]  dof_pos - default_pos                            ← 和45维完全一样
  [21:33] dof_vel × scale                                   ← 和45维完全一样
  [33:45] last_actions                                      ← 和45维完全一样
  [45]    height_command = (height_cmd - 0.25) / 0.1      ← ★ 新增第46维

核心发现:
  Isaac Gym dof 顺序: FL, FR, RL, RR
  MuJoCo XML 顺序:    FL, FR, RR, RL
  后两条腿反了! 需要映射!
"""
import time
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml
from pynput import keyboard


# ============================================================
#  关节映射 — 和 Go1 同样的问题!
# ============================================================
# MuJoCo qpos[7:19] 顺序: FL(0-2), FR(3-5), RR(6-8), RL(9-11)
# Isaac Gym dof 顺序:      FL(0-2), FR(3-5), RL(6-8), RR(9-11)
#
# MuJoCo[i] -> Isaac[j]:
#   MuJoCo 0-5 (FL,FR) -> Isaac 0-5 (FL,FR)  不变
#   MuJoCo 6-8 (RR)    -> Isaac 9-11 (RR)
#   MuJoCo 9-11 (RL)   -> Isaac 6-8 (RL)

MUJOCO_TO_ISAAC = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]
ISAAC_TO_MUJOCO = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]

# Isaac Gym 顺序的 default_dof_pos (从 Isaac Gym 直接打印确认):
# FL: 0.1, 0.8, -1.5 | FR: -0.1, 0.8, -1.5 | RL: 0.1, 1.0, -1.5 | RR: -0.1, 1.0, -1.5
DEFAULT_ANGLES_ISAAC = np.array([
    0.1,  0.8, -1.5,   # FL (Isaac dof 0-2)
   -0.1,  0.8, -1.5,   # FR (Isaac dof 3-5)
    0.1,  1.0, -1.5,   # RL (Isaac dof 6-8)
   -0.1,  1.0, -1.5,   # RR (Isaac dof 9-11)
], dtype=np.float64)

# MuJoCo 顺序: FL, FR, RR, RL
DEFAULT_ANGLES_MUJOCO = DEFAULT_ANGLES_ISAAC[ISAAC_TO_MUJOCO]


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
#  全局键盘状态
# ============================================================
vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
height_cmd = 0.25    # 身体高度指令 [m], 默认0.25m
reset_flag = False

def on_press(key):
    global vx_cmd, vy_cmd, wz_cmd, height_cmd, reset_flag
    try:
        if key.char == 'w': vx_cmd = 1.0
        elif key.char == 's': vx_cmd = -1.0
        elif key.char == 'a': vy_cmd = 1.0
        elif key.char == 'd': vy_cmd = -1.0
        elif key.char == 'q': wz_cmd = 1.0
        elif key.char == 'e': wz_cmd = -1.0
        elif key.char == 'r': reset_flag = True
        elif key.char == 'f':  # F: 站起 (升高高度)
            height_cmd = min(0.35, height_cmd + 0.02)
        elif key.char == 'z':  # Z: 重置高度到默认
            height_cmd = 0.25
    except AttributeError:
        if key == keyboard.Key.up: vx_cmd = 1.0
        elif key == keyboard.Key.down: vx_cmd = -1.0
        elif key == keyboard.Key.left: vy_cmd = 1.0
        elif key == keyboard.Key.right: vy_cmd = -1.0
        elif key == keyboard.Key.space: vx_cmd = vy_cmd = wz_cmd = 0.0

def on_release(key):
    global vx_cmd, vy_cmd, wz_cmd
    try:
        if key.char in 'ws': vx_cmd = 0.0
        elif key.char in 'ad': vy_cmd = 0.0
        elif key.char in 'qe': wz_cmd = 0.0
    except AttributeError:
        if key in [keyboard.Key.up, keyboard.Key.down]: vx_cmd = 0.0
        elif key in [keyboard.Key.left, keyboard.Key.right]: vy_cmd = 0.0


def build_single_obs(quat_xyzw, omega, joint_q_isaac, joint_dq_isaac,
                     last_action_isaac, default_angles_isaac,
                     cmd, cmd_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale,
                     clip_obs, height_cmd):
    """
    构建46维单步观测

    前45维与 plane_onnx_45.py 的 build_single_obs 完全一致:
      [0:3]   commands × scale (vx, vy, wz)
      [3:6]   base_ang_vel × scale
      [6:9]   projected_gravity
      [9:21]  dof_pos - default_pos
      [21:33] dof_vel × scale
      [33:45] last_actions

    ★ 第46维 (index 45): 高度指令归一化
      height_obs = (height_cmd - 0.25) / 0.1
      与 Isaac Gym compute_observations 中完全一致
    """
    obs = np.zeros(46, dtype=np.float32)

    # ★ 前45维: 和 plane_onnx_45.py 的 build_single_obs 完全一样
    obs[0:3] = cmd * cmd_scale[:3]

    # 角速度: quat_rotate_inverse
    omega_body = quat_rotate_inverse(quat_xyzw, omega)
    obs[3:6] = omega_body.astype(np.float32) * ang_vel_scale

    gravity_world = np.array([0., 0., -1.], dtype=np.float64)
    proj_gravity = quat_rotate_inverse(quat_xyzw, gravity_world)
    obs[6:9] = proj_gravity.astype(np.float32)

    obs[9:21] = ((joint_q_isaac - default_angles_isaac) * dof_pos_scale).astype(np.float32)
    obs[21:33] = (joint_dq_isaac * dof_vel_scale).astype(np.float32)
    obs[33:45] = last_action_isaac

    # ★ 第46维: 高度指令归一化 (与 Isaac Gym 一致)
    # Isaac Gym: height_obs = (self.commands[:, 4] - 0.25) / 0.1
    obs[45] = (height_cmd - 0.25) / 0.1

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

    base = "/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym"
    config_path = f"{base}/mujoco/dog/config/{args.config_file}"
    policy_path = f"/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/dog_rough/old_urdf/46_terrain-good-1/model_4500.onnx"
    xml_path    = f"{base}/resources/robots/dog/xml/dog_terrain.xml"

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

    # 46维配置
    NUM_ONE_STEP_OBS = config.get("num_one_step_obs", 46)
    NUM_OBS          = config.get("num_obs", 276)
    HISTORY_LEN      = NUM_OBS // NUM_ONE_STEP_OBS   # 276 / 46 = 6
    NUM_ACTIONS      = config.get("num_actions", 12)

    # 高度指令参数
    height_cmd_default = config.get("height_cmd_default", 0.25)
    height_cmd         = height_cmd_default

    print(f"\n{'='*60}")
    print(f"[CONFIG] Dog 46维 — 带 RL/RR 映射 + 高度指令")
    print(f"{'='*60}")
    print(f"  单步观测维度: {NUM_ONE_STEP_OBS}")
    print(f"  总观测维度:   {NUM_OBS} ({HISTORY_LEN}步历史)")
    print(f"  Isaac Gym dof: FL, FR, RL, RR")
    print(f"  MuJoCo dof:    FL, FR, RR, RL")
    print(f"  映射: MUJOCO_TO_ISAAC = {MUJOCO_TO_ISAAC}")
    print(f"  default_isaac:  {DEFAULT_ANGLES_ISAAC}")
    print(f"  default_mujoco: {DEFAULT_ANGLES_MUJOCO}")
    print(f"  kps={kps[0]}, kds={kds[0]}, action_scale={action_scale}")
    print(f"  高度指令: 默认={height_cmd_default}m")
    print(f"  观测结构: cmds(3) + ang_vel(3) + gravity(3) + dof_pos(12) + dof_vel(12) + actions(12) + height(1) = 46")
    print(f"{'='*60}")

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = simulation_dt
    mj_data = mujoco.MjData(mj_model)

    print(f"\n[验证] MuJoCo joint 顺序:")
    for i in range(mj_model.njnt):
        jn = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jn: print(f"  [{i}] {jn}")

    reset_robot(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)

    # 验证: MuJoCo qpos 转换到 Isaac 顺序应该等于 DEFAULT_ANGLES_ISAAC
    q_isaac = mj_data.qpos[7:19][MUJOCO_TO_ISAAC]
    print(f"\n[验证] qpos(MuJoCo→Isaac): {q_isaac}")
    print(f"[验证] 期望(Isaac):         {DEFAULT_ANGLES_ISAAC}")
    print(f"[验证] {'✓ 一致' if np.allclose(q_isaac, DEFAULT_ANGLES_ISAAC) else '✗ 不一致!'}")

    if not args.no_policy:
        policy = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
        input_name  = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name
        print(f"[ONNX] {policy.get_inputs()[0].shape} → {policy.get_outputs()[0].shape}")

    obs_history      = ObsHistoryBuffer(HISTORY_LEN, NUM_ONE_STEP_OBS)
    target_q_mujoco  = DEFAULT_ANGLES_MUJOCO.copy()
    action_isaac     = np.zeros(NUM_ACTIONS, dtype=np.float64)
    last_action_isaac = np.zeros(NUM_ACTIONS, dtype=np.float32)
    count = 0

    # 预热 (用46维观测填充历史缓冲区)
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
            height_cmd
        )
        obs_history.push(obs)

    print(f"[INFO] 预热完成, norm={np.linalg.norm(obs_history.buffer):.4f}")
    print(f"\n  W/S:前后 A/D:左右 Q/E:转 空格:停 R:重置")
    print(f"  F:站起(升高) Z:重置高度  当前高度指令: {height_cmd:.2f}m\n")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            cmd = np.array([vx_cmd, vy_cmd, wz_cmd], dtype=np.float32)

            if reset_flag:
                reset_robot(mj_model, mj_data, DEFAULT_ANGLES_MUJOCO)
                obs_history.reset()
                last_action_isaac[:] = 0; action_isaac[:] = 0; count = 0
                height_cmd = height_cmd_default
                reset_flag = False

            # 读取 MuJoCo 状态
            joint_q_mujoco  = mj_data.qpos[7:19].astype(np.float64)
            joint_dq_mujoco = mj_data.qvel[6:18].astype(np.float64)

            # ★ 映射到 Isaac 顺序 (送给策略)
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
                        height_cmd
                    )
                    obs_history.push(single_obs)

                    obs_input = obs_history.get()
                    action_raw = policy.run([output_name], {input_name: obs_input})[0][0]
                    action_isaac[:] = np.clip(action_raw, -10.0, 10.0)
                    last_action_isaac = action_isaac.astype(np.float32)

                    # ★ 目标角: Isaac顺序 → MuJoCo顺序
                    target_q_isaac = action_isaac * action_scale + DEFAULT_ANGLES_ISAAC
                    target_q_mujoco = target_q_isaac[ISAAC_TO_MUJOCO]
                else:
                    target_q_mujoco = DEFAULT_ANGLES_MUJOCO.copy()

            # PD 控制 (MuJoCo 顺序)
            tau = pd_control(target_q_mujoco, joint_q_mujoco, kps,
                           np.zeros(NUM_ACTIONS), joint_dq_mujoco, kds)
            tau = np.clip(tau, -tau_limit, tau_limit)
            mj_data.ctrl[:NUM_ACTIONS] = tau

            mujoco.mj_step(mj_model, mj_data)
            count += 1

            if count % (control_decimation * 50) == 0:
                grav = quat_rotate_inverse(quat_xyzw, np.array([0., 0., -1.]))
                h = mj_data.qpos[2]
                print(f"[{time.time()-start:.1f}s] Step {count} H={h:.3f} "
                      f"vx={mj_data.qvel[0]:.2f} vy={mj_data.qvel[1]:.2f} wz={mj_data.qvel[5]:.2f} "
                      f"grav_z={grav[2]:.3f} height_cmd={height_cmd:.2f}m "
                      f"act=[{action_isaac.min():.2f},{action_isaac.max():.2f}]")

            viewer.sync()
            elapsed = time.time() - step_start
            if simulation_dt - elapsed > 0:
                time.sleep(simulation_dt - elapsed)

    listener.stop()
    print("\n[INFO] 仿真结束")