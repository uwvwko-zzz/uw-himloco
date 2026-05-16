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
from pynput import keyboard


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


vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
reset_flag = False
print_action_flag = False

def on_press(key):
    global vx_cmd, vy_cmd, wz_cmd, reset_flag, print_action_flag
    try:
        if key.char == 'w': vx_cmd = 1.0
        elif key.char == 's': vx_cmd = -1.0
        elif key.char == 'a': vy_cmd = 1.0
        elif key.char == 'd': vy_cmd = -1.0
        elif key.char == 'q': wz_cmd = 1.0
        elif key.char == 'e': wz_cmd = -1.0
        elif key.char == 'r': reset_flag = True
        elif key.char == 'y': print_action_flag = True
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
                     clip_obs):
    """
    构建45维单步观测 — 所有关节量使用 IsaacGym dof 顺序: FL, FR, RL, RR
    """
    obs = np.zeros(45, dtype=np.float32)
    obs[0:3] = cmd * cmd_scale[:3]

    omega_body = quat_rotate_inverse(quat_xyzw, omega)
    obs[3:6] = omega_body.astype(np.float32) * ang_vel_scale

    gravity_world = np.array([0., 0., -1.], dtype=np.float64)
    proj_gravity = quat_rotate_inverse(quat_xyzw, gravity_world)
    obs[6:9] = proj_gravity.astype(np.float32)

    obs[9:21] = ((joint_q_isaac - default_angles_isaac) * dof_pos_scale).astype(np.float32)
    obs[21:33] = (joint_dq_isaac * dof_vel_scale).astype(np.float32)
    obs[33:45] = last_action_isaac
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
    policy_path = f"/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/dog_rough/45_terrain_1/model_5000.onnx"
    xml_path    = f"{base}/resources/robots/dog/xml/dog.xml"

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

    NUM_ONE_STEP_OBS = 45
    HISTORY_LEN      = 6
    NUM_ACTIONS      = 12

    print(f"\n{'='*60}")
    print(f"[CONFIG] Dog — 带 RL/RR 映射")
    print(f"{'='*60}")
    print(f"  IsaacGym dof: FL, FR, RL, RR")
    print(f"  MuJoCo dof:   FL, FR, RR, RL")
    print(f"  映射: MUJOCO_TO_ISAAC = {MUJOCO_TO_ISAAC}")
    print(f"  default_isaac:  {DEFAULT_ANGLES_ISAAC}")
    print(f"  default_mujoco: {DEFAULT_ANGLES_MUJOCO}")
    print(f"  kps={kps[0]}, kds={kds[0]}, action_scale={action_scale}")
    print(f"  tau_limits: hip/thigh={TAU_LIMIT_HIP_THIGH}, calf={TAU_LIMIT_CALF}")
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
    labels = ["FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf",
              "RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf"]
    count = 0
    inference_count = 0  # 前10次打印 obs，按 y 后打印模型输出

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
            ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs
        )
        obs_history.push(obs)

    print(f"[INFO] 预热完成, norm={np.linalg.norm(obs_history.buffer):.4f}")
    print(f"\n  W/S:前后 A/D:左右 Q/E:转 空格:停 R:重置 Y:开始打印模型输出\n")

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
                last_action_isaac[:] = 0; action_isaac[:] = 0; count = 0; inference_count = 0
                print_action_flag = False
                reset_flag = False

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
                        ang_vel_scale, dof_pos_scale, dof_vel_scale, clip_obs
                    )
                    obs_history.push(single_obs)
                    obs_input = obs_history.get()

                    if inference_count < 10:
                        print(f"\n{'='*60}")
                        print(f"[推理 #{inference_count}] 完整 270 维 history obs (6帧×45维):")
                        buf = obs_history.buffer
                        for slot in range(HISTORY_LEN):
                            base = slot * 45
                            frame = buf[base:base+45]
                            print(f"\n  --- 帧{slot} (offset {base}) ---")
                            print(f"    cmd:          {frame[0:3]}")
                            print(f"    gyro_body:    {frame[3:6]}")
                            print(f"    proj_gravity: {frame[6:9]}")
                            print(f"    dof_pos_dev:  {np.array2string(frame[9:21], precision=4, separator=', ')}")
                            print(f"    dof_vel:      {np.array2string(frame[21:33], precision=4, separator=', ')}")
                            print(f"    last_action:  {np.array2string(frame[33:45], precision=4, separator=', ')}")
                            print(f"    frame norm:   {np.linalg.norm(frame):.4f}")
                        if inference_count == 9:
                            print(f"\n[INFO] 前10帧 obs 已输出完毕，按 Y 开始打印模型输出。")
                        print(f"{'='*60}")

                    action_raw = policy.run([output_name], {input_name: obs_input})[0][0]
                    action_isaac[:] = np.clip(action_raw, -10.0, 10.0)
                    last_action_isaac = action_isaac.astype(np.float32)

                    if inference_count >= 10 and print_action_flag:
                        print(f"\n[推理 #{inference_count}] 模型输出 action x {OUTPUT_PRINT_SCALE} (Isaac顺序 FL,FR,RL,RR):")
                        for j in range(12):
                            print(f"    {labels[j]:12s}: action={action_isaac[j] * OUTPUT_PRINT_SCALE:+.4f}")

                    inference_count += 1

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
                print(f"[{time.time()-start:.1f}s] Step {count} H={h:.3f} "
                      f"vx={mj_data.qvel[0]:.2f} vy={mj_data.qvel[1]:.2f} wz={mj_data.qvel[5]:.2f} "
                      f"grav_z={grav[2]:.3f} "
                      f"act=[{action_isaac.min():.2f},{action_isaac.max():.2f}]")

            viewer.sync()
            elapsed = time.time() - step_start
            if simulation_dt - elapsed > 0:
                time.sleep(simulation_dt - elapsed)

    listener.stop()
    print("\n[INFO] 仿真结束")
