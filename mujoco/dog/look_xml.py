import math
import numpy as np
import mujoco
from mujoco import viewer
from tqdm import tqdm
import onnxruntime as ort
import time
from pynput import keyboard


# URDF/MuJoCo 关节顺序: FL, FR, RR, RL
# 对应 IsaacGym 的 dof 顺序 (由 URDF 中 joint 定义顺序决定)
default_dof_pos = np.array([
    -0.1, -0.8, -1.5,    # FL  (dof 0-2):  hip=-0.1, thigh=-0.8, calf=-1.5
     0.1,  0.8,  1.5,    # FR  (dof 3-5):  hip= 0.1, thigh= 0.8, calf= 1.5
    -0.1,  1.0,  1.5,    # RR  (dof 6-8):  hip=-0.1, thigh= 1.0, calf= 1.5
     0.1, -1.0, -1.5,    # RL  (dof 9-11): hip= 0.1, thigh=-1.0, calf=-1.5
], dtype=np.float32)



# qpos 顺序 → ctrl 顺序的映射
# QPOS_TO_CTRL_MAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# PD 参数
KPS = 40.0
KDS = 1.0
TAU_LIMIT_HIP_THIGH = 17.0
TAU_LIMIT_CALF = 34.0

exit_flag = False

def on_press(key):
    global exit_flag
    try:
        if key.char == 'q' or key == keyboard.Key.esc:
            exit_flag = True
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_flag = True

def on_release(key):
    pass

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

class Sim2simConfig:
    class sim_config:
        mujoco_model_path = '/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/resources/robots/dog/xml/dog.xml'
        sim_duration = 60.0
        dt = 0.005
        decimation = 4

    class robot_config:
        kps = KPS
        kds = KDS
        tau_limit_hip_thigh = TAU_LIMIT_HIP_THIGH
        tau_limit_calf = TAU_LIMIT_CALF

def main():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    try:
        # === 仅注释策略加载 ===
        # policy_model_path = "..."
        # policy = ort.InferenceSession(...)
        
        model = mujoco.MjModel.from_xml_path(Sim2simConfig.sim_config.mujoco_model_path)
        model.opt.timestep = Sim2simConfig.sim_config.dt
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        model.opt.gravity = (0, 0, -9.81)

        # 正确的初始姿态（与 IsaacGym init_state 一致）
        initial_qpos = np.array([
            0.0, 0.0, 0.42,  # x, y, z (dog 站立高度 0.42m)
            1.0, 0.0, 0.0, 0.0,  # quaternion
            *default_dof_pos
        ])
        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)

        target_dof_pos = default_dof_pos.copy()
        counter = 0

        print("\n=== Dog Model Validation (Fixed Posture) ===")
        print("Using correct joint order: FL, FR, RR, RL (matching URDF/MuJoCo)")
        print("Check Joint panel for actual angles (not Control panel!)\n")
        
        with viewer.launch_passive(model, data) as v:
            v.cam.distance = 1.5
            v.cam.elevation = -15
            v.cam.azimuth = 45
            v.cam.lookat[:] = [0.0, 0.0, 0.2]
            
            start_time = time.time()
            while v.is_running() and not exit_flag and time.time() - start_time < Sim2simConfig.sim_config.sim_duration:
                step_start = time.time()
                
                # PD 控制（qpos 顺序: FL, FR, RR, RL）
                tau = pd_control(
                    target_dof_pos, data.qpos[7:], 
                    Sim2simConfig.robot_config.kps,
                    np.zeros(12), data.qvel[6:], 
                    Sim2simConfig.robot_config.kds
                )
                # 按关节分别裁剪力矩: hip/thigh 23.7Nm, calf 35.55Nm
                tau_limits = np.array([
                    Sim2simConfig.robot_config.tau_limit_hip_thigh,
                    Sim2simConfig.robot_config.tau_limit_hip_thigh,
                    Sim2simConfig.robot_config.tau_limit_calf,
                ] * 4)  # FL, FR, RR, RL
                tau = np.clip(tau, -tau_limits, tau_limits)
                
                # 映射到 ctrl 顺序
                data.ctrl[:] = tau
                # data.ctrl[:] = tau
                # 仿真步进
                mujoco.mj_step(model, data)
                counter += 1

                # === 注释策略更新 ===
                # if counter % Sim2simConfig.sim_config.decimation == 0:
                #     ... # 策略相关代码
                
                # 同步 viewer
                v.sync()
                
                # 时间同步
                time_until_next_step = Sim2simConfig.sim_config.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        print("Simulation finished.")
        
    finally:
        listener.stop()

if __name__ == '__main__':
    main()