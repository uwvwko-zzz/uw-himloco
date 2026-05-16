import math
import numpy as np
import mujoco
from mujoco import viewer
from tqdm import tqdm
import onnxruntime as ort
import time
from pynput import keyboard


default_dof_pos = np.array([
    0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,  # FR
    0.1, 1.0, -1.5,   # RL  
    -0.1, 1.0, -1.5   # RR
], dtype=np.float32)



# qpos 顺序 → ctrl 顺序的映射
# QPOS_TO_CTRL_MAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# PD 参数
KPS = 20.0
KDS = 0.5
TAU_LIMIT = 23.7

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
        mujoco_model_path = '/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/resources/robots/go1/xml/go1.xml'
        sim_duration = 60.0
        dt = 0.005
        decimation = 4

    class robot_config:
        kps = KPS
        kds = KDS
        tau_limit = TAU_LIMIT

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

        # 正确的初始姿态（与 keyframe 一致）
        initial_qpos = np.array([
            0.0, 0.0, 0.27,  # x, y, z (Go2 站立高度 0.27m)
            1.0, 0.0, 0.0, 0.0,  # quaternion
            *default_dof_pos
        ])
        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)

        target_dof_pos = default_dof_pos.copy()
        counter = 0

        print("\n=== Go2 Model Validation (Fixed Posture) ===")
        print("Using correct joint order mapping and standing height")
        print("Check Joint panel for actual angles (not Control panel!)\n")
        
        with viewer.launch_passive(model, data) as v:
            v.cam.distance = 1.5
            v.cam.elevation = -15
            v.cam.azimuth = 45
            v.cam.lookat[:] = [0.0, 0.0, 0.2]
            
            start_time = time.time()
            while v.is_running() and not exit_flag and time.time() - start_time < Sim2simConfig.sim_config.sim_duration:
                step_start = time.time()
                
                # PD 控制（qpos 顺序）
                tau = pd_control(
                    target_dof_pos, data.qpos[7:], 
                    Sim2simConfig.robot_config.kps,
                    np.zeros(12), data.qvel[6:], 
                    Sim2simConfig.robot_config.kds
                )
                tau = np.clip(tau, -Sim2simConfig.robot_config.tau_limit, Sim2simConfig.robot_config.tau_limit)
                
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