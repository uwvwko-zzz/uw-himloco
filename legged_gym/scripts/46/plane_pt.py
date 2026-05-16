import os
import sys

# === 设置路径 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
himloco_gym_path = os.path.abspath(os.path.join(script_dir, '../../../..', 'himloco_gym'))

import isaacgym

if himloco_gym_path in sys.path:
    sys.path.remove(himloco_gym_path)
sys.path.insert(0, himloco_gym_path)

import torch
import numpy as np
from pynput import keyboard

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

# ========== 全局键盘状态 ==========
vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
height_cmd = 0.25  
reset_flag = False
exit_flag = False

def on_press(key):
    global vx_cmd, vy_cmd, wz_cmd, height_cmd, reset_flag, exit_flag
    try:
        if key.char == 'w':
            vx_cmd = 1.0
        elif key.char == 's':
            vx_cmd = -1.0
        elif key.char == 'a':
            vy_cmd = 1.0
        elif key.char == 'd':
            vy_cmd = -1.0
        elif key.char == 'q':
            wz_cmd = -1.0
        elif key.char == 'e':
            wz_cmd = 1.0
        elif key.char == 'r':  # R：蹲下（降低）
            height_cmd = max(0.20, height_cmd - 0.02)  # 范围 [0.20, 0.30]，每次降 0.02m
        elif key.char == 'f':  # F：站起（升高）
            height_cmd = min(0.30, height_cmd + 0.02)  # 范围 [0.20, 0.30]，每次升 0.02m
        elif key.char == 'z':  # Z：重置高度
            height_cmd = 0.25
        elif key.char == 'p':  # P：重置机器人位置
            reset_flag = True
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_flag = True

def on_release(key):
    global vx_cmd, vy_cmd, wz_cmd
    try:
        if key.char in 'ws':
            vx_cmd = 0.0
        elif key.char in 'ad':
            vy_cmd = 0.0
        elif key.char in 'qe':
            wz_cmd = 0.0
    except AttributeError:
        pass

def play(args):
    global reset_flag, exit_flag, height_cmd

    if args.headless:
        print("Keyboard control requires GUI. Run without --headless.")
        return

    # --- 1. 获取配置（保持 HIMLoco 原始配置）---
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False  # 必须为 True
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.env.episode_length_s = 1e6
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.env.test = True

    # --- 2. 创建环境 ---
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera([2.0, 2.0, 2.0], [0.0, 0.0, 1.0])

    print(f"[INFO] Observation dimension: {env.num_obs}")  # ✅ 应该是 276

    # --- 3. 创建 runner 但不自动加载 ---
    train_cfg.runner.resume = False  # 禁用自动加载
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, 
        name=args.task, 
        args=args, 
        train_cfg=train_cfg
    )
    
    # === 手动指定完整路径加载（关键修复）===
    policy_path = "/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/dog_rough/46_plane_1/model_2500.pt"
    print(f"[INFO] Loading policy from: {policy_path}")
    
    # 直接调用 runner.load() 传入完整路径
    # ppo_runner.jit.load(policy_path)
    ppo_runner.load(policy_path)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    print("[✓] HIMActorCritic loaded successfully! (Actor input: 65D, Estimator input: 276D)")

    # --- 4. 键盘控制 ---
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\n=== Keyboard Control ===")
    print("Focus this TERMINAL window to control!")
    print("Movement:")
    print("  W/S: forward/backward (±1.5 m/s)")
    print("  A/D: strafe left/right (±1.5 m/s)")
    print("  Q/E: turn left/right (±1.0 rad/s)")
    print("\nHeight Control (站立/蹲下):")
    print("  F: 站起 ↑ (升高身体，范围 0.20-0.30m)")
    print("  R: 蹲下 ↓ (降低身体，范围 0.20-0.30m)")
    print("  Z: 重置高度到默认 (0.25m)")
    print("\nOther:")
    print("  P: 重置机器人位置")
    print("  ESC: 退出\n")

    try:
        obs = env.get_observations()
        step_count = 0
        
        while not exit_flag:
            # 设置速度命令
            env.commands[0, 0] = vx_cmd
            env.commands[0, 1] = vy_cmd
            env.commands[0, 2] = wz_cmd
            
            # 设置高度指令（第 5 个命令，索引 4）
            env.commands[0, 4] = height_cmd

            with torch.no_grad():
                actions = policy(obs)  # ← 直接调用 policy

            obs, privileged_obs, rewards, dones, infos, timeouts, _ = env.step(actions)
            
            # 每 50 步打印一次状态
            step_count += 1
            if step_count % 50 == 0:
                base_height = env.root_states[0, 2].item()
                print(f"[Step {step_count}] Height: {base_height:.3f}m | "
                      f"Command: {height_cmd:.3f}m | "
                      f"Velocity: ({vx_cmd:.2f}, {vy_cmd:.2f}, {wz_cmd:.2f})")

            if reset_flag:
                print("[RESET] Resetting robot...")
                env.reset_idx(torch.tensor([0], device=env.device))
                obs = env.get_observations()
                reset_flag = False

            env.render()

    finally:
        listener.stop()
        if hasattr(env, 'viewer') and env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
        print("\n[INFO] Simulation terminated cleanly.")

if __name__ == '__main__':
    args = get_args()
    play(args)