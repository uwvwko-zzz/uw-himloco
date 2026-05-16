import os
import sys

# === 设置路径 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
himloco_gym_path = os.path.abspath(os.path.join(script_dir, '../../../..'))

import isaacgym

if himloco_gym_path in sys.path:
    sys.path.remove(himloco_gym_path)
sys.path.insert(0, himloco_gym_path)

import torch
import numpy as np
from pynput import keyboard
from types import MethodType
import onnxruntime as ort


from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.terrain import Terrain

# ========== 全局键盘状态 ==========
vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
reset_flag = False
exit_flag = False
print_obs_flag = False
print_action_flag = False

def on_press(key):
    global vx_cmd, vy_cmd, wz_cmd, reset_flag, exit_flag, print_obs_flag, print_action_flag
    try:
        if key.char == 'w':
            vx_cmd = 1.0
            print(f"[KEY] W pressed → vx={vx_cmd:+.1f}")
        elif key.char == 's':
            vx_cmd = -1.0
            print(f"[KEY] S pressed → vx={vx_cmd:+.1f}")
        elif key.char == 'a':
            vy_cmd = 1.0
            print(f"[KEY] A pressed → vy={vy_cmd:+.1f}")
        elif key.char == 'd':
            vy_cmd = -1.0
            print(f"[KEY] D pressed → vy={vy_cmd:+.1f}")
        elif key.char == 'q':
            wz_cmd = -1.0
            print(f"[KEY] Q pressed → wz={wz_cmd:+.1f}")
        elif key.char == 'e':
            wz_cmd = 1.0
            print(f"[KEY] E pressed → wz={wz_cmd:+.1f}")
        elif key.char == 'r':
            reset_flag = True
            print("[KEY] R pressed → RESET")
        elif key.char == 'n':
            print_obs_flag = True
        elif key.char == 'm':
            print_action_flag = True
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_flag = True
            print("[KEY] ESC pressed → EXIT")

def on_release(key):
    global vx_cmd, vy_cmd, wz_cmd
    try:
        if key.char in 'ws':
            vx_cmd = 0.0
            print(f"[KEY] W/S released → vx={vx_cmd:+.1f}")
        elif key.char in 'ad':
            vy_cmd = 0.0
            print(f"[KEY] A/D released → vy={vy_cmd:+.1f}")
        elif key.char in 'qe':
            wz_cmd = 0.0
            print(f"[KEY] Q/E released → wz={wz_cmd:+.1f}")
    except AttributeError:
        pass


def warmup_history(env, num_warmup_steps=5):
    """预热历史缓冲区：用零动作走几步，让 obs_buf 的 6 步历史被填满"""
    env.commands[0, :] = 0.0
    zero_actions = torch.zeros(1, env.num_actions, device=env.device)
    for _ in range(num_warmup_steps):
        step_result = env.step(zero_actions)
    obs = step_result[0]  # obs_buf
    return obs


def play(args):
    global reset_flag, exit_flag, print_obs_flag, print_action_flag

    if args.headless:
        print("Keyboard control requires GUI. Run without --headless.")
        return

    # ==================== 1. 配置 ====================
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 部署/测试配置
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False    # 部署时通常不用高度扫描
    env_cfg.commands.heading_command = False    # 用直接角速度控制
    env_cfg.commands.resampling_time = 1e6     # 不自动重采样命令
    env_cfg.env.episode_length_s = 1e6         # 不自动结束 episode
    env_cfg.noise.add_noise = False            # 测试时关闭噪声
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.delay = False          # 关闭动作延迟
    env_cfg.env.test = True

    # ==================== 2. 创建环境 ====================
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera([2.0, 2.0, 2.0], [0.0, 0.0, 1.0])

    # 打印关键维度信息
    print(f"\n{'='*60}")
    print(f"📊 环境信息:")
    print(f"   num_obs (总观测维度):          {env.num_obs}")
    print(f"   num_one_step_obs (单步观测):   {env.num_one_step_obs}")
    print(f"   history_size:                  {env.num_obs // env.num_one_step_obs}")
    print(f"   num_actions:                   {env.num_actions}")
    print(f"   num_commands:                  {env.cfg.commands.num_commands}")
    print(f"{'='*60}")

    # ==================== 3. 加载 ONNX 模型 ====================
    ONNX_MODEL_PATH = '/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/dog_rough/Apr19_16-55-39_/model_2500.onnx'
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"[ERROR] ONNX model not found: {ONNX_MODEL_PATH}")
        sys.exit(1)

    ort_session = ort.InferenceSession(
        ONNX_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    input_shape = ort_session.get_inputs()[0].shape
    output_shape = ort_session.get_outputs()[0].shape
    print(f"[✓] ONNX loaded: {ONNX_MODEL_PATH}")
    print(f"    Input:  {input_name}, shape={input_shape}")
    print(f"    Output: {output_name}, shape={output_shape}")

    # ==================== 4. 键盘控制 ====================
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\n" + "="*60)
    print("🎮 键盘控制说明 (焦点需在终端窗口)")
    print("   W/S: 前进/后退  | A/D: 左移/右移  | Q/E: 左转/右转")
    print("   R: 手动复位     | N: 打印当前obs  | M: 打印模型输出  | ESC: 退出")
    print("="*60 + "\n")

    try:
        # ========== 初始化 ==========
        env.reset_idx(torch.tensor([0], device=env.device))
        obs = warmup_history(env, num_warmup_steps=6)
        print("[✓] 历史缓冲区预热完成，开始控制!\n")

        step = 0
        last_print = -100
        while not exit_flag:
            # ---------- 设置命令 ----------
            # commands 有 4 维: [vx, vy, wz, heading]
            # heading_command=False 时, heading 不会被使用
            env.commands[0, 0] = vx_cmd
            env.commands[0, 1] = vy_cmd
            env.commands[0, 2] = wz_cmd
            env.commands[0, 3] = 0.0   # heading 设为 0

            # ---------- ONNX 推理 ----------
            # 关键点：env.get_observations() 返回的 obs_buf 已经包含
            # 所有归一化（commands_scale, obs_scales 等），
            # 不需要再做任何额外的归一化处理！
            obs_np = obs.cpu().numpy().astype(np.float32)
            actions_np = ort_session.run([output_name], {input_name: obs_np})[0]
            actions = torch.from_numpy(actions_np).to(env.device)

            # ---------- 按N打印obs ----------
            if print_obs_flag:
                print_obs_flag = False
                obs_np = obs.cpu().numpy()
                print(f"\n========== [OBS DATA] ==========")
                print(f"Shape: {obs_np.shape}")
                print(f"Full obs:\n{obs_np}")
                print(f"=================================\n")

            # ---------- 按M打印模型输出 ----------
            if print_action_flag:
                print_action_flag = False
                actions_np = actions.cpu().numpy()
                action_scale = env.cfg.control.action_scale
                scaled_np = actions_np * action_scale
                print(f"\n========== [MODEL OUTPUT (ACTIONS)] ==========")
                print(f"Shape: {actions_np.shape}")
                print(f"action_scale: {action_scale}")
                print(f"Raw model output (unscaled):\n{actions_np}")
                print(f"Scaled output (×{action_scale}):\n{scaled_np}")
                print(f"===============================================\n")

            # ---------- 调试输出 ----------
            if step - last_print >= 100:
                # obs_buf 中最新的单步 obs 在最前面 [0:45]
                # 其中 commands*scale 占前 4D
                cmd_scaled = obs[0, :4].cpu().numpy()
                act_stats = f"[{actions.min().item():+.3f}, {actions.max().item():+.3f}]"
                print(f"Step {step:5d} | "
                      f"Cmd(raw): vx={vx_cmd:+.1f} vy={vy_cmd:+.1f} wz={wz_cmd:+.1f} | "
                      f"Cmd(scaled in obs): [{cmd_scaled[0]:+.2f},{cmd_scaled[1]:+.2f},{cmd_scaled[2]:+.2f},{cmd_scaled[3]:+.2f}] | "
                      f"Action range: {act_stats}")
                last_print = step

            # ---------- 环境步进 ----------
            # step() 返回 7 个值:
            # obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras,
            # termination_ids, termination_privileged_obs
            step_result = env.step(actions)
            obs = step_result[0]           # obs_buf (已归一化+裁剪)
            dones = step_result[3]         # reset_buf

            step += 1

            # ---------- 自动复位 ----------
            if isinstance(dones, torch.Tensor) and dones.numel() > 0 and dones[0]:
                if step > 20:
                    print(f"[AUTO-RESET] 倒地复位 @ step {step}")
                env.reset_idx(torch.tensor([0], device=env.device))
                obs = warmup_history(env, num_warmup_steps=6)

            # ---------- 手动复位 ----------
            if reset_flag:
                print(f"[MANUAL RESET] @ step {step}")
                env.reset_idx(torch.tensor([0], device=env.device))
                obs = warmup_history(env, num_warmup_steps=6)
                reset_flag = False

            env.render()

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        listener.stop()
        if hasattr(env, 'viewer') and env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
        print(f"\n[INFO] 仿真结束，共运行 {step} 步")


if __name__ == '__main__':
    args = get_args()
    play(args)