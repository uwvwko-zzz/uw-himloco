# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ... （许可证部分保持不变）

import os
import numpy as np
from datetime import datetime
import sys
import isaacgym
import torch
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger


def play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0, height_cmd=0.25, model_path=None):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    
    # ✅ 设置高度指令（如果支持的话）
    if env_cfg.commands.num_commands >= 5:
        env.commands[:, 4] = height_cmd

    obs = env.get_observations()
    
    # ✅ 加载策略
    train_cfg.runner.resume = False  # 禁用自动加载
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    # ✅ 如果指定了模型路径，使用该路径；否则使用默认最新模型
    if model_path:
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            sys.exit(1)
        print(f"📂 加载模型: {model_path}")
        ppo_runner.load(model_path)
    else:
        # 使用默认行为：加载最新的模型
        print(f"📂 自动加载最新模型...")
        train_cfg.runner.resume = True
        ppo_runner.load(ppo_runner.load_path)
    
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    print(f"\n[INFO] 运行参数:")
    print(f"  前进速度 (x_vel): {x_vel} m/s")
    print(f"  侧移速度 (y_vel): {y_vel} m/s")
    print(f"  旋转速度 (yaw_vel): {yaw_vel} rad/s")
    if env_cfg.commands.num_commands >= 5:
        print(f"  身体高度 (height): {height_cmd} m")
    print()

    for i in range(10*int(env.max_episode_length)):
    
        actions = policy(obs.detach())
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        
        # ✅ 设置高度指令
        if env_cfg.commands.num_commands >= 5:
            env.commands[:, 4] = height_cmd
        
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'command_height': env.commands[robot_index, 4].item() if env_cfg.commands.num_commands >= 5 else 0.0,
                    'base_height': env.root_states[robot_index, 2].item(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    
    # ✅ 在 get_args() 之前添加自定义参数
    import sys
    
    # 检查是否有 --model 参数，如果有就先解析它
    model_path = None
    x_vel = 1.0
    y_vel = 0.0
    yaw_vel = 0.0
    height_cmd = 0.25
    
    # 简单的命令行解析
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--model' and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            sys.argv.pop(i)  # 删除 --model
            sys.argv.pop(i)  # 删除 model 路径
        elif sys.argv[i] == '--x-vel' and i + 1 < len(sys.argv):
            x_vel = float(sys.argv[i + 1])
            sys.argv.pop(i)
            sys.argv.pop(i)
        elif sys.argv[i] == '--y-vel' and i + 1 < len(sys.argv):
            y_vel = float(sys.argv[i + 1])
            sys.argv.pop(i)
            sys.argv.pop(i)
        elif sys.argv[i] == '--yaw-vel' and i + 1 < len(sys.argv):
            yaw_vel = float(sys.argv[i + 1])
            sys.argv.pop(i)
            sys.argv.pop(i)
        elif sys.argv[i] == '--height' and i + 1 < len(sys.argv):
            height_cmd = float(sys.argv[i + 1])
            sys.argv.pop(i)
            sys.argv.pop(i)
        else:
            i += 1
    
    # 现在调用 get_args()，它会处理剩余的参数
    args = get_args()
    
    play(args, x_vel=x_vel, y_vel=y_vel, 
         yaw_vel=yaw_vel, height_cmd=height_cmd, 
         model_path=model_path)
    

# python play.py --task=dog  --model /home/extra/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/logs/dog_rough/Z_46_demo_plane/model_2400.pt