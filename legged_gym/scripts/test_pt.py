import os
import numpy as np
from datetime import datetime
import sys
import isaacgym
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1  # ★ 只用1个环境
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.mesh_type = 'plane'  # ★ 平地
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.heading_command = False  # ★ 关闭heading

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel

    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ★★★ 打印关键信息 ★★★
    print("\n" + "=" * 60)
    print("Isaac Gym DOF 信息")
    print("=" * 60)
    print(f"dof_names: {env.dof_names}")
    print(f"num_dofs:  {env.num_dof}")
    print(f"default_dof_pos: {env.default_dof_pos[0].cpu().numpy()}")
    print(f"p_gains: {env.p_gains.cpu().numpy()}")
    print(f"d_gains: {env.d_gains.cpu().numpy()}")
    print(f"torque_limits: {env.torque_limits.cpu().numpy()}")
    print(f"action_scale: {env.cfg.control.action_scale}")
    print(f"commands_scale: {env.commands_scale.cpu().numpy()}")
    print(f"init_pos: {env.root_states[0, :3].cpu().numpy()}")
    print(f"base_quat(xyzw): {env.base_quat[0].cpu().numpy()}")
    print(f"gravity_vec: {env.gravity_vec[0].cpu().numpy()}")
    print("=" * 60)

    for i in range(10):
        actions = policy(obs.detach())
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel

        # ★★★ 在 step 之前打印当前状态 ★★★
        if i < 3:
            r = 0  # robot index
            print(f"\n--- Step {i} (before env.step) ---")
            print(f"  base_pos:        {env.root_states[r, :3].cpu().numpy()}")
            print(f"  base_quat(xyzw): {env.base_quat[r].cpu().numpy()}")
            print(f"  root_angvel_world: {env.root_states[r, 10:13].cpu().numpy()}")
            print(f"  base_ang_vel(body): {env.base_ang_vel[r].cpu().numpy()}")
            print(f"  projected_gravity:  {env.projected_gravity[r].cpu().numpy()}")
            print(f"  dof_pos: {env.dof_pos[r].cpu().numpy()}")
            print(f"  dof_vel: {env.dof_vel[r].cpu().numpy()}")
            print(f"  actions: {actions[r].detach().cpu().numpy()}")
            print(f"  self.actions: {env.actions[r].cpu().numpy()}")
            print(f"  last_actions: {env.last_actions[r].cpu().numpy()}")
            print(f"  torques: {env.torques[r].cpu().numpy()}")
            print(f"  obs_buf[0:3]:   {env.obs_buf[r, 0:3].cpu().numpy()}")
            print(f"  obs_buf[3:6]:   {env.obs_buf[r, 3:6].cpu().numpy()}")
            print(f"  obs_buf[6:9]:   {env.obs_buf[r, 6:9].cpu().numpy()}")
            print(f"  obs_buf[9:21]:  {env.obs_buf[r, 9:21].cpu().numpy()}")
            print(f"  obs_buf[21:33]: {env.obs_buf[r, 21:33].cpu().numpy()}")
            print(f"  obs_buf[33:45]: {env.obs_buf[r, 33:45].cpu().numpy()}")

        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        # ★★★ step 之后打印 ★★★
        if i < 3:
            print(f"\n--- Step {i} (after env.step) ---")
            print(f"  base_pos:        {env.root_states[r, :3].cpu().numpy()}")
            print(f"  base_ang_vel(body): {env.base_ang_vel[r].cpu().numpy()}")
            print(f"  projected_gravity:  {env.projected_gravity[r].cpu().numpy()}")
            print(f"  dof_pos: {env.dof_pos[r].cpu().numpy()}")
            print(f"  torques: {env.torques[r].cpu().numpy()}")
            print(f"  height: {env.root_states[r, 2].item():.4f}")

    print("\n[INFO] 完成")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0)