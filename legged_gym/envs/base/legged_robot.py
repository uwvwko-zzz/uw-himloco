# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, # 配置对象
                 sim_params,                # 模拟器参数
                 physics_engine,            # 物理引擎类型
                 sim_device,                # 模拟设备
                 headless):                 # 是否无头模式
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg                      # 保存配置对象
        self.sim_params = sim_params        # 模拟参数
        self.height_samples = None          # 初始化高度采样缓冲区
        self.debug_viz = False              # 默认不可视化
        self.init_done = False              # 初始化完成标志
        self._parse_cfg(self.cfg)           # 解析配置文件
        # 调用父类初始化 
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        # 获取单步观测维度 
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        # 获取单步特权观测维度
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        # 历史布数
        self.history_length = int(self.num_obs / self.num_one_step_obs)

        if not self.headless:
            # 设置相机
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()                # 初始化 PyTorch 张量缓冲区 
        self._prepare_reward_function()     # 准备奖励函数
        self.init_done = True               # 标记初始化完成


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 动作裁剪
        clip_actions = self.cfg.normalization.clip_actions      # 从配置中读取的裁剪值
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # 延迟动作生成
        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
       
        # 随机延迟采样
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        # 应用动作延迟
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        # step physics and render each frame
        
        self.render()               # 渲染
        # 物理模拟循环
        for _ in range(self.cfg.control.decimation):
            # 计算力矩
            # 目标位置 ← 动作 × 缩放系数 + 默认位置
            # 位置误差 = 目标位置 - 当前位置
            # 力矩 = 刚度 × 位置误差 - 阻尼 × 速度
            #       ↑                ↑
            #      比例项            微分项
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            # 设置力矩
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # 运行物理模拟
            self.gym.simulate(self.sim)
            # 同步 CPU
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            # 刷新状态，更新 PyTorch 张量中的关节状态
            self.gym.refresh_dof_state_tensor(self.sim)
        # 模拟后的处理
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        # 观测裁剪
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # 特权观测裁剪
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # 返回结果
        # obs_buf	            [4096, 276]	观测（Actor 使用）
        # privileged_obs_buf	[4096, 1434]	特权观测（Critic 使用）
        # rew_buf	            [4096]	奖励
        # reset_buf	            [4096]	是否重置（bool）
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)      # 刷新根节点状态
        self.gym.refresh_net_contact_force_tensor(self.sim)     # 刷新接触力
        self.gym.refresh_rigid_body_state_tensor(self.sim)      # 刷新刚体状态

        self.episode_length_buf += 1        # Episode 长度计数
        self.common_step_counter += 1       # 全局计数器

        # prepare quantities
        # 准备观测量，提取身体四元数
        self.base_quat[:] = self.root_states[:, 3:7]
        # 将速度从世界坐标转到身体坐标
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 重力投影到身体坐标系
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # 提取足部信息
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        # 公共处理
        self._post_physics_step_callback()

        # ===== 倒地自动恢复（在终止检测之前执行）=====
        # 如果启用了恢复功能，倒地的环境会就地恢复，不会被终止
        if getattr(self.cfg, 'recovery', None) and self.cfg.recovery.enable_recovery:
            self._check_and_recover()

        # compute observations, rewards, resets, ...
        # 检查终止条件
        self.check_termination()
        # 计算奖励
        self.compute_reward()
        # 获取终止的环境 ID
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # 计算终止时的观测
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        # 重置环境
        self.reset_idx(env_ids)
        # 重新计算观测
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # 清除干扰
        self.disturbance[:, :, :] = 0.0
        # 更新历史动作
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        # 保存速度历史
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # 调试可视化
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        # 需要重置的环境 ID，摔倒时的观测
        return env_ids, termination_privileged_obs

    # 是否重置
    def check_termination(self):
        """ 倒地恢复任务的终止条件：
            恢复任务初始即为随机倒地姿态（base 必然贴地），因此不能用 base 接触判定失败，
            否则 episode 第一步就会重置。终止只依赖：
              - 超时 -> 重置（仍未站起）
              - 成功站起来(upright & height达标) -> 重置（成功）
        """
        # 恢复任务不靠接触力终止（初始倒地时 base/thigh/calf 都会着地）
        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        # ===== 更新"近期扰动峰值"（平稳到达门槛用）=====
        # agitation = ang_vel_xy² + lin_vel_z²：翻滚(高角速度)+腾空(高垂直速度)的合成指标。
        # 每步取 max(自身×decay, 当前值)，模拟一个带衰减的峰值追踪器——
        # 即使翻滚后这一帧恰好平稳，前几步的剧烈扰动仍会让峰值维持高位，从而拦住"翻滚到达"。
        agitation = (torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
                     + torch.square(self.base_lin_vel[:, 2]))
        decay = self.cfg.rewards.smooth_success_decay
        self.recent_max_agitation = torch.maximum(
            self.recent_max_agitation * decay, agitation)

        # 严格直立判定：grav_z < -0.9（正立时 grav_z 约为 -1）。
        # 绝不能用 abs()——abs>0.9 会同时命中"完全倒立"(grav_z=+1)，让策略学会翻倒骗奖励。
        upright = self.projected_gravity[:, 2] < -0.9
        # 高度达标跟随指令：当前高度落在 [cmd - tol, cmd + tol] 区间
        height_tol = self.cfg.rewards.stand_height_tolerance
        base_height_ok = torch.abs(self.root_states[:, 2] - self.commands[:, 4]) < height_tol
        # 关节接近 default：强制恢复到标准站姿，而非"直立但姿势奇怪"的状态
        joint_err = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        joint_ok = joint_err < self.cfg.rewards.recovery_joint_tol
        recovered = upright & base_height_ok & joint_ok

        # "平稳到达"门槛：recovered 额外要求近期扰动峰值低于阈值。
        # 不加门槛时，"大力翻滚恰好落成 default"也算 recovered → 拿到首次成功 ×10，作弊划算。
        # 加门槛后，翻滚到达时近期峰值很高，不算 recovered，从源头排除作弊。
        # 注意：此门槛同时影响 ever_success_buf（成功率统计），即"翻滚到达不计入成功"——符合语义。
        if getattr(self.cfg.rewards, 'smooth_success_enable', False):
            smooth_ok = self.recent_max_agitation < self.cfg.rewards.smooth_success_threshold
            recovered = recovered & smooth_ok

        self.success_buf = recovered.clone()
        # 不再"成功即终止"：原 reset_buf |= recovered 会让达标环境立刻重置，
        # 策略随即发现"悬停在成功边缘(grav_z≈-0.8)刷满整段 shaping 收益"远高于
        # "真正达标触发的一次性奖励"，从而主动维持不达标——这正是"背朝上前倾抽搐"
        # 局部最优的根源。改为：成功只持续给密集奖励，跑到 timeout 才重置，
        # 这样"站得越早、剩余时间拿满分的步数越多"，策略会从"避免成功"翻转为"尽快成功并保持"。
        self.ever_success_buf |= recovered   # 本 episode 内曾成功达标（统计成功率用）
        # recovery_success 改为每 episode 首次成功才发，置位 already_succeeded_buf 防止倒下重来再刷
        self.already_succeeded_buf[recovered] = True


    def reset_idx(self, env_ids):
        """ 记录恢复成功率并重置环境。 """
        if not hasattr(self, "recovery_success_count"):
            self.recovery_success_count = 0
            self.recovery_total_count = 0
        if len(env_ids) > 0:
            # 取消"成功即终止"后，episode 跑到 timeout 才重置。
            # success_buf 只反映"最后一步"是否达标，无法代表整个 episode。
            # 用 ever_success_buf（本 episode 内任意一步曾达标）统计成功率。
            self.recovery_success_count += int(torch.sum(self.ever_success_buf[env_ids]).item())
            self.recovery_total_count += len(env_ids)
            # 成功率驱动课程：必须在 ever_success_buf 清零之前调用（读取本 episode 成败标志）
            self._update_recovery_curriculum(env_ids)
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        # 检查是否有环境需要重置
        if len(env_ids) == 0:
            return
        # update curriculum
        # 根据重置的环境数量，逐步增加地形难度
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # 每个全局 episode 结束时，增加命令范围难度
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)               # 重置关节位置和速度
        self._reset_root_states(env_ids)        # 重置机器人的位置、方向、速度

        self._resample_commands(env_ids)        # 为重置的环境采样新的命令

        # reset buffers
        # 重置 buffer
        self.last_actions[env_ids] = 0.         # 清除动作历史
        self.last_last_actions[env_ids] = 0.    
        self.last_dof_vel[env_ids] = 0.         
        self.feet_air_time[env_ids] = 0.        # 清除脚部腾空时间
        self.reset_buf[env_ids] = 1             # 标记重置完成
        self.ever_success_buf[env_ids] = False  # 清零本 episode 成功标记，供下一个 episode 重新统计
        self.already_succeeded_buf[env_ids] = False  # 清零"已领过成功重奖"标记，新 episode 重新可领
        self.recent_max_agitation[env_ids] = 0.0     # 清零近期扰动峰值，新 episode 从 0 开始累积

        # update height measurements
        # 重新测量地形高度
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
        # reset randomized prop
        # Domain Randomization - 随机刚度（随机改变关节刚度）
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        # 随机阻尼
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        # 随机电机强度（随机改变电机的最大力矩输出）
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        # 将随机化的参数应用到物理引擎
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # fill extras
        # 填充日志信息
        self.extras["episode"] = {}
        # 计算奖励平均值
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            # 重置累积值
            self.episode_sums[key][env_ids] = 0.

        # log additional curriculum info
        # 地形课程学习日志
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        # 命令课程学习日志
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # ===== 恢复课程诊断（tensorboard + 控制台）=====
        # 记录当前倒地难度、滚动成功率、近期扰动峰值分布，用于监控课程进度和收紧平稳阈值。
        if getattr(self.cfg.rewards, 'recovery_curriculum', False):
            self.extras["episode"]["fall_angle"] = float(self.fall_angle_current)
            if hasattr(self, 'recent_success_flags') and len(self.recent_success_flags) > 0:
                rate = sum(self.recent_success_flags) / len(self.recent_success_flags)
                self.extras["episode"]["recovery_success_rate"] = float(rate)
        self.extras["episode"]["recent_max_agitation_mean"] = float(torch.mean(self.recent_max_agitation).item())
        # 低频控制台打印：每 500 全局步一次，便于实时看阈值是否合理
        if self.common_step_counter % 500 == 0:
            agit = self.recent_max_agitation
            gz = self.projected_gravity[:, 2]
            print(f"[RECOVERY DIAG] step={self.common_step_counter} "
                  f"fall_angle={getattr(self, 'fall_angle_current', 0):.2f} "
                  f"rate={sum(getattr(self, 'recent_success_flags', [0]))/max(1,len(getattr(self,'recent_success_flags',[1]))):.2f} "
                  f"agit(p50/p95/max)={torch.quantile(agit,0.5).item():.1f}/"
                  f"{torch.quantile(agit,0.95).item():.1f}/{agit.max().item():.1f} "
                  f"lin_vel_z_rms={torch.sqrt(torch.mean(self.base_lin_vel[:,2]**2)).item():.2f} "
                  f"ang_xy_rms={torch.sqrt(torch.mean(torch.sum(self.base_ang_vel[:,:2]**2,dim=1))).item():.2f}")
        # send timeout info to the algorithm
        # 超时信息日志
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # 重置 Episode 计数器
        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # 初始化奖励缓冲区，将所有环境的奖励重置为 0
        self.rew_buf[:] = 0.
        # 循环计算各个奖励项
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]                                         # 获取奖励函数和名称
            rew = self.reward_functions[i]() * self.reward_scales[name]         # 计算单个奖励项
            self.rew_buf += rew                                                 # 累加到总奖励
            self.episode_sums[name] += rew                                      # 累加到 Episode 总和
        
        # 只保留正奖励
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        # 添加终止奖励
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    # def compute_observations(self):
    #     """ Computes observations
    #     """
    #     # 正常的45维度
    #     current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
    #                                 self.base_ang_vel  * self.obs_scales.ang_vel,
    #                                 self.projected_gravity,
    #                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                                 self.dof_vel * self.obs_scales.dof_vel,
    #                                 self.actions
    #                                 ),dim=-1)
    #     # 加噪声
    #     if self.add_noise:
    #         current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

    #     # add perceptive inputs if not blind
    #     # +3D 真实线速度，3D 外部扰动力
    #     current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)

    #     # 如果开了terrain，加182高度
    #     if self.cfg.terrain.measure_heights:
    #         heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
    #         heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
    #         current_obs = torch.cat((current_obs, heights), dim=-1)

    #     # 普通观测：6步历史 × 45D = 270D
    #     self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
    #     # 特权观测：1步 × 238D
    #     self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)



    def compute_observations(self):
        """ Computes observations
        """
        # 第一步：构建单步观测基础部分（45维，不含高度指令）
        current_obs = torch.cat((
            self.commands[:, :3] * self.commands_scale,         # [3] 命令：线速度x,y + 角速度z
            self.base_ang_vel * self.obs_scales.ang_vel,        # [3] 身体角速度（来自IMU）
            self.projected_gravity,                             # [3] 重力加速度（投影到机器人坐标系）       
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,         # [12] 关节位置偏差
            self.dof_vel * self.obs_scales.dof_vel,             # [12] 关节速度
            self.actions[:, :self.num_actions]                  # [12] 上一步的动作 
        ), dim=-1)

        # 第二步：在 cat 之前加噪声（教师-学生蒸馏的前提：学生看到带噪观测）
        # 必须在拼高度指令之前，否则 cat 产生的新张量与 current_obs 脱钩，加噪无效
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[-1]]

        # 第三步：添加第46维（高度指令，不加噪——它是确定性的目标值）
        if self.cfg.commands.num_commands >= 5:
            height_obs = (self.commands[:, 4] - 0.25) / 0.1
            current_obs_one_step = torch.cat([current_obs, height_obs.unsqueeze(1)], dim=-1)
        else:
            current_obs_one_step = current_obs

        # 第四步：维护历史缓冲区（46 维历史，带噪）—— Actor/学生输入
        self.obs_buf = torch.cat((
            current_obs_one_step,                               # [batch, 46] 带噪单步观测
            self.obs_buf[:, :-self.num_one_step_obs]            # [batch, 230] 旧历史
        ), dim=-1)

        # 第五步：构建特权观测（Critic/教师输入）
        # 单步部分用【带噪】的学生视角（与 obs_buf 一致），lin_vel/disturbance 是真实特权值（不加噪）
        current_obs_full = torch.cat((
            current_obs_one_step,                           # [46] 带噪单步观测（与 Actor 输入一致）
            self.base_lin_vel * self.obs_scales.lin_vel,    # [3] 真实线速度（特权，不加噪）
            self.disturbance[:, 0, :]                       # [3] 真实外扰动（特权，不加噪）
        ), dim=-1)

        # 第六步：添加高度测量（特权，加噪）
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, 
                -1, 1.
            ) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[
                (9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)
            ]
            current_obs_full = torch.cat((current_obs_full, heights), dim=-1)

        # 第七步：维护特权观测缓冲区
        self.privileged_obs_buf = torch.cat((
            current_obs_full[:, :self.num_one_step_privileged_obs], 
            self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]
        ), dim=-1)
    def get_current_obs(self):
        """返回单步观测（46维），用于其他计算"""
        current_obs = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        
        # 添加第46维（与 compute_observations 保持一致）
        if self.cfg.commands.num_commands >= 5:
            height_obs = (self.commands[:, 4] - 0.25) / 0.1
            current_obs_one_step = torch.cat([current_obs, height_obs.unsqueeze(1)], dim=-1)
        else:
            current_obs_one_step = current_obs

        return current_obs_one_step


    def compute_termination_observations(self, env_ids):
        """计算终止观测"""
        # 第一步：构建单步观测基础部分（45维，不含高度指令）
        current_obs = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        # 第二步：在 cat 之前加噪声（与 compute_observations 保持一致）
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[-1]]

        # 第三步：添加第46维（高度指令，不加噪）
        if self.cfg.commands.num_commands >= 5:
            height_obs = (self.commands[:, 4] - 0.25) / 0.1
            current_obs_one_step = torch.cat([current_obs, height_obs.unsqueeze(1)], dim=-1)
        else:
            current_obs_one_step = current_obs

        # 第四步：构建特权观测（单步部分带噪，lin_vel/disturbance 真实值不加噪）
        current_obs_full = torch.cat((
            current_obs_one_step,
            self.base_lin_vel * self.obs_scales.lin_vel, 
            self.disturbance[:, 0, :]
        ), dim=-1)

        # 第五步：添加高度测量（特权，加噪）
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, 
                -1, 1.
            ) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[
                (9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)
            ]
            current_obs_full = torch.cat((current_obs_full, heights), dim=-1)

        # 第六步：返回特权观测的历史缓冲区
        privileged_obs = torch.cat((
            current_obs_full[:, :self.num_one_step_privileged_obs], 
            self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]
        ), dim=-1)
        
        return privileged_obs[env_ids]
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # 创建物理引擎
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        # 创建物理模拟实例
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # 获取地形配置
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        # 根据地形类型创建地面
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        # 创建所有环境
        self._create_envs()

    # 创建相机
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # 摩擦系数随机化
        # 检查是否启用摩擦随机化
        if self.cfg.domain_rand.randomize_friction:
            # 仅在第一个环境时初始化
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            # 为当前环境的所有形状设置摩擦
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        # 恢复系数随机化
        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        # 返回修改后的属性
        return props
    
    # 动态物理属性刷新函数
    def refresh_actor_rigid_shape_props(self, env_ids):
        # 刷新摩擦系数
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        # 刷新恢复系数
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        # 对每个环境应用新属性
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            # 更新每个形状的属性
            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]
            # 提交修改到物理引擎
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    # 在环境创建时，提取和处理关节约束信息
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            # 位置限制
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            # 速度限制
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            # 力矩限制
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            # 从URDF中读取约束数据
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                # 读取速度和扭矩限制
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                # 计算软限制
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        
        return props

    # 随机化机器人身体的物理质量特性
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # 随机化有效载荷质量
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
        # 随机化重心位置
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
        # 随机化链接质量
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]
        # 返回修改后的属性
        return props
    
    # 回调函数，在每个物理仿真步后自动调用
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 定期重新采样命令
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        # 根据目标方向计算角速度命令
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)
        # 测量地形高度
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # 随机推动机器人
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        # 随机扰动机器人
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """ 倒地恢复任务：速度命令清零（不做速度追踪），只采样目标站立高度指令。

        Args:
            env_ids (List[int]): 需要重采样的环境 ID
        """
        # 速度命令（前 3 维）恒为 0：恢复任务只要求原地站起，不做任何速度追踪
        self.commands[env_ids, 0:3] = 0.
        # 第 4 维（heading）也清零
        if self.cfg.commands.num_commands >= 4:
            self.commands[env_ids, 3] = 0.
        # 高���指令：从高度范围随机采样（网络据此恢复到指定高度）
        if self.cfg.commands.num_commands >= 5:
            self.commands[env_ids, 4] = torch_rand_float(
                self.command_ranges["range_height"][0],
                self.command_ranges["range_height"][1],
                (len(env_ids), 1),
                device=self.device
            ).squeeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        # 缩放动作
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *=self.cfg.control.hip_reduction
        # 计算目标关节位置
        self.joint_pos_target = self.default_dof_pos + actions_scaled

        # 获取控制类型
        control_type = self.cfg.control.control_type
        # P控制
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        # V控制
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        # T控制
        elif control_type=="T":
            torques = actions_scaled
        else:
            # 错误处理
            raise NameError(f"Unknown controller type: {control_type}")
        # 扭矩限制
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # 环境重置时，随机初始化机器人的关节位置和速度
    def _reset_dofs(self, env_ids):
        """ 关节在 default 附近小范围扰动，模拟倒地时腿的姿势（避免超出关节限位）。 """
        num = len(env_ids)
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(
            -0.3, 0.3, (num, self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _reset_root_states(self, env_ids):
        """ 随机倒地初始化（带难度课程）。

        roll/pitch 的随机范围由难度课程决定（成功率驱动，回退到时间线性兜底）；
        yaw 用小范围（恢复任务不关心朝向）；
        z 随倒地姿态取合理贴地高度，避免穿地。
        """
        num = len(env_ids)
        # ===== 倒地难度课程 =====
        # 当前全局难度 cur_angle（所有环境共享一个角度上限）由两种方式驱动：
        #   1) 成功率驱动（默认）：用滚动窗口成功率，>up 升一档，<down 降一档。
        #      避免原"纯时间线性"在策略还没学会时硬推到最高难度导致抽搐/取巧。
        #   2) 时间线性兜底：当成功率课程关闭、或未收集到足够 episode 时，回退到按
        #      common_step_counter 线性增长，保证难度无论如何都会推进（防卡死）。
        angle_init = self.cfg.rewards.fall_angle_init
        angle_final = self.cfg.rewards.fall_angle_final
        if getattr(self.cfg.rewards, 'recovery_curriculum', False) and len(getattr(self, 'recent_success_flags', [])) >= self.cfg.rewards.success_rate_window:
            cur_angle = self.fall_angle_current
        else:
            # 兜底：时间线性（早期没足够 episode 评估成功率时）
            steps = self.cfg.rewards.fall_angle_curriculum_steps
            t = min(1.0, self.common_step_counter / max(steps, 1))
            cur_angle = angle_init + t * (angle_final - angle_init)
            self.fall_angle_current = cur_angle   # 同步给成功率课程做基线

        self.root_states[env_ids, :3] = self.base_init_state[:3]
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (num, 2), device=self.device)
        # z 随倒地程度取贴地高度：倒得越狠（cur_angle 越大）高度越低
        z_hi = 0.20
        z_lo = 0.10
        self.root_states[env_ids, 2] = torch_rand_float(z_lo, z_hi, (num, 1), device=self.device).squeeze(1)
        # roll/pitch 在课程范围内随机（倒地姿态），yaw 小范围（恢复不关心朝向）
        roll = torch_rand_float(-cur_angle, cur_angle, (num, 1), device=self.device).squeeze(1)
        pitch = torch_rand_float(-cur_angle, cur_angle, (num, 1), device=self.device).squeeze(1)
        yaw = torch_rand_float(-0.5, 0.5, (num, 1), device=self.device).squeeze(1)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        self.root_states[env_ids, 3:7] = quat
        self.root_states[env_ids, 7:13] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_recovery_curriculum(self, env_ids):
        """ 成功率驱动的倒地难度课程。

        在 reset_idx 中调用（已有 ever_success_buf 统计）。
        维护一个滚动窗口的成功标志队列 recent_success_flags（长度=success_rate_window）。
        每完成一个 episode（env reset），把"该 episode 是否曾成功达标"压入窗口。
        窗口满后，按成功率升降难度：
          - rate > success_rate_up   → fall_angle_current 上升一档
          - rate < success_rate_down → fall_angle_current 下降一档
        难度在 [fall_angle_init, fall_angle_final] 内，步长 = 区间的 1/20（20 档）。
        """
        if not getattr(self.cfg.rewards, 'recovery_curriculum', False):
            return
        if not hasattr(self, 'recent_success_flags'):
            self.recent_success_flags = []
            self.fall_angle_current = self.cfg.rewards.fall_angle_init
            self.angle_step = (self.cfg.rewards.fall_angle_final - self.cfg.rewards.fall_angle_init) / 20.0
        # 把本批 reset 的 episode 成败标志压入滚动窗口
        if len(env_ids) > 0:
            flags = self.ever_success_buf[env_ids].cpu().tolist()
            self.recent_success_flags.extend(flags)
        window = self.cfg.rewards.success_rate_window
        # 只保留最近 window 个
        if len(self.recent_success_flags) > window:
            self.recent_success_flags = self.recent_success_flags[-window:]
        # 窗口未满不评估（避免早期样本太少误判）
        if len(self.recent_success_flags) < window:
            return
        rate = sum(self.recent_success_flags) / len(self.recent_success_flags)
        if rate > self.cfg.rewards.success_rate_up:
            self.fall_angle_current = min(self.fall_angle_current + self.angle_step,
                                          self.cfg.rewards.fall_angle_final)
        elif rate < self.cfg.rewards.success_rate_down:
            self.fall_angle_current = max(self.fall_angle_current - self.angle_step,
                                          self.cfg.rewards.fall_angle_init)


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        # 读取配置参数
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        # 设置随机XY速度
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        # 应用到物理引擎
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # 随机施加持续的扰动力到机器人躯干
    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        # 生成随机扰动向量
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        # 存储扰动到缓冲区
        self.disturbance[:, 0, :] = disturbance
        # 应用到物理引擎
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    # ============ 自动恢复（倒地后自动恢复到默认位置）============
    def _check_and_recover(self):
        """ 检测倒地并自动恢复到默认站立位置（保留当前 xy 位置）。

        工作流程：
        1. 通过 projected_gravity 判断是否倒地（roll/pitch 过大）
        2. 用 fall_counter 进行防抖确认（连续 N 步才算真正倒地）
        3. 恢复：重置 root state（原地站立）+ dof（默认角度）+ 速度清零
        4. 设置恢复冷却，防止短时间内重复触发
        5. 返回本次刚恢复的环境 ID（用于在 check_termination 中屏蔽终止）
        """
        self.just_recovered[:] = False  # 重置"刚恢复"标记

        # 未启用恢复功能，直接返回
        if not getattr(self.cfg, 'recovery', None) or not self.cfg.recovery.enable_recovery:
            return

        # 倒地判定：重力投影在 xy 平面的分量绝对值大于阈值
        fall_threshold = self.cfg.recovery.fall_threshold
        tilted = torch.any(torch.abs(self.projected_gravity[:, :2]) > fall_threshold, dim=1)

        # 冷却期内的环境不计入倒地（防止刚恢复又触发）
        in_cooldown = self.recovery_cooldown > 0
        tilted = tilted & ~in_cooldown

        # 防抖：连续倒地步数累加
        self.fall_counter[tilted] += 1
        self.fall_counter[~tilted] = 0

        # 递减冷却计数
        self.recovery_cooldown = (self.recovery_cooldown - 1).clamp(min=0)

        # 真正倒地：连续倒地步数 >= recovery_delay
        recovery_delay = self.cfg.recovery.recovery_delay
        truly_fallen = self.fall_counter >= recovery_delay

        # 没有倒地的环境，直接返回
        fallen_ids = truly_fallen.nonzero(as_tuple=False).flatten()
        if len(fallen_ids) == 0:
            return

        # ===== 执行恢复 =====
        # 1) 恢复 root state：保留 xy，z 提升到默认高度，姿态归正，速度清零
        init_z = self.base_init_state[2].item()
        self.root_states[fallen_ids, 2] = init_z                      # z 回到默认高度
        self.root_states[fallen_ids, 3:7] = self.base_init_state[3:7] # 四元数归正（朝向重置）
        self.root_states[fallen_ids, 7:13] = 0.                       # 线/角速度清零

        # 2) 恢复 dof：关节位置回到默认，速度清零
        self.dof_pos[fallen_ids] = self.default_dof_pos
        self.dof_vel[fallen_ids] = 0.

        # 3) 提交到物理引擎（root state）
        env_ids_int32 = fallen_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
        # 4) 提交到物理引擎（dof state）
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        # 5) 清除这些环境的动作历史，避免恢复后动作跳变
        self.actions[fallen_ids] = 0.
        self.last_actions[fallen_ids] = 0.
        self.last_last_actions[fallen_ids] = 0.
        self.last_dof_vel[fallen_ids] = 0.
        self.feet_air_time[fallen_ids] = 0.

        # 6) 重置计数器与冷却
        self.fall_counter[fallen_ids] = 0
        self.recovery_cooldown[fallen_ids] = self.cfg.recovery.min_recovery_interval

        # 7) 标记"刚恢复"，用于 check_termination 屏蔽终止
        self.just_recovered[fallen_ids] = True

        # 控制台提示（便于调试，大规模训练时可注释掉）
        if self.num_envs <= 100:
            print(f"[Recovery] {len(fallen_ids)} env(s) recovered to default stance.")

    # terrain 的课程学习
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        # 跳过初始化
        if not self.init_done:
            # don't change on initial reset
            return
        # 计算移动距离
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        # 升级到更难地形
        # 如果机器人从起点走了超过地形长度的一半
        # 说明机器人有足够的能力应对当前难度
        # 升级到更难的地形
        move_up = distance > self.terrain.env_length / 2
        
        # robots that walked less than half of their required distance go to simpler terrains
        # 降级到更简单地形
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        # 调整地形难度       
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        # 循环和边界处理
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        # 更新起始位置
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    # 命令难度的动态调整
    # 高速环境（前20%）：
    #   ├─ 地形：简单（平坦）
    #   ├─ 目的：快速学习高速运动
    #   └─ 期望：快速达到 >80% 性能，升级命令

    # 低速环境（后80%）：
    #   ├─ 地形：困难（复杂）
    #   ├─ 目的：精细控制、复杂地形适应
    #   └─ 期望：逐步提高速度要求
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # 布尔掩码
        low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        # 索引提取
        low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # 性能评估
        if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            # 扩展X方向速度范围
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)


    # def _get_noise_scale_vec(self, cfg):
    #     """ Sets a vector used to scale the noise added to the observations.
    #         [NOTE]: Must be adapted when changing the observations structure

    #     Args:
    #         cfg (Dict): Environment config file

    #     Returns:
    #         [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
    #     """
    #     # noise_vec = torch.zeros_like(self.obs_buf[0])\
    #     if self.cfg.terrain.measure_heights:
    #         noise_vec = torch.zeros(9 + 3*self.num_actions + 187, device=self.device)
    #     else:
    #         noise_vec = torch.zeros(9 + 3*self.num_actions, device=self.device)
    #     self.add_noise = self.cfg.noise.add_noise
    #     noise_scales = self.cfg.noise.noise_scales
    #     noise_level = self.cfg.noise.noise_level
    #     noise_vec[0:3] = 0. # commands
    #     noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    #     noise_vec[6:9] = noise_scales.gravity * noise_level
    #     noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    #     noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    #     noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0. # previous actions
    #     if self.cfg.terrain.measure_heights:
    #         noise_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

    #     # 高度
    #     if self.cfg.commands.num_commands >= 5:
    #         noise_vec[45] = 0.01 * noise_level  # 小噪声
    #     #noise_vec[232:] = 0
    #     return noise_vec

    # 噪声缩放向量生成函数
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # 这个噪声向量是用于 compute_observations 中的 current_obs
        # 结构：命令(3) + 角速度(3) + 重力(3) + dof_pos(12) + dof_vel(12) + actions(12) + [高度指令(1)]
        
        if self.cfg.terrain.measure_heights:
            # 完整的特权观测噪声向量（包括高度测量）
            noise_vec = torch.zeros(9 + 3*self.num_actions + 187, device=self.device)
        else:
            # 基础观测噪声向量
            noise_vec = torch.zeros(9 + 3*self.num_actions, device=self.device)
        
        # 获取噪声参数
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # 命令部分噪声
        noise_vec[0:3] = 0.  # commands
        # 角速度部分
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # 重力向量部分
        noise_vec[6:9] = noise_scales.gravity * noise_level
        
        # 关节噪声
        noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0.  # previous actions
        
        # 高度测量噪声
        if self.cfg.terrain.measure_heights:
            noise_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)] = (
                noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
            )
        
        # 高度指令（第 46 维，索引 45）噪声位的处理说明：
        # compute_observations 只对前 45 维 current_obs 加噪（取 noise_scale_vec[0:45]），
        # 高度指令作为确定性目标值不加噪，因此这里的第 46 位噪声实际未被使用。
        # 保留扩展逻辑仅为维度对齐与未来可配置性，如需对高度指令加噪可在此启用。
        if self.cfg.commands.num_commands >= 5:
            # 如果没有高度测量，高度指令在位置 45
            if not self.cfg.terrain.measure_heights:
                # 扩展 noise_vec 到 46 维（第 46 位当前未使用）
                noise_vec_extended = torch.zeros(46, device=self.device)
                noise_vec_extended[:45] = noise_vec
                noise_vec_extended[45] = 0.01 * noise_level  # 高度指令噪声（当前未使用）
                noise_vec = noise_vec_extended
            else:
                # 如果有高度测量，高度指令在 232 位后面
                noise_vec_extended = torch.zeros(len(noise_vec) + 1, device=self.device)
                noise_vec_extended[:-1] = noise_vec
                noise_vec_extended[-1] = 0.01 * noise_level  # 高度指令噪声（当前未使用）
                noise_vec = noise_vec_extended
        
        return noise_vec

    #----------------------------------------
    # 初始化时，分配和准备所有必要的张量
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        # 躯干位置，旋转，速度
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # 关节位置，速度
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 接触力
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # 每个刚体的状态
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # 刷新张量
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # 创建张量视图和包装器
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        # 分离位置，速度
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        # 提取躯干四元数
        self.base_quat = self.root_states[:, 3:7]
        # 提取足端的位置，速度
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        # 提取接触力
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on.
        # 初始化杂项变量
        self.common_step_counter = 0            # 全局步计数器
        self.extras = {}                        # 字典
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # 噪声缩放向量
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1)) # 重力向量
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1)) # 前向向量
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # 所有关节的扭矩
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)    # P 控制增益
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)    # D 控制增益
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # 当前动作
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # 上一个动作
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # 上上动作
        self.last_dof_vel = torch.zeros_like(self.dof_vel) # 上一个关节速度
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13]) # 上一个躯干速度
        # 初始化命令张量
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        # 初始化足空相时间张量
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False) # 脚空相时间
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False) # 上一个接触状态
        # 初始身体运动
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 线速度
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) # 角速度
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)      # 重力
        # 初始化地形高度
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.base_height_points = self._init_base_height_points()

        # joint positions offsets and PD gains
        # 初始化默认关节位置，PD增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]      # 默认关节角度
            self.default_dof_pos[i] = angle                             # 存储位置
            found = False
            # 匹配PD增益
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        # 重塑为[1,num_dof]
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        
        #randomize kp, kd, motor strength
        # 初始化随机因子张量
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) # P 控制增益因子
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) # D 控制增益因子
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) # 所有电机强度随机化
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) # 有效载荷质量随机化
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # 重心位移
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False) # 扰动
        
        # 条件随机化
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        # 初始化摩擦系数和恢复系数
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # 自动恢复相关缓冲区（倒地检测与恢复冷却）
        self.fall_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)       # 连续倒地步数计数
        self.recovery_cooldown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 恢复冷却计数
        self.just_recovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)     # 本步刚恢复的环境标记
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)        # 是否成功恢复（站立达标）
        # 本 episode 内是否曾经成功达标：取消"成功即终止"后，统计成功率不能只看最后一步，
        # 需要在每步用 |= 累积，reset 时清零。
        self.ever_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)    # 本 episode 曾成功达标
        # 本 episode 内是否已经领过 recovery_success 大奖励：用于把"成功重奖"改成每 episode 只发一次，
        # 阻断"恢复→倒下→恢复"刷分循环。check_termination 成功时置位，reset 时清零。
        self.already_succeeded_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 本 episode 已领过成功重奖
        # "平稳到达"门槛用的近期扰动峰值缓冲：每步 = max(自身×decay, 当前 agitation)。
        # 用于 recovered 判定，排除翻滚/腾空作弊到达。agitation = ang_vel_xy² + lin_vel_z²。
        self.recent_max_agitation = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # 奖励函数准备函数
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        # 移除零权重奖励
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        # 准备奖励函数列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            # 跳过终止奖励
            if name=="termination":
                continue
            # 记录奖励函数名称
            self.reward_names.append(name)
            name = '_reward_' + name
            # 动态获取函数
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        # 创建奖励累计张量
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    # 创建平面
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    # 创建高度场    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    # 创建三角形网格
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    # 创建环境
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # 确定资源路径
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        # 设置资源选项
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        # 加载资源并获取属性
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        # 提取体名和关机名
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        # 初始化质量存储
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        # 设置初始状态
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        # 获取环境原点
        self._get_env_origins()
        # 初始化环境范围
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        # 初始化域随机化参数
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        # 每个环境创建机器人   
        for i in range(self.num_envs):
            # create env instance
            # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # 设置起始位置
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            # 处理，应用形状属性
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            # 创建actor
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            # 处理，应用关节属性
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            
            # 处理刚体属性
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
            # 处理和应用刚体属性
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 索引足端
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        # 索引惩罚接触点
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
        # 索引终止接触点
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            
    # 环境原点设置函数
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    # 配置解析函数
    def _parse_cfg(self, cfg):
        # 计算有效时间步
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        # 提取观测缩放因子
        self.obs_scales = self.cfg.normalization.obs_scales
        # 提取奖励缩放因子
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # 提取命令范围
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # 验证课程学习配置
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        # 提取最大episode长度
        self.max_episode_length_s = self.cfg.env.episode_length_s
        # 计算最大episode长度
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # 转换推送间隔
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # 调试可视化函数
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    # 高度采样点初始化函数
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    # 躯干高度采样点初始化函数
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    # 地形高度测量函数
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        # 如果不测量高度，直接返回全零（避免使用未定义的 num_height_points）
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device, requires_grad=False)  # 返回形状 (num_envs,) 的零张量，或根据需要调整为 (num_envs, 1)

        if self.cfg.terrain.mesh_type == 'plane':
            # 如果 num_height_points 已定义，则使用；否则返回零（但由于上面检查，通常已定义）
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    # 躯干高度测量函数
    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height

    # 脚部高度测量函数    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height

    # ============ 倒地恢复奖励/惩罚函数 ============
    # 设计原则（标准腿足 RL reward 结构）：
    #   - 少数正奖励做驱动（跟踪目标：正立、高度、站姿、成功）
    #   - 多个负惩罚做约束（约束不期望行为：倒立、抖动、速度、力矩、关节速度等）
    #   - 惩罚一律用线性/平方形式（梯度随违规增大），不用 exp（exp 在大违规处梯度归零）
    #   - 惩罚项不加 upright gate：约束应对全过程生效；正奖励才视情况加 gate
    #
    # ====================================================
    # 正奖励（驱动项）
    # ====================================================
    def _reward_upright_linear(self):
        # 正立驱动（主）：(1 - grav_z)/2，grav_z 从 +1(倒立)→-1(正立) 时奖励从 0→1 线性增长。
        # 恒定梯度，覆盖倒地→正立整条路径，是恢复过程最主要的正向路标。
        grav_z = self.projected_gravity[:, 2]
        return (1.0 - grav_z) * 0.5

    def _reward_stand_height(self):
        # 高度驱动：仅在接近正立时生效，把身体从半蹲拉到完整站姿。
        # gate 必要：否则躺平贴地(z≈0.15)+低目标高度(0.2)时误差小、奖励近满分，形成"贴地"局部最优。
        base_height = self.root_states[:, 2]
        target_height = self.commands[:, 4]
        upright_gate = (self.projected_gravity[:, 2] < -0.7).float()
        return torch.exp(-torch.square(base_height - target_height) / 0.05) * upright_gate

    def _reward_joint_to_default(self):
        # 站姿驱动：奖励关节接近 default。err 用均值（不求和）避免梯度过早消失。
        # sigma 从 1.0 收紧到 0.25：原版在 default 附近太平（偏 0.2rad 就给 0.96 分），
        # 策略没动力精修；收紧后 default 附近梯度变陡，逼策略把最后 0.1~0.2 rad 推到位。
        err = torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return torch.exp(-err / 0.25)

    def _reward_joint_to_default_penalty(self):
        # default 近处精修惩罚：补 _reward_joint_to_default 在 default 附近梯度太平的缺陷。
        # |·| 形式（近处梯度陡、远处不至于压垮翻身），返回正值，config 里 scale 为负即惩罚。
        # upright gate 收到 -0.9：仅在几乎完全正立（倾斜<26°）时才触发，避免翻身末段把腿过早拽到
        # 站立 default 姿态（身体此时还没平衡）导致前扑头着地。早期用 -0.7（45°）触发过猛。
        upright_gate = (self.projected_gravity[:, 2] < -0.9).float()
        err = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        return err * upright_gate

    def _reward_recovery_success(self):
        # 每 episode 首次成功才给大奖励（×10），防止"恢复→倒下→恢复"刷分循环。
        # 已用 already_succeeded_buf 记录"本 episode 是否已领过"，领过就不再发。
        # success_buf 由 check_termination 每步更新（当前步是否达标）；
        # already_succeeded_buf 在 check_termination 里成功时置位、reset 时清零。
        first_success = self.success_buf & ~self.already_succeeded_buf
        return first_success.float() * 10.0

    # ====================================================
    # 负惩罚（约束项，全程生效，线性/平方形式）
    # ====================================================
    def _reward_upside_down_penalty(self):
        # 倒立惩罚：grav_z>0（身体朝下）时负分，完全倒立时 -1。
        grav_z = self.projected_gravity[:, 2]
        return -torch.clamp(grav_z, min=0.0)

    def _reward_action_rate(self):
        # 动作变化率惩罚（sim2real 核心）：抑制电机指令高频跳变。
        # 全程生效，压制真机无法跟随的抖动式动作。
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_action_rate_upright(self):
        # 动作变化率惩罚（正立门控加强版）：翻身阶段只用基础 action_rate 轻约束，
        # 正立后叠加重罚，逼策略在已经站稳时停止抖动/抽搐。
        upright_gate = (self.projected_gravity[:, 2] < -0.7).float()
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1) * upright_gate

    def _reward_lin_vel_xy(self):
        # 水平线速度惩罚：恢复任务要求原地站起，不应有水平漂移。
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)

    def _reward_lin_vel_z(self):
        # 垂直线速度惩罚（专打腾空翻滚）：平顺起身垂直速度≈0（脚不离地），
        # 暴力蹬地腾空翻才有大垂直速度。这是区分"平顺起身"与"腾空作弊"最干净的量——
        # 起身本身几乎不产生垂直速度，故零误伤风险。仅打"跳起来翻"。
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # roll/pitch 角速度惩罚：站起过程不应有剧烈翻滚，约束姿态稳定。
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_torques(self):
        # 力矩惩罚（sim2real 核心）：限制电机出力，防止真机过载/电流尖峰。
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_torques_upright(self):
        # 力矩惩罚（正立门控加强版）：翻身需要用力，全程重罚会咬死翻身；
        # 正立后重罚，逼策略"用最小力保持站立"而非持续大力支撑。
        upright_gate = (self.projected_gravity[:, 2] < -0.7).float()
        return torch.sum(torch.square(self.torques), dim=1) * upright_gate

    def _reward_dof_vel(self):
        # 关节速度惩罚：限制关节运动速度，减少机械磨损和真机驱动器压力。
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # 关节加速度惩罚：限制冲击，平滑关节运动。
        return torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) / self.dt

    def _reward_dof_acc_upright(self):
        # 关节加速度惩罚（正立门控加强版）：翻身蹬腿必然伴随大加速度，全程重罚会压制翻身；
        # 正立后重罚，消除站稳后的高频抖动/微冲击。
        upright_gate = (self.projected_gravity[:, 2] < -0.7).float()
        return (torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) / self.dt) * upright_gate
