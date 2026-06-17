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
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""倒地恢复策略测试 / 可视化。

加载训练好的 dog_recovery 模型，观察机器狗能否从随机倒地姿态站起来。

用法：
    python legged_gym/scripts/play_recovery.py --task=dog_recovery
    python play_recovery.py --task=dog_recovery --model /home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_hop/logs/dog_recovery/model_2200.pt
    python legged_gym/scripts/play_recovery.py --task=dog_recovery --headless  # 仅统计成功率
"""

import isaacgym  # noqa: F401  必须最先导入，在任何引入 torch 的模块之前
import os
import sys
import numpy as np
import torch

# 确保项目根目录在 sys.path 中
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from legged_gym.envs import *  # noqa: F401,F403  触发任务注册
from legged_gym.utils import get_args, task_registry


def play(args, model_path=None, num_eval_episodes=200):
    """加载恢复策略并统计成功率。

    Args:
        args: 命令行参数（task_registry 解析）。
        model_path: 显式指定的模型文件；为 None 时自动加载最新 checkpoint。
        num_eval_episodes: headless 模式下的评估 episode 数（用于统计成功率）。
    """
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ===== 评估配置：少量环境 + 关闭随机化 + 关闭恢复自动介入 =====
    if args.headless:
        env_cfg.env.num_envs = 1024          # headless 下并行评估多个环境
    else:
        env_cfg.env.num_envs = 1             # 可视化时单环境
    env_cfg.env.episode_length_s = 10
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    # 评估时关闭自动恢复介入（测试策略本身能否站起来）
    env_cfg.recovery.enable_recovery = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera([2.5, 2.5, 1.2], [0.0, 0.0, 0.4])

    # ===== 加载策略 =====
    train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    if model_path:
        if not os.path.exists(model_path):
            print(f'[ERROR] 模型文件不存在: {model_path}')
            sys.exit(1)
        print(f'[INFO] 加载模型: {model_path}')
        ppo_runner.load(model_path)
    else:
        train_cfg.runner.resume = True
        print('[INFO] 自动加载最新 checkpoint...')
        ppo_runner.load(ppo_runner.load_path)

    policy = ppo_runner.get_inference_policy(device=env.device)

    # ===== 评估循环 =====
    obs = env.get_observations()
    num_envs = env.num_envs
    success_count = 0.0
    total_count = 0.0
    needed_episodes = num_eval_episodes if args.headless else 10**9
    # 训练侧已取消"成功即终止"，episode 跑到 timeout 才 reset。
    # 因此每个 done 都是一个完整 episode，成功判定必须是"本 episode 内是否曾达标"，
    # 而非 done 瞬间的姿态。这里维护一个 ever_success 跟踪张量，语义与训练侧一致。
    ever_success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    print(f'\n[INFO] 开始评估：{num_envs} 个环境，目标 {needed_episodes} episodes')
    print('[INFO] 成功判定：grav_z < -0.9 且 |base_height - cmd| < tol 且关节归位（episode 内任意一步达标即计成功）\n')

    step = 0
    while total_count < needed_episodes:
        actions = policy(obs.detach())
        obs, _, _, dones, infos, _, _ = env.step(actions.detach())

        # 每步判定是否达标，达标即置位（与训练成功判定完全一致）
        upright = env.projected_gravity[:, 2] < -0.9
        height_tol = env.cfg.rewards.stand_height_tolerance
        height_ok = torch.abs(env.root_states[:, 2] - env.commands[:, 4]) < height_tol
        joint_err = torch.mean(torch.abs(env.dof_pos - env.default_dof_pos), dim=1)
        joint_ok = joint_err < env.cfg.rewards.recovery_joint_tol
        ever_success |= (upright & height_ok & joint_ok)

        # 仅在 episode 结束（timeout）时统计：该 episode 内是否曾成功达标
        if torch.any(dones):
            done_mask = dones.bool()
            success_count += float(torch.sum(ever_success & done_mask).item())
            total_count += float(torch.sum(dones).item())
            ever_success[done_mask] = False   # 清零，供下一个 episode 重新统计

        if not args.headless and step % 50 == 0:
            h = env.root_states[0, 2].item()
            gz = env.projected_gravity[0, 2].item()
            status = '已站立' if (abs(gz) > 0.9 and h > 0.25) else '恢复中'
            print(f'  step={step:5d}  height={h:.3f}  grav_z={gz:+.3f}  [{status}]')

        step += 1

    if total_count > 0:
        rate = success_count / total_count * 100
        print(f'\n========== 倒地恢复评估结果 ==========')
        print(f'  总 episode 数 : {int(total_count)}')
        print(f'  成功站起来   : {int(success_count)}')
        print(f'  成功率       : {rate:.2f}%')
        print(f'======================================')
    else:
        print('[WARN] 未统计到任何 episode，请增大评估步数或检查环境配置。')


if __name__ == '__main__':
    # 在 get_args 之前解析 --model 自定义参数
    model_path = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--model' and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            sys.argv.pop(i)
            sys.argv.pop(i)
        else:
            i += 1

    args = get_args()
    play(args, model_path=model_path)
