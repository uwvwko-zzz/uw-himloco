# -*- coding: utf-8 -*-
"""
dog_recovery_config.py
倒地恢复（Recovery）训练专用环境配置。

目标：让机器人从各种随机倒地姿态学会自己站起来，按指令恢复到指定高度。
- 初始化：随机 roll/pitch 倒地（角度随课程递增）+ 低高度 + 随机关节
- 终止：站起来了（upright + 高度达标）即成功重置；超时则重置
- 奖励：身体恢复直立 + 高度达标 + 关节回 default + 平滑/低能耗（均归一化到 [0,1]）
- 命令：速度命令全部为 0，第 5 维为目标站立高度

观测维度：单步 46 维（命令3 + 角速度3 + 重力3 + dof_pos12 + dof_vel12 + action12 + 高度指令1）
          x 6 步历史 = 276 维
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class DogRecoveryCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        # 单步观测：45 基础 + 1 高度指令 = 46
        num_one_step_observations = 46
        num_observations = num_one_step_observations * 6          # 276：6 步历史
        num_one_step_privileged_obs = 46 + 3 + 3                  # 52：单步观测 + 线速度 + 外力
        num_privileged_obs = num_one_step_privileged_obs * 1
        episode_length_s = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        # 恢复任务：速度命令清零，只保留目标站立高度指令
        num_commands = 5
        curriculum = False
        heading_command = False
        resampling_time = 1e6

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]
            range_height = [0.2, 0.35]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]

        default_joint_angles = {
            'FL_hip_joint': -0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': 0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': -0.8,
            'RL_thigh_joint': -1.,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': 1.5,
            'RR_calf_joint': 1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 40.}
        damping = {'joint': 1.0}
        action_scale = 0.5
        decimation = 4
        hip_reduction = 1.0

    class asset(LeggedRobotCfg.asset):
        file = LEGGED_GYM_ROOT_DIR + '/resources/robots/dog/urdf/dog.urdf'
        name = 'dog'
        foot_name = 'foot'
        penalize_contacts_on = ['thigh', 'calf', 'base']
        terminate_after_contacts_on = []  # 恢复任务初始即倒地，不靠接触终止
        self_collisions = 1
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        # 恢复训练阶段关闭域随机化，先学会站起来；后续可逐步开启增强鲁棒性
        randomize_friction = False
        randomize_payload_mass = False
        randomize_com_displacement = False
        randomize_motor_strength = False
        randomize_kp = False
        randomize_kd = False
        push_robots = False
        disturbance = False
        delay = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100.
        # 成功判定/高度奖励的目标都跟随 commands[:,4]，|base_height - cmd| < tol 即达标
        stand_height_tolerance = 0.05
        # 成功判定：关节平均偏差 < tol 才算恢复成功（强制回到 default 站姿）
        # 0.3→0.2：原版 0.3 rad(17°) 太松，成功容差允许明显偏离；收紧到 0.2 rad(11°)。
        # 注意：太紧会让早期策略够不到、训练崩。0.2 是相对安全的中间值。
        recovery_joint_tol = 0.2
        # 倒地难度课程：roll/pitch 随机范围随训练进度从 init→final
        fall_angle_curriculum_steps = 5000   # 达到最大难度的迭代数
        fall_angle_init = 0.3                # 初始倒地角度上限（rad，约 17°，接近站立的小扰动）
        fall_angle_final = 3.14159           # 最终倒地角度上限（rad，π=全方向倒地）

        class scales(LeggedRobotCfg.rewards.scales):
            # ===== 倒地恢复 reward：4 正驱动 + 7 负约束（标准腿足 RL 结构）=====
            # 正奖励做驱动（跟踪目标），负惩罚做约束（抑制不期望行为）。
            # 惩罚一律线性/平方形式（梯度随违规增大），不用 exp。
            # 所有 scale 会被 _prepare_reward_function 乘以 dt(=0.02) 后生效。

            # ----- 正奖励（驱动项）-----
            upright_linear = 3.0       # 正立驱动（主路标）：线性，覆盖倒地→正立全程
            stand_height = 1.5         # 高度驱动（upright gate：仅正立时生效，把半蹲拉到完整站姿）
            joint_to_default = 1.5     # 站姿驱动：关节接近 default
            recovery_success = 1.0     # 成功重奖（每步 ×10，达标环境拿满）：破除悬停局部最优

            # ----- 负惩罚（约束项，全程生效，scale 为负）-----
            upside_down_penalty = 2.0  # 倒立惩罚：grav_z>0 时负分（函数内已取负，此处 scale 为正）
            joint_to_default_penalty = -0.5  # default 近处精修惩罚（upright gate -0.9，温和）：从 -2.0 软化，避免 gate 触发后把腿过早拽到站立姿态导致前扑头着地
            # 以下 smoothing 项回退到基线值（之前加大 10-400 倍导致翻身发力被压制、头着地，回退后单独重训验证）
            # 第三组：只把 action_rate 从 -0.1 加到 -0.2（翻一倍，最小幅度），其余 smoothing 保持基线。
            # 一次只动一项 + 小幅，避免再犯"一次加大 400 倍导致崩"的错误。验证 OK 后下一轮再加 torques/dof_acc。
            action_rate = -0.2         # 动作变化率²惩罚（基线 -0.1 → 小幅加强到 -0.2）
            lin_vel_xy = -0.5          # 水平线速度²惩罚：恢复任务要求原地站起，不应漂移
            ang_vel_xy = -0.05         # roll/pitch 角速度²惩罚：约束姿态稳定
            torques = -1e-5            # 力矩²惩罚（基线）
            dof_vel = -1e-4            # 关节速度²惩罚（基线）
            dof_acc = -2.5e-7          # 关节加速度²惩罚（基线）

            # ----- 必须显式置 0：屏蔽基类 LeggedRobotCfg.rewards.scales 的非零行走任务默认值。
            # 这些项在本任务没有对应的 _reward_ 函数，若继承基类非零值会导致
            # _prepare_reward_function 中 getattr 报 AttributeError。-----
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            base_height = 0.0
            orientation = 0.0
            stand_still = 0.0


class DogRecoveryCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'dog_recovery'
        max_iterations = 5000
        save_interval = 100
