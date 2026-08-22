from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class W1WRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # 3 commands + 3 angular velocity + 3 gravity + 3 * 16 DOFs/actions
        # + 1 body-height command.
        num_actions = 16
        num_one_step_observations = 58
        num_observations = num_one_step_observations * 6
        # Critic additionally receives base velocity, disturbance and the
        # 17 x 11 terrain-height scan used by rough-terrain curriculum.
        num_one_step_privileged_obs = 58 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs

    class init_state(LeggedRobotCfg.init_state):
        # 默认腿姿态的轮底离地高度约为 0.45 m，略抬高生成以避免初始穿透。
        pos = [0.0, 0.0, 0.48]
        default_joint_angles = {
            "fl_hip_joint": 0.0,
            "fl_thigh_joint": -0.8,
            "fl_calf_joint": 1.55,
            "fl_foot_joint": 0.0,
            "fr_hip_joint": 0.0,
            "fr_thigh_joint": 0.8,
            "fr_calf_joint": -1.55,
            "ft_foot_joint": 0.0,  # Name used by the supplied URDF.
            "rl_hip_joint": 0.0,
            "rl_thigh_joint": -0.8,
            "rl_calf_joint": 1.55,
            "rl_foot_joint": 0.0,
            "rr_hip_joint": 0.0,
            "rr_thigh_joint": 0.8,
            "rr_calf_joint": -1.55,
            "rr_foot_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = "hybrid"
        stiffness = {
            "hip_joint": 80.0,
            "thigh_joint": 80.0,
            "calf_joint": 80.0,
            "foot_joint": 0.0,
            "ft_foot_joint": 0.0,
        }
        damping = {
            "hip_joint": 2.0,
            "thigh_joint": 2.0,
            "calf_joint": 2.0,
            "foot_joint": 0.0,
            "ft_foot_joint": 0.0,
        }
        action_scale = 0.25
        hip_reduction = 1.0
        wheel_action_scale = 1.5  # rad/s for a normalized action of 1.
        wheel_velocity_kp = 1.5
        wheel_joint_names = [
            "fl_foot_joint",
            "ft_foot_joint",
            "rl_foot_joint",
            "rr_foot_joint",
        ]
        # Positive policy action means forward rolling despite mirrored URDF axes.
        wheel_directions = {
            "fl_foot_joint": -1.0,
            "ft_foot_joint": 1.0,
            "rl_foot_joint": -1.0,
            "rr_foot_joint": 1.0,
        }
        decimation = 4

    class commands(LeggedRobotCfg.commands):
        num_commands = 5
        # 直接训练 yaw-rate，使移动姿态奖励的触发条件与实际命令一致。
        heading_command = False
        # 每次命令采样时，15%的环境使用严格零速度和默认站立高度。
        standing_env_ratio = 0.15

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-2.0, 2.0]
            heading = [-3.14, 3.14]
            range_height = [0.4, 0.62]

    class asset(LeggedRobotCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            "w1w_urdf1/urdf/w1w_urdf1.urdf"
        )
        name = "w1w"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False

    class terrain(LeggedRobotCfg.terrain):
        # W1W 必须在 rough terrain 上学习抬腿；只用 plane 会收敛到四轮始终
        # 贴地的局部最优，部署到台阶时没有任何腾空/落脚经验。
        mesh_type = "trimesh"
        measure_heights = True
        curriculum = True
        max_init_terrain_level = 5
        # 将最高课程等级（Level 9）明确映射到这些物理上限。
        max_slope_angle_deg = 30.0
        min_step_height = 0.05
        max_step_height = 0.30
        min_discrete_obstacle_height = 0.05
        max_discrete_obstacle_height = 0.30
        # smooth slope, rough slope, stairs up, stairs down, discrete obstacles,
        # stepping stones, T stairs, ramp, gravel, rubble, wall, speed bump
        # T stairs 和 ramp 已移除；释放的15%均分给普通上下楼梯和离散障碍。
        terrain_proportions = [
            0.05, 0.05, 0.35, 0.30, 0.15, 0.0,
            0.0, 0.0, 0.05, 0.05, 0.0, 0.0,
        ]

    class rewards(LeggedRobotCfg.rewards):
        # 训练早期策略尚不会抬腿时，多个正则惩罚会淹没速度跟踪信号。
        # 先裁掉每步负总奖励，避免 critic 长期只看到负回报而停滞。
        only_positive_rewards = True
        # 12个腿关节处于 default_joint_angles 时对应的默认机身高度。
        base_height_target = 0.45
        # 高度跟踪指数奖励的平方误差尺度；0.0025 对应约 5 cm。
        height_tracking_sigma = 0.0025
        # 轮子接触力软阈值，超过该值时可用于接触力惩罚。
        max_contact_force = 250.0
        # 实际轮速超过该阈值后，wheel_speed 奖励项开始产生惩罚。
        max_wheel_speed = 45.0
        # RobotLab DDT Tita 轮足奖励中使用的门限。
        reward_command_threshold = 0.1
        joint_pos_velocity_threshold = 100.0
        wheel_vel_velocity_threshold = 0.5
        joint_pos_stand_still_scale = 5.0

        class scales(LeggedRobotCfg.rewards.scales):
            # 非超时终止惩罚，例如机身触地造成的 episode 终止。
            termination = -0.5
            # 奖励机身 XY 线速度跟踪 vx、vy 指令。
            tracking_lin_vel = 3.0
            # 奖励机身 yaw 角速度跟踪 wz 指令。
            tracking_ang_vel = 2.0
            # 惩罚机身竖直方向速度，减少上下弹跳。
            lin_vel_z = -0.5
            # 惩罚 roll、pitch 角速度，减少横滚和俯仰抖动。
            ang_vel_xy = -0.1
            # 惩罚机身倾斜，使机身的 z 轴尽量保持朝上。
            orientation = -0.5
            # 高度平方误差惩罚，提供不会因误差较大而消失的修正梯度。
            base_height = -2.0
            # 高度指数跟踪奖励，在接近目标高度后提供更精细的跟踪信号。
            height_tracking = 1.5
            # 惩罚所有执行器的力矩平方，限制持续的大力矩输出。
            torques = -5.0e-6
            # 惩罚 |关节速度 × 关节力矩|，减少无效功率消耗。
            joint_power = -2.0e-6
            # 关闭包含全部16个关节的通用加速度惩罚，改为腿轮分开计算。
            dof_acc = 0.0
            # 12个腿关节加速度平方惩罚，抑制腿部高频机械抖动。
            leg_dof_acc = -1.0e-7
            # 4个轮关节加速度平方惩罚。轮速变化天然更大，因此权重更小。
            wheel_dof_acc = -2.5e-9
            # 惩罚相邻控制周期的动作变化，改善动作连续性。
            action_rate = -0.005
            # 惩罚动作的二阶差分，进一步减少动作突变。
            smoothness = -0.005
            # 髋部、大腿、小腿或机身与地面发生接触时惩罚。
            collision = -0.5
            # RobotLab 风格：静止时单独加强默认姿态约束。
            stand_still = -2.0
            # 旧的零指令姿态项由 stand_still + joint_pos_penalty 取代。
            zero_command_leg_default = 0.0
            # 存在移动命令时约束四个 hip-roll 回到默认零位，防止形成内八；
            # thigh/calf 不受该项限制，仍可主动抬腿越障。
            moving_hip_default = -1.0
            # RobotLab joint_pos_penalty：运动时使用 L2 范数约束腿部姿态，
            # 静止时再乘以 joint_pos_stand_still_scale。
            joint_pos_penalty = -0.5
            default_pos_linear = 0.0
            # 仅当实际轮速超过 max_wheel_speed 时惩罚超出的部分。
            wheel_speed = -2.0e-4
            # RobotLab wheel_vel_penalty：运动时只惩罚离地轮空转，
            # 静止时惩罚所有轮速。
            wheel_vel_penalty = -0.01
            # 轻量惩罚运动时轮子失去接触。平地抬轮没有额外收益，
            # 会被该项压制；越障时速度跟踪收益仍能覆盖这个小代价。
            wheel_contact_loss = -0.2
            wheel_spin_air = 0.0
            zero_command_wheel_action = 0.0
            zero_command_wheel_velocity = 0.0

            # 必须显式置零：基类默认 feet_air_time=1.0，不覆盖就会继承
            # 腾空奖励。RobotLab DDT Tita 同样关闭腾空、抬脚高度和步态奖励。
            feet_air_time = 0.0
            foot_clearance = 0.0
            # 不强制传统四足对角小跑接触时序。
            diagonal_sync = 0.0
            # 不额外使用旧的硬编码髋关节镜像奖励。
            hip_mirror_symmetry = 0.0
            # 不强制每个静止时刻都由传统“四足接触”奖励约束。
            stand_four_feet = 0.0
            # 静止姿态已由 orientation 和速度相关奖励约束。
            stand_orientation = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.9, 1.1]


class W1WRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "w1w_rough"
        run_name = ""
