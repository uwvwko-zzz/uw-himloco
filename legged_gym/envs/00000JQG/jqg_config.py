from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class JQGRoughCfg(LeggedRobotCfg):
    """00000JQG 四足机器人崎岖地形训练配置。"""

    class init_state(LeggedRobotCfg.init_state):
        # JQG 零位时大小腿基本向下，轻微屈膝后的足端高度约 0.52 m。
        pos = [0.0, 0.0, 0.45]

        # action=0 时的关节目标角。小腿符号按 00000JQG.urdf 的
        # 左侧正角、右侧负角限位设置。
        default_joint_angles = { # = target angles [rad] when action = 0.0
                    'FL_hip_joint': -0.1,   # [rad] 
                    'RL_hip_joint': 0.1,   # [rad]                  
                    'FR_hip_joint': 0.1 ,  # [rad]                     
                    'RR_hip_joint': -0.1,   # [rad]                      
        
                    'FL_thigh_joint': -0.8,     # [rad]               
                    'RL_thigh_joint': -0.8,   # [rad]                    
                    'FR_thigh_joint': 0.8,     # [rad]                
                    'RR_thigh_joint': 0.8,   # [rad]             
        
        
                    # -1.78  虽然说这个关节rviz显示的0,但是实际上是-1.78
                    'FL_calf_joint': 1.5,   # [rad]                    
                    'RL_calf_joint': 1.5,    # [rad]            
                    'FR_calf_joint': -1.5,  # [rad]                      
                    'RR_calf_joint': -1.5,    # [rad]                   
                }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 100.0}
        damping = {"joint": 2.5}

        # target_angle = action_scale * action + default_joint_angle
        action_scale = 0.25
        decimation = 4
        hip_reduction = 1.0

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 2.0

        # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, range_height
        num_commands = 5
        resampling_time = 10.0

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-3.14, 3.14]
            heading = [-3.14, 3.14]
            range_height = [0.35, 0.55]

    class asset(LeggedRobotCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            "00000JQG/urdf/00000JQG.urdf"
        )
        name = "00000JQG"
        foot_name = "foot"

        # 四个 foot_Link 通过 fixed joint 与小腿连接。必须保留这些刚体，
        # 否则 Isaac Gym 会将其合并进 calf，导致 feet_indices 为空。
        collapse_fixed_joints = False

        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]

        # 1：关闭自碰撞；0：启用自碰撞。
        self_collisions = 1
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.25

        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0

        base_height_target = 0.45
        height_tracking_sigma = 0.05
        max_contact_force = 100.0
        clearance_height_target = -0.20

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.8
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.0

            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = 0.5
            foot_clearance = -0.01
            action_rate = -0.01
            feet_air_time = 1.0

            collision = -0.0
            feet_stumble = -0.0
            smoothness = -0.02

            stand_still = -1.0
            stand_four_feet = -1.0
            stand_orientation = -1.0

            diagonal_sync = -0.15
            hip_mirror_symmetry = -0.2
            default_pos_linear = -0.05

            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0


class JQGRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "jqg_rough"
