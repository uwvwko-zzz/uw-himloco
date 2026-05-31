from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class DogRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m] # 0.42

        # 这里就是假设实物的角度已经在urdf中的0位了，他到default的一个转角的正负
        # 模型里面的电机  值越大，往顺时针转
        # 实际的电机  值越大，往顺时针转
    

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': -0.1,   # [rad] 
            'RL_hip_joint': 0.1,   # [rad]                  
            'FR_hip_joint': 0.1 ,  # [rad]                     
            'RR_hip_joint': -0.1,   # [rad]                      

            'FL_thigh_joint': -0.8,     # [rad]               
            'RL_thigh_joint': -1.,   # [rad]                    
            'FR_thigh_joint': 0.8,     # [rad]                
            'RR_thigh_joint': 1.,   # [rad]             


            # -1.78  虽然说这个关节rviz显示的0,但是实际上是-1.78
            'FL_calf_joint': -1.5,   # [rad]                    
            'RL_calf_joint': -1.5,    # [rad]            
            'FR_calf_joint': 1.5,  # [rad]                      
            'RR_calf_joint': 1.5,    # [rad]                   
        }


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # P控制类型
        control_type = 'P'
        # 刚度
        stiffness = {'joint': 40.}          # 20   
        # stiffness = {'joint': 20.}  
        # 阻尼
        damping = {'joint': 1.0}            # 0.5  
        # damping = {'joint': 0.5} 

        # 动作缩放系数
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
    
        # 策略每决策 1 次，同一个动作在物理引擎中执行 4 次
        decimation = 4

        # 髋关节缩放因子
        hip_reduction = 1.0                 


    class commands( LeggedRobotCfg.commands ):
            curriculum = True
            # 课程学习的最大难度倍数
            max_curriculum = 2.0
            # 命令维度数
            # [0] lin_vel_x：前进速度
            # [1] lin_vel_y：侧移速度
            # [2] ang_vel_yaw：旋转速度
            # [3] heading：目标朝向
            # [4] range_height：（自己新增）

            # num_commands = 5 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            num_commands = 5

            # 命令更新间隔，每 10 秒重新采样一个新命令
            resampling_time = 10. # time before command are changed[s]

            # 朝向命令模式开关
            class ranges( LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-1.0, 1.0]        # 前进速度范围
                lin_vel_y = [-1.0, 1.0]        # 侧移速度范围
                ang_vel_yaw = [-3.14, 3.14]    # 旋转速度范围
                heading = [-3.14, 3.14]        # 目标朝向范围
                range_height = [0.2, 0.35]    # 身体高度范围   0.25-0.3


    class asset( LeggedRobotCfg.asset ):
        file = '/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/resources/robots/dog/urdf/dog_1.urdf'
        name = "dog"             
        foot_name = "foot"            

        # penalize_contacts_on = ["thigh", "calf"]  
        # terminate_after_contacts_on = ["base"]   
        # self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # flip_visual_attachments = False 

        # 机器人与环境接触这些部位时，给予负奖励
        penalize_contacts_on = ["thigh", "calf", "base"]
        # 当这些部位与环境接触时，终止当前 episode
        terminate_after_contacts_on = ["base"]
        # 特权观测中包含的接触信息
        privileged_contacts_on = ["base", "thigh", "calf"]
        # 自碰撞检测开关，1禁用，0启用
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # 坐标系翻转开关
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
  


    class rewards( LeggedRobotCfg.rewards ):
        # 允许负奖励
        only_positive_rewards = False

        # 速度追踪的高斯分布方差，值越小，对速度误差的惩罚越严厉
        tracking_sigma = 0.25

        # 软 DOF 状态限制
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.

        # 目标高度
        base_height_target = 0.35               # 0.35
        # 最大允许接触力
        max_contact_force = 100.
        # 抬腿高度目标，-0.20 = 20 厘米离地
        clearance_height_target = -0.20
        
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            # 追踪线性速度有奖励
            tracking_lin_vel = 1.0      # 1.0
            # 角速度追踪奖励
            tracking_ang_vel = 0.5
            # 竖直速度惩罚
            lin_vel_z = -1.0            # -1.0
            # 水平面滚转/俯仰速度惩罚
            ang_vel_xy = -0.05
            # 身体朝向惩罚 
            orientation = -0.
            # 关节加速度惩罚
            dof_acc = -2.5e-7
            # 关节加速度惩罚
            joint_power = -2e-5
            # 身体高度偏离惩罚 
            base_height = 0.5          # -1.0(45)    0.5(46)
            # 脚部间隙惩罚
            foot_clearance = -0.01
            # 动作变化率惩罚
            action_rate = -0.01
            # 动作平滑性惩罚
            # 腾空相奖励
            feet_air_time =  1.0
            
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -1.0
            # 新增奖励函数（参考 OpenDoge_train）
            smoothness = -0.02
            diagonal_sync = -0.15
            hip_mirror_symmetry = -0.1
            default_pos_linear = -0.05
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0

class DogRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'dog_rough'  # 🔁 建议改为对应名称