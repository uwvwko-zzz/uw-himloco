from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class DogHighRoughCfg( LeggedRobotCfg ):
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
            # 60% 命令专注前后通过平台，40% 保留完整平移和转向训练。
            platform_command_ratio = 0.6

            # 朝向命令模式开关
            class ranges( LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-1.0, 1.0]        # 前进速度范围
                lin_vel_y = [-1.0, 1.0]
                ang_vel_yaw = [-3.14, 3.14]
                heading = [-3.14, 3.14]
                range_height = [0.2, 0.35]    # 身体高度范围   0.25-0.3


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog/urdf/dog_1.urdf'
        name = "dog"             
        foot_name = "foot"            

        # penalize_contacts_on = ["thigh", "calf"]  
        # terminate_after_contacts_on = ["base"]   
        # self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # flip_visual_attachments = False 

        # 机器人与环境接触这些部位时，给予负奖励
        # 注意：全部清空——爬 30cm 垂直墙时小腿必然蹬墙借力、大腿贴墙翻越，
        #       保留任何一项都会让狗不敢把腿伸到墙上去（base 同理）。
        #       若训练后出现"用膝盖硬怼墙"等坏步态，可只加回 ["thigh"] 微调。
        penalize_contacts_on = []
        # 当这些部位与环境接触时，终止当前 episode
        # 注意：base 接触不再终止——30cm 垂直薄墙必须允许躯干蹭过墙顶
        # 平台顶部足够宽，不再需要允许躯干贴墙翻越。
        terminate_after_contacts_on = []
        # 特权观测中包含的接触信息
        privileged_contacts_on = ["base", "thigh", "calf"]
        # 自碰撞检测开关，1禁用，0启用
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # 坐标系翻转开关
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
  


    class terrain( LeggedRobotCfg.terrain ):
        # 多道高墙相对块中心的 X 偏移（米），与 high_wall_terrain / _reward_wall_crossing 共用
        # ±1.5/±2.5m：避开 spawn 中心 ±1m 扰动区，每侧 2 道墙形成密集翻越训练
        wall_x_offsets = [2.0]
        # 宽顶平台沿 X 方向的长度（米）。
        platform_length = 1.0

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
        # 高度追踪的sigma，越大越宽松（蹲起时过渡期也有梯度信号）
        height_tracking_sigma = 0.05
        # 最大允许接触力
        max_contact_force = 100.
        # 分区摆腿目标：平地低抬脚，只在高台前缘附近高抬脚。
        clearance_height_target = -0.07  # 旧接口的平地默认值
        flat_clearance_height_target = -0.07
        platform_clearance_height_target = -0.18
        flat_air_time_target = 0.20
        platform_air_time_target = 0.35
        flat_air_time_reward_scale = 0.20
        terminate_on_fall = True
        min_base_height = 0.12
        
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -10.0
            alive = 0.2
            # 追踪线性速度有奖励（降权：爬墙瞬间必然掉速，原1.5会让"撞墙不动"成局部最优）
            tracking_lin_vel = 2.0
            # 角速度追踪奖励
            tracking_ang_vel = 0.5
            # 竖直速度惩罚（大幅降权：爬墙/跳墙都需要向上的 vz，原-1.0会把起跳压死）
            lin_vel_z = -0.5
            # 水平面滚转/俯仰速度惩罚
            ang_vel_xy = -0.05
            # 身体朝向惩罚
            orientation = -0.2
            # 关节加速度惩罚
            dof_acc = -2.5e-7
            # 关节加速度惩罚
            joint_power = -1e-6
            # 身体高度偏离惩罚（置0：爬墙时车顶升高，身高恒定约束与爬墙直接冲突）
            base_height = 0.0          # 0.5→0.0
            # 脚部间隙惩罚
            foot_clearance = -0.005
            # 动作变化率惩罚
            action_rate = -0.02
            # 动作平滑性惩罚
            # 腾空相奖励（提权：鼓励跳/腾空，利于上墙）
            feet_air_time = 0.3

            collision = 0.0
            feet_stumble = -0.0

            # 二阶动作平滑性，让动作变化更平稳，避免突然抖动
            smoothness = -0.005
            # 新增奖励函数

            # stand 系列降权：爬墙被减速到≈0时会误触发 stand，逼狗四脚着地/身体水平，
            # 正好和贴墙攀爬冲突（原-1.0→-0.1）
            stand_still = -0.0          # 惩罚关节偏离PD目标位置（joint_pos_target/default）
            stand_four_feet = -0.0      # 0 command 时惩罚抬脚，强制4脚着地（joint_pos_target）
            stand_orientation =  -0.0    #-0.5    # 0 command 时惩罚身体倾斜（joint_pos_target）
            
            # 对角线步态同步
            diagonal_sync = -0.02
            # 髋关节左右对称
            hip_mirror_symmetry = -0.0      # -0.2
            # 默认姿态线性惩罚
            default_pos_linear = -0.01
            
            # 惩罚电机输出力矩过大
            torques = -1e-6
            # 惩罚关节运动速度过快
            dof_vel = -0.0
            # 关节位置限位惩罚
            dof_pos_limits = -0.05
            # 关节速度限位惩罚
            dof_vel_limits = -0.0005
            # 力矩限位惩罚
            torque_limits = -0.0005
            # 过大落地冲击惩罚，改善平台上下的可部署性。
            feet_contact_forces = -1e-5
            # 稀疏成功奖励之前的连续引导：沿当前 X 命令方向前进。
            platform_progress = 0.5
            # 首次稳定登上每个平台的一次性中间奖励。
            platform_mount = 50.0
            # 越过高墙奖励（稀疏大奖励）：每越过一道墙给一次，×dt 后每道≈40分。
            # 是当前唯一的正向爬墙信号，引导策略主动翻越而非"撞墙不动刷 tracking 分"。
            # 防刷分：同一道墙每 episode 只奖一次（_reward_wall_crossing 内标记 wall_crossed）。
            wall_crossing = 100.0

class DogHighRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'dog_high_rough'  # 🔁 建议改为对应名称
