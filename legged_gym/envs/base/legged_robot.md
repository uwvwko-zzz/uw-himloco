LeggedRobot类的方法一览

初始化入口
def __init__() —— 初始化入口。解析配置、创建仿真/地形/环境、初始化 PyTorch 缓冲区、准备奖励函数

主训练循环
def step() —— 主训练循环，裁剪动作 → 动作延迟 → decimation 次物理步（计算扭矩→施加力→模拟）→ post_physics_step() → 裁剪观测 → 返回
    
物理步后处理
def post_physics_step() —— 刷新状态张量 → 提取四元数/速度/重力/足端信息 → _post_physics_step_callback() → 检查终止 → 计算奖励 → 计算终止观测 → 重置环境 → 重新计算观测 → 清除扰动 → 更新历史

是否重置
def check_termination() 

重置指定环境
def reset_idx() —— 更新地形/命令课程 → _reset_dofs() → _reset_root_states() → _resample_commands() → 清空历史 buffer → 重新测量高度 → Domain Randomization（Kp/Kd/电机强度/摩擦/恢复系数）→ 记录日志

计算奖励
def compute_reward() —— 计算奖励,遍历所有非零权重的奖励函数，累加到 rew_buf 和 episode_sums

构建观测向量
def compute_observations() 

返回单步观测
def get_current_obs() 

计算终止时刻的特权观测
def compute_termination_observations()

创建物理仿真
def create_sim() —— 创建物理仿真实例、地形（heightfield/trimesh/plane）、调用 _create_envs() 创建所有并行环境

设置相机位置
def set_camera()

环境创建回调
def _process_rigid_shape_props() —— 随机化摩擦系数和恢复系数

动态物理属性刷新函数
def refresh_actor_rigid_shape_props() —— 环境重置时重新随机化摩擦/恢复系数并应用到物理引擎

在环境创建时，提取和处理关节约束信息
def _process_dof_props() —— 环境创建回调：从 URDF 提取关节位置/速度/扭矩限制，计算软限制

随机化机器人身体的物理质量特性
def _process_rigid_body_props() —— 环境创建回调：随机化有效载荷质量、重心位置、链接质量

回调函数，在每个物理仿真步后自动调用
def _post_physics_step_callback() —— 物理步回调：定期重采样命令 → 计算航向角速度命令 → 测量地形高度 → 随机推力/扰动力

def _resample_commands() —— 随机采样新命令。前 20% 环境为"高速组"（大 X 速度，Y=0），后 80% 为"低速组"（小速度）。小命令（<0.2）置零。如果 num_commands >= 5，额外采样高度指令

扭矩计算函数
def _compute_torques() —— 将神经网络输出转换为电机扭矩。支持 P 控制（PD位置）、V 控制（速度）、T 控制（直接扭矩）。Hip 关节有额外减速比缩放。最终 clip 到扭矩限制

环境重置时，随机初始化机器人的关节位置和速度
def _reset_dofs()

重置机器人身体的位置和速度
def _reset_root_states() 

随机给机器人施加一个横向推力
def _push_robots() 

随机施加持续的扰动力到机器人躯干
def _disturbance_robots() 

terrain 的课程学习
def _update_terrain_curriculum() —— 地形课程学习。行走距离 > 地形长度一半 → 升级；行走距离 < 命令距离×50% → 降级。到达最高级后随机重置

命令难度的动态调整
def update_command_curriculum() —— 命令课程学习：当高速组和低速组的线速度追踪奖励都 > 80% 最大值时，扩展 X 方向速度范围 ±0.2

噪声缩放向量生成函数
def _get_noise_scale_vec() ——  生成观测噪声缩放向量。命令不加噪声，角速度/重力/关节位置/关节速度/高度测量按配置比例加噪声，高度指令加微小噪声

初始化时，分配和准备所有必要的张量
def _init_buffers() —— 初始化所有 PyTorch 张量：状态视图（root_states, dof_pos, dof_vel 等）、默认关节位置、PD 增益、命令、历史 buffer、Domain Randomization 因子、噪声向量

奖励函数准备函数
def _prepare_reward_function() —— 初始化时扫描 cfg.rewards.scales，将非零项 × dt，动态查找 self._reward_<name> 方法，构建函数列表

创建地面平面
def _create_ground_plane()

创建高度场
def _create_heightfield()

创建三角形网格
def _create_trimesh()

创建所有并行环境
def _create_envs() ——加载 URDF 资产 → 为每个环境创建实例 → 应用刚体/关节属性回调 → 创建 actor → 索引足端/惩罚接触/终止接触刚体

环境原点设置函数
def _get_env_origins() 

配置解析函数
def _parse_cfg()

调试可视化函数
def _draw_debug_vis()

高度采样点初始化函数
def _init_height_points()

躯干高度采样点初始化函数
def _init_base_height_points()

地形高度测量函数
def _get_heights()

躯干高度测量函数
def _get_base_heights()

脚部高度测量函数    
def _get_feet_heights()

然后就是奖励函数了