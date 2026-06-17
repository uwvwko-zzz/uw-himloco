# Dog Recovery Gym

基于 [Isaac Gym](https://developer.nvidia.com/isaac-gym) 的四足机器人**倒地恢复（Recovery）**强化学习训练框架。

让机器狗从各种随机倒地姿态（仰面、侧翻、趴着）学会自己站起来，恢复到默认站立姿态。算法采用 [HIMLoco (Hybrid Internal Model)](https://arxiv.org/abs/2312.11460) 的 PPO + 教师-学生蒸馏架构。

## 项目结构

```
himloco_hop/
├── legged_gym/                          # 核心训练代码
│   ├── envs/
│   │   ├── base/
│   │   │   ├── base_config.py            # 配置基类
│   │   │   ├── base_task.py              # 任务基类
│   │   │   ├── legged_robot.py           # 腿式机器人环境（含倒地初始化、恢复奖励、成功判定）
│   │   │   └── legged_robot_config.py    # 腿式机器人配置（奖励、域随机化、噪声等）
│   │   └── dog/
│   │       └── dog_recovery_config.py   # Dog 倒地恢复训练配置（URDF 路径、PD参数、奖励权重）
│   ├── scripts/
│   │   ├── train.py                      # 训练入口
│   │   ├── play_recovery.py             # 恢复策略测试 / 成功率评估
│   │   └── test_angle.py                 # 默认关节角度可视化调试工具
│   ├── utils/                           # task_registry / helpers / terrain / math / logger
│   └── tests/test_env.py
├── resources/robots/dog/                # Dog 机器人 URDF / STL 网格 / MuJoCo XML
├── logs/                                # 训练日志和模型
├── setup.py
└── README.md
```

算法实现位于上级目录的 `rsl_rl/` 包：

| 文件 | 说明 |
|------|------|
| `rsl_rl/algorithms/him_ppo.py` | HIM PPO 算法 |
| `rsl_rl/modules/him_actor_critic.py` | HIM Actor-Critic 网络 |
| `rsl_rl/modules/him_estimator.py` | HIM 特权信息估计器（教师→学生蒸馏） |
| `rsl_rl/runners/him_on_policy_runner.py` | HIM 在策略训练器 |
| `rsl_rl/storage/him_rollout_storage.py` | HIM 数据存储 |

## 环境要求

- Isaac Gym Preview 4
- PyTorch（带 CUDA）
- 上级目录已安装 `rsl_rl`：`pip install -e ../rsl_rl`

## 使用方法

### 训练

```bash
# 恢复训练（headless 模式）
python legged_gym/scripts/train.py --task=dog_recovery --headless
```

模型与日志保存在 `logs/dog_recovery/<run_name>/`。

### 测试 / 可视化

```bash
# 可视化单环境恢复过程
python legged_gym/scripts/play_recovery.py --task=dog_recovery

# 指定模型文件
python legged_gym/scripts/play_recovery.py --task=dog_recovery --model logs/dog_recovery/<run>/model_XXXX.pt

# headless 批量评估成功率（1024 环境 × 200 episodes）
python legged_gym/scripts/play_recovery.py --task=dog_recovery --headless
```

### 默认姿态调试

```bash
python legged_gym/scripts/test_angle.py
```

键盘交互调整各关节角度，`P` 打印当前所有关节角度（可复制到配置的 `default_joint_angles`）。

## 恢复任务设计

### 训练目标

从随机倒地姿态学会站起来，恢复到 default 站立姿态。

### 命令空间

`commands.num_commands = 5`，向量结构为 `[lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stand_height]`。
恢复任务中前三者恒为 0（不追踪速度），`stand_height` 在每个 episode 开始时从 `[0.2, 0.35] m` 随机采样，奖励和成功判定都跟随它。

### 初始化（倒地）

每次 episode 重置时，机器人以随机姿态倒地：
- 位置：原点附近小扰动（xy ±0.3 m），高度 `0.10~0.20 m`（贴地）
- 姿态：`roll/pitch` 随机，**范围随训练进度课程递增**（`fall_angle_init=0.3 rad` → `fall_angle_final=π`，过渡 `fall_angle_curriculum_steps=5000` 步）；`yaw` 用小范围 ±0.5 rad（恢复任务不关心朝向）
- 关节：default 附近 `±0.3 rad` 随机扰动（倒地时腿的各种姿势）
- 速度：清零

### 终止条件

恢复任务初始即倒地（base/thigh/calf 必然着地），因此**不靠接触力判定失败**，否则 episode 第一步就会重置。终止只看：

| 条件 | 结果 | 说明 |
|------|------|------|
| 超时（10s） | 重置 | episode 结束（仍未站起） |
| **成功站起** | 重置 | 三条件同时满足：`grav_z < -0.9`（严格正立，避免倒立骗奖励）且 `\|base_height - cmd[4]\| < stand_height_tolerance(0.05m)` 且关节平均偏差 `< recovery_joint_tol(0.3 rad)` |

### 观测空间

| 版本 | 单步维度 | 历史步数 | 总维度 | 说明 |
|------|---------|---------|--------|------|
| 恢复任务 | 46 | 6 | 276 | 无速度命令，含目标站立高度指令 |

单步 46 维结构：

| 索引 | 维度 | 内容 |
|------|------|------|
| [0:3] | 3 | 命令（恒为 0，站立恢复） × 缩放 |
| [3:6] | 3 | 身体角速度 × 缩放 |
| [6:9] | 3 | 重力投影向量 |
| [9:21] | 12 | 关节角度偏差 (dof_pos - default) |
| [21:33] | 12 | 关节速度 × 缩放 |
| [33:45] | 12 | 上一步动作 |
| [45] | 1 | 归一化目标站立高度 `(cmd[4] - 0.25) / 0.1` |

特权观测（Critic/教师用）：52 维 = 46 单步 + 3 线速度 + 3 外力。

### 动作空间

- **维度**：12（四足 × 3 关节）
- **控制**：PD 位置控制，`target = action × 0.5 + default_angle`
- **控制频率**：50Hz（decimation=4, dt=0.005s）

### 奖励函数

除 `recovery_success`、`feet_contact` 外，其余项均以 `exp(-x²/σ)` 形式归一化到 `[0,1]`，权重为正（鼓励项）。`feet_contact` 额外用 `grav_z < -0.5` 做门控，避免初始倒地（脚都贴地）时诱导「保持倒地」。

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| upright_orientation | 2.0 | 鼓励 `grav_z → -1`（身体朝上） |
| stand_height | 1.5 | 鼓励身体高度跟随指令 `cmd[4]` |
| joint_to_default | 1.5 | 鼓励关节回到 default（强约束站姿） |
| body_orientation_flat | 0.5 | 鼓励 projected_gravity xy 分量小 |
| upright_velocity_penalty | 0.2 | 鼓励站起过程水平速度小 |
| base_upright_accel | 0.2 | 鼓励站起过程翻滚角速度小 |
| action_smoothness | 0.2 | 鼓励动作平滑（与上一步差异小） |
| energy | 0.1 | 鼓励低能耗（扭矩×速度小） |
| recovery_success | 1.0 | 成功站起一次性大奖励（函数内再 ×10） |
| feet_contact | 0.3 | 鼓励脚着地提供支撑（upright 门控） |

行走/速度追踪相关奖励（`tracking_lin_vel`、`tracking_ang_vel`、`feet_air_time` 等）全部置 0。

### PD 控制器参数

| 参数 | 值 | 说明 |
|------|-----|------|
| stiffness (KP) | 40.0 | 位置增益 |
| damping (KD) | 1.0 | 速度阻尼 |
| action_scale | 0.5 | 动作缩放 |
| decimation | 4 | 控制衰减 |

### 域随机化

恢复训练阶段**全部关闭**（先让机器狗学会站起来）。需要增强鲁棒性时可在 `dog_recovery_config.py` 的 `domain_rand` 中逐步开启摩擦、负载、电机强度、外力扰动等随机化。

## 许可证

BSD-3-Clause，详见 [LICENSE](LICENSE)。

## 致谢

- [legged_gym](https://github.com/leggedrobotics/legged_gym) — 代码基础
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) — NVIDIA 物理仿真环境
- [HIMLoco](https://arxiv.org/abs/2312.11460) — Hybrid Internal Model 算法
