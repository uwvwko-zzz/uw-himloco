# Dog Recovery Gym

基于 [Isaac Gym](https://developer.nvidia.com/isaac-gym) 的四足机器人**倒地恢复（Recovery）**强化学习训练框架。


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
- 姿态：`roll/pitch` 随机，范围由课程学习决定（见下）；`yaw` 用小范围 ±0.5 rad（恢复任务不关心朝向）
- 关节：default 附近 `±0.3 rad` 随机扰动（倒地时腿的各种姿势）
- 速度：清零

### 课程学习

倒地难度（roll/pitch 随机范围）随训练进度递增，实现于 `_reset_root_states`：

```python
t = min(1.0, common_step_counter / fall_angle_curriculum_steps)   # 0→1，纯按全局步数
cur_angle = fall_angle_init + t * (fall_angle_final - fall_angle_init)  # 0.3 → π
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `fall_angle_curriculum_steps` | 5000 | 达到最大难度的步数 |
| `fall_angle_init` | 0.3 rad（17°） | 初始：接近站立的小扰动 |
| `fall_angle_final` | π（180°） | 最终：全方向任意倒地 |

**当前是"按步数线性递增"**（不看策略学没学会，到点就加难度），这是 legged_gym 默认的 terrain curriculum 风格。直接从 π 开始训几乎学不会，课程把搜索空间从小到大渐进展开是能学会的关键。

> 已知可优化点：纯时间驱动的课程存在"学得快也得等、学不会也硬加"的问题。更优的设计是 **performance-based（按成功率自适应）** —— 当前难度成功率超过阈值才加难度，学不会就停住。当前版本未采用，因为时间课程在此任务上节奏匹配良好、训练能稳定收敛。若后续观察到课程拖后腿（提前学会）或推太快（踩点没学会），可改为按成功率自适应。

### 终止条件

恢复任务初始即倒地（base/thigh/calf 必然着地），因此**不靠接触力判定失败**，否则 episode 第一步就会重置。

| 条件 | 结果 | 说明 |
|------|------|------|
| 超时（10s） | 重置 | episode 结束（跑到 max_episode_length 才重置） |

**成功不立即重置**：`upright & height_ok & joint_ok` 三条件同时满足时只标记 `success_buf`、**不触发重置**，episode 继续跑到超时。这是有意为之 —— 见下方奖励函数里 `recovery_success` 的设计。

三个成功条件：
- `grav_z < -0.9`：严格正立（**绝不取 abs**，否则倒立 grav_z=+1 也会被判正立，策略会学翻倒骗奖励）
- `|base_height - cmd[4]| < 0.05 m`：高度达标（跟随指令）
- `mean(|dof_pos - default|) < recovery_joint_tol(0.2 rad)`：关节平均偏差 < 0.2 rad（约 11°）

> 容差从 0.3 收紧到 0.2 是为了把"成功"标准从"踉跄站住"提升到"基本贴 default"。配合奖励函数里 `joint_to_default` 的 sigma 收紧，共同保证停姿精度。

**刷分防护**：`recovery_success` 大奖励每个 episode 只发一次（见下方），由 `already_succeeded_buf` 记录"本 episode 是否已领过"。这阻断了"恢复→故意倒下→再恢复"的刷分循环 —— 倒下重来没有额外收益，站着才是唯一持续拿基础奖励的方式。


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

设计遵循 **"正奖励驱动 + 轻量惩罚约束"** 的结构：正奖励告诉策略"去哪里"（达到目标给分），负惩罚告诉策略"别干什么"（不期望行为扣分，力度整体偏轻，避免压制翻身所需的爆发力）。所有 scale 会被 `_prepare_reward_function` 乘以 `dt(=0.02)` 后生效。

#### 第一层：正奖励（驱动项，scale 为正）

| 奖励项 | scale | 形式 | 说明 |
|--------|-------|------|------|
| `upright_linear` | 3.0 | `(1 - grav_z)/2`，线性 | **主路标**：倒立→正立全程线性给分，恒定梯度。用线性而非 exp 是关键 —— exp 在倒地处梯度小、策略没动力翻身；线性保证翻身全程都有拉力 |
| `stand_height` | 1.5 | `exp(-(h-cmd)²/0.05) × upright_gate` | 把高度拉到指令值。**gate 必须**（`grav_z < -0.7`）：躺平贴地时高度可能恰好接近低目标高度，不 gate 会让策略学会"躺着"（局部最优） |
| `joint_to_default` | 1.5 | `exp(-err/0.25)` | 站姿吸引，把关节拉向 default。err 取 12 关节**均值**（不求和，避免梯度过早消失）。sigma=0.25 收紧（原 1.0 太平，偏 0.2rad 就给 0.96 分，没动力精修） |
| `recovery_success` | 1.0 | `(success & ~already_succeeded) × 10` | **破局重奖**：达标时一次性 ×10。**每 episode 只发一次**（`already_succeeded_buf` 记录），阻断"恢复→倒→恢复"刷分循环 |

#### 第二层：负惩罚（约束项，scale 为负，整体偏轻）

力度梯度设计：核心行为约束（不漂移、不倒立）用中等力度；翻身过程几乎不约束（惩罚太重会压制翻身爆发力，导致头着地）；sim2real 平滑项从轻到极轻。

| 惩罚项 | scale | 形式 | 说明 |
|--------|-------|------|------|
| `upside_down_penalty` | 2.0（函数内取负） | `-clamp(grav_z, min=0)` | **只罚完全倒立**（grav_z>0），不罚侧躺（grav_z≈0 是翻身必经状态，罚侧躺会让策略卡在仰面不敢翻） |
| `joint_to_default_penalty` | -0.5 | `mean(\|dof-default\|) × upright_gate(-0.9)` | default 近处精修，补 `joint_to_default` 奖励在 default 附近梯度仍不够陡的缺陷。gate=-0.9 极晚触发（只在几乎完全正立时），避免翻身末段被拽导致前扑 |
| `action_rate` | -0.2 | `Σ(actions - last_actions)²` | 动作变化率惩罚（sim2real：抑制电机高频跳变）。从基线 -0.1 小幅加强到 -0.2 |
| `lin_vel_xy` | -0.5 | `Σ(base_lin_vel_xy)²` | 水平速度惩罚：恢复任务要求原地站起，不应漂移 |
| `ang_vel_xy` | -0.05 | `Σ(base_ang_vel_xy)²` | roll/pitch 角速度惩罚：约束姿态稳定 |
| `torques` | -1e-5 | `Σ(torques)²` | 力矩惩罚（极轻，几乎占位，防策略走极端出力） |
| `dof_vel` | -1e-4 | `Σ(dof_vel)²` | 关节速度惩罚（极轻） |
| `dof_acc` | -2.5e-7 | `Σ(last_dof_vel - dof_vel)²/dt` | 关节加速度惩罚（极轻） |

行走/速度追踪相关奖励（`tracking_lin_vel`、`tracking_ang_vel`、`feet_air_time` 等）全部置 0。

#### 设计要点

1. **线性 `upright_linear` + gate `stand_height` + 一次性 `recovery_success`** 三者协同对抗两个局部最优："躺平"（被 gate 的 height 奖励破除）和"悬停半立"（被 success 大奖励破除）。
2. **惩罚整体偏轻是有意为之**：恢复任务需要爆发力翻身，重惩罚会压塌策略（曾实验把 smoothing 加大 400 倍，直接导致头着地）。代价是动作偏剧烈 —— 这是已知 trade-off，后续小步优化。
3. **`only_positive_rewards = False`**：必须为 False，否则总 reward 被截断为 ≥0，所有负惩罚失效。

> 三组关键调参：① `recovery_success` 改每 episode 一次（修刷分）；② sigma `1.0→0.25` + tol `0.3→0.2`（修精度）；③ `action_rate` `-0.1→-0.2`（修剧烈，最小步）。一次只改一组、小步验证。


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


