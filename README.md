# HIMLoco Gym

基于 [Isaac Gym](https://developer.nvidia.com/isaac-gym) 的四足机器人强化学习 locomotion 训练框架，实现自 [HIMLoco (Hybrid Internal Model)](https://arxiv.org/abs/2312.11460) 和 [H-Infinity Locomotion Control](https://arxiv.org/abs/2404.14405) 论文。支持从仿真训练到 MuJoCo sim2sim 验证、ONNX 模型导出的完整流程。

## 项目结构

```
himloco_gym/
├── legged_gym/                    # 核心训练代码
│   ├── envs/                      # 环境定义
│   │   ├── base/                  # 基类
│   │   │   ├── base_config.py         # 配置基类
│   │   │   ├── base_task.py           # 任务基类
│   │   │   ├── legged_robot.py        # 腿式机器人环境核心逻辑
│   │   │   └── legged_robot_config.py # 腿式机器人配置（奖励、域随机化、噪声等）
│   │   ├── dog/                   # Dog 机器人配置
│   │   │   └── dog_config.py          # Dog URDF 路径、PD参数、奖励权重
│   │   ├── a1/                    # Unitree A1 配置
│   │   ├── go1/                   # Unitree Go1 配置
│   │   ├── go2/                   # Unitree Go2 配置
│   │   └── aliengo/              # Aliengo 配置
│   ├── scripts/                   # 训练/测试脚本
│   │   ├── train.py                   # 通用训练入口
│   │   ├── play.py                    # 通用测试入口
│   │   ├── 45/                        # 45维观测版本脚本
│   │   │   ├── plane_pt.py               # 平地训练测试
│   │   │   └── mang_terrain_pt.py        # 复杂地形训练测试
│   │   ├── 46/                        # 46维观测版本脚本（含高度指令）
│   │   │   ├── plane_pt.py               # 平地训练测试（键盘控制）
│   │   │   └── mang_terrain_pt.py        # 复杂地形训练测试
│   │   └── onnx/                      # ONNX 导出脚本
│   │       ├── 45/                        # 45维 ONNX 导出
│   │       └── 46/                        # 46维 ONNX 导出
│   ├── utils/                     # 工具模块
│   │   ├── task_registry.py           # 任务注册表
│   │   ├── helpers.py                 # 辅助函数
│   │   ├── terrain.py                 # 地形生成
│   │   ├── math.py                    # 数学工具
│   │   └── logger.py                  # 日志工具
│   └── tests/                     # 测试
│       └── test_env.py
├── mujoco/                        # MuJoCo sim2sim 部署验证
│   ├── dog/                       # Dog 机器人 MuJoCo 验证
│   │   ├── plane_onnx_45.py           # 45维观测 ONNX 推理 (MuJoCo)
│   │   ├── plane_onnx_46.py           # 46维观测 ONNX 推理 (MuJoCo, 含高度指令)
│   │   ├── play_onnx_46.py            # 46维 ONNX 复杂地形键盘控制
│   │   ├── look_xml.py                # MuJoCo XML 可视化
│   │   └── config/
│   │       └── dog.yaml               # MuJoCo 仿真参数配置
│   ├── go1/                       # Go1 MuJoCo 验证
│   ├── go2/                       # Go2 配置
│   └── scripts/
│       └── urdf_to_xml.py             # URDF 转 MuJoCo XML 工具
├── resources/                     # 机器人模型资源
│   └── robots/
│       ├── dog/                   # Dog 机器人
│       │   ├── urdf/                  # URDF 模型文件
│       │   ├── meshes/               # STL 网格文件
│       │   └── xml/                  # MuJoCo XML 模型
│       ├── a1/                    # Unitree A1
│       ├── go1/                   # Unitree Go1
│       ├── go2/                   # Unitree Go2
│       └── aliengo/              # Aliengo
├── licenses/                      # 许可证
├── setup.py                       # 安装配置
└── README.md                      # 本文件
```

## 支持的机器人

| 机器人 | 任务名 | 说明 |
|--------|--------|------|
| **Dog** | `dog` | 自定义四足机器人（主要开发目标） |
| Unitree A1 | `a1` | Unitree A1 四足机器人 |
| Unitree Go1 | `go1` | Unitree Go1 四足机器人 |
| Unitree Go2 | `go2` | Unitree Go2 四足机器人 |
| Aliengo | `aliengo` | Aliengo 四足机器人 |

## 使用方法

### 训练

```bash
# 通用训练（headless 模式）
python legged_gym/scripts/train.py --task=dog --headless

# 指定不同的机器人
python legged_gym/scripts/train.py --task=go1 --headless
python legged_gym/scripts/train.py --task=a1 --headless
```

### 测试 / 可视化

```bash
# 通用测试（加载最新模型）
python legged_gym/scripts/play.py --task=dog

# 46维观测版本 - 平地键盘控制
python legged_gym/scripts/46/plane_pt.py --task=dog

# 46维观测版本 - 复杂地形
python legged_gym/scripts/46/mang_terrain_pt.py --task=dog

# 45维观测版本
python legged_gym/scripts/45/plane_pt.py --task=dog
```

#### 键盘控制（plane_pt.py）

| 按键 | 功能 |
|------|------|
| W / S | 前进 / 后退 |
| A / D | 左移 / 右移 |
| Q / E | 左转 / 右转 |
| F | 站起（升高身体） |
| R | 蹲下（降低身体） |
| Z | 重置高度到默认 |
| P | 重置机器人位置 |
| ESC | 退出 |

### ONNX 模型导出

```bash
# 46维版本导出
python legged_gym/scripts/onnx/46/pt_to_onnx.py

# 45维版本导出
python legged_gym/scripts/onnx/45/pt_to_onnx.py
```

### MuJoCo Sim2Sim 验证

```bash
# 46维 ONNX 复杂地形键盘控制（当前 himloco_high 模型）
python mujoco/dog/play_onnx_46.py

# 46维观测 ONNX 推理（含高度指令）
python mujoco/dog/plane_onnx_46.py dog.yaml

# 45维观测 ONNX 推理
python mujoco/dog/plane_onnx_45.py dog.yaml

# 无策略模式（仅测试 MuJoCo 物理环境）
python mujoco/dog/plane_onnx_46.py dog.yaml --no-policy
```

`play_onnx_46.py` 默认加载 `logs/dog_high_rough/model_2600.onnx`。这是使用
`dog_high` 任务训练的**高墙/宽顶平台专用模型**，重点能力是主动抬腿、借助墙面接触，
登上并越过垂直高障碍；训练地形通过课程学习将平台高度从 5 cm 逐步提高到 30 cm，
平台顶面沿前进方向长 1 m。

MuJoCo 验证场景使用 `resources/robots/dog/xml/dog_terrain.xml`。脚本中的 `policy_path`、`xml_path` 和
`config_path` 为本地绝对路径，换机器或移动项目后需先修改为实际路径。

#### MuJoCo 键盘控制（play_onnx_46.py）

| 按键 | 功能 |
|------|------|
| W / S 或 ↑ / ↓ | 前进 / 后退（按住运动，松开停止） |
| A / D 或 ← / → | 左移 / 右移 |
| Q / E | 左转 / 右转 |
| F / R | 升高 / 降低身体（步长 0.02 m） |
| Z | 恢复默认高度 0.25 m |
| X | 紧急停止移动指令 |
| T | 重置机器人和观测历史 |
| Y | 开关模型输出打印 |

Linux 下键盘监听使用 `evdev`。如果出现 `/dev/input/event*` 权限错误，
可将当前用户加入 `input` 组后重新登录：

```bash
sudo usermod -aG input $USER
```

MuJoCo free-joint 的 `qvel[3:6]` 已是机体坐标系角速度，构造观测时不应再做逆旋转；
否则机器人转向后的角速度观测会失真。

## 核心配置说明

### 观测空间

项目支持两种观测维度版本：

| 版本 | 单步观测维度 | 历史步数 | 总观测维度 | 说明 |
|------|-------------|---------|-----------|------|
| **45维** | 45 | 6 | 270 | 基础观测 |
| **46维** | 46 | 6 | 276 | 45维 + 高度指令 |

46维单步观测结构：

| 索引 | 维度 | 内容 |
|------|------|------|
| [0:3] | 3 | 速度命令 (vx, vy, wz) × 缩放 |
| [3:6] | 3 | 身体角速度 × 缩放 |
| [6:9] | 3 | 重力投影向量 |
| [9:21] | 12 | 关节角度偏差 (dof_pos - default) |
| [21:33] | 12 | 关节速度 × 缩放 |
| [33:45] | 12 | 上一步动作 |
| [45] | 1 | 高度指令归一化值（仅46维版本） |

### 动作空间

- **维度**：12（四足机器人 × 3关节）
- **控制方式**：PD位置控制，`target_angle = action × action_scale + default_angle`
- **动作缩放**：0.25
- **控制频率**：50Hz（decimation=4, dt=0.005s）

### 奖励函数

`dog_high` 的奖励配置位于
`legged_gym/envs/dog/dog_high_config.py`。该模型不是普通平地行走模型，奖励设计围绕
“接近平台 → 稳定登顶 → 完整越过”展开：

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| wall_crossing | 100.0 | 完整越过平台后的一次性核心成功奖励；每道墙每回合只奖励一次 |
| platform_mount | 50.0 | 身体达到平台高度、姿态稳定且至少两脚支撑时的一次性登顶奖励 |
| platform_progress | 0.5 | 平台前 1 m 范围内沿命令方向前进的连续引导奖励 |
| tracking_lin_vel | 2.0 | 线速度跟踪奖励，维持主动向平台运动 |
| tracking_ang_vel | 0.5 | 角速度跟踪奖励 |
| alive | 0.2 | 存活奖励 |
| feet_air_time | 0.3 | 鼓励跨越时抬腿和腾空；平台附近目标腾空时间为 0.35 s |
| lin_vel_z | -0.5 | 竖直速度惩罚，降低权重以免压制起跳和上墙 |
| orientation | -0.2 | 抑制过度翻滚和俯仰 |
| termination | -10.0 | 摔倒终止惩罚，最低允许基座高度为 0.12 m |
| action_rate | -0.02 | 抑制动作突变 |
| smoothness | -0.005 | 二阶动作平滑惩罚 |
| dof_acc | -2.5e-7 | 关节加速度惩罚 |
| joint_power | -1e-6 | 关节功率惩罚 |
| feet_contact_forces | -1e-5 | 抑制过大的落地冲击 |

为避免策略“害怕接触墙面”，训练中不惩罚大腿、小腿和基座与障碍物接触，也不因这些
接触直接终止回合。`base_height`、`collision`、`feet_stumble` 和站立姿态相关权重均设为
0，因为翻越 30 cm 垂直平台时需要身体升高、腿部蹬墙以及短时较大俯仰。与此同时，
`wall_crossing` 只在真正到达平台远端时触发，防止机器人靠撞墙或在墙边来回移动刷分。

### 域随机化

训练时启用的随机化参数：

| 参数 | 范围 | 说明 |
|------|------|------|
| payload_mass | [-1, 2] kg | 负载质量 |
| com_displacement | [-0.05, 0.05] m | 质心位移 |
| friction | [0.2, 1.25] | 摩擦系数 |
| motor_strength | [0.9, 1.1] | 电机强度 |
| kp | [0.9, 1.1] | 比例增益 |
| kd | [0.9, 1.1] | 微分增益 |
| disturbance | [-30, 30] N·m | 外力干扰 |
| push_robots | max 1.0 m/s | 随机推力 |

### 地形类型

当前 `dog_high` 专门生成横跨地形宽度的宽顶高墙平台，而不是混合普通崎岖地形：

- 平台相对地形块中心位于 X = 2.0 m
- 平台顶面长度为 1.0 m
- 课程高度从 0.05 m 线性增加到 0.30 m
- 60% 的速度命令专门用于前后穿越平台，40% 保留完整平移和转向训练

### PD 控制器参数

| 参数 | Dog | 说明 |
|------|-----|------|
| stiffness (KP) | 40.0 | 位置增益 |
| damping (KD) | 1.0 | 速度阻尼 |
| action_scale | 0.25 | 动作缩放 |
| decimation | 4 | 控制衰减 |

## MuJoCo 部署注意事项

### 关节顺序映射

Isaac Gym 和 MuJoCo 的后两条腿关节顺序不同：

```
Isaac Gym: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
MuJoCo:    FL(0-2), FR(3-5), RR(6-8), RL(9-11)
```

映射关系：
```python
MUJOCO_TO_ISAAC = [0,1,2, 3,4,5, 9,10,11, 6,7,8]
ISAAC_TO_MUJOCO = [0,1,2, 3,4,5, 9,10,11, 6,7,8]
```

### URDF 转 MuJoCo XML

```bash
python mujoco/scripts/urdf_to_xml.py
```

## 训练日志

训练日志和模型保存在：
```
himloco_gym/logs/<experiment_name>/<run_name>/
```
