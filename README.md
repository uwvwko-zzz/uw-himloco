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

## 环境要求

- Ubuntu 20.04
- NVIDIA Driver: 525.147.05
- CUDA 12.0
- Python 3.7.16
- PyTorch 1.10.0+cu113
- Isaac Gym: Preview 4

## 安装

1. **创建环境并安装 PyTorch**：

```bash
conda create -n himloco python=3.7.16
conda activate himloco
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

2. **安装 Isaac Gym Preview 4**：

```bash
# 从 https://developer.nvidia.com/isaac-gym 下载
cd isaacgym/python && pip install -e .
```

3. **安装本框架和 rsl_rl**：

```bash
cd HIMLoco-main/rsl_rl && pip install -e .
cd ../himloco_gym && pip install -e .
```

> **注意：** 请使用本项目提供的 `himloco_gym` 和 `rsl_rl`，包含对原始仓库的修改。

4. **MuJoCo 部署（可选）**：

```bash
pip install mujoco onnxruntime pynput pyyaml
```

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
# 46维观测 ONNX 推理（含高度指令）
python mujoco/dog/plane_onnx_46.py dog.yaml

# 45维观测 ONNX 推理
python mujoco/dog/plane_onnx_45.py dog.yaml

# 无策略模式（仅测试 MuJoCo 物理环境）
python mujoco/dog/plane_onnx_46.py dog.yaml --no-policy
```

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

主要奖励项（以 Dog 为例）：

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| tracking_lin_vel | 1.0 | 线速度跟踪奖励 |
| tracking_ang_vel | 0.5 | 角速度跟踪奖励 |
| lin_vel_z | -1.0 | 竖直速度惩罚 |
| base_height | -1.0 | 身体高度偏差惩罚 |
| feet_air_time | 1.0 | 腾空相奖励 |
| dof_acc | -2.5e-7 | 关节加速度惩罚 |
| action_rate | -0.01 | 动作变化率惩罚 |
| smoothness | -0.01 | 动作平滑性惩罚 |
| joint_power | -2e-5 | 关节功率惩罚 |

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

支持多种地形用于课程学习训练：

1. 平坦地面 (plane)
2. 不平地面
3. 台阶 (上/下)
4. 踏脚石
5. 斜坡
6. 长台阶

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

## 关键算法文件

算法实现位于 `rsl_rl/` 目录：

| 文件 | 说明 |
|------|------|
| `rsl_rl/algorithms/him_ppo.py` | HIM PPO 算法实现 |
| `rsl_rl/modules/him_actor_critic.py` | HIM Actor-Critic 网络 |
| `rsl_rl/modules/him_estimator.py` | HIM Estimator 模块 |
| `rsl_rl/runners/him_on_policy_runner.py` | HIM 在策略训练器 |
| `rsl_rl/storage/him_rollout_storage.py` | HIM 数据存储 |

## 许可证

本项目的代码基于 [BSD-3-Clause](LICENSE) 许可证。

## 致谢

- [legged_gym](https://github.com/leggedrobotics/legged_gym) — 本项目的代码基础
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) — NVIDIA 物理仿真环境

## 引用

如果本项目对您有帮助，请引用：

```bibtex
@inproceedings{long2023him,
  title={Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response},
  author={Long, Junfeng and Wang, ZiRui and Li, Quanyi and Cao, Liu and Gao, Jiawei and Pang, Jiangmiao},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}

@misc{long2024hinf,
  title={Learning H-Infinity Locomotion Control}, 
  author={Junfeng Long and Wenye Yu and Quanyi Li and Zirui Wang and Dahua Lin and Jiangmiao Pang},
  year={2024},
  eprint={2404.14405},
  archivePrefix={arXiv},
}