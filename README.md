# UW HIMLoco · 轮足机器人

基于 Isaac Gym 和 HIMLoco 的轮足机器人强化学习训练框架，支持复杂地形课程学习、高台攀爬、ONNX 导出与 MuJoCo sim2sim 验证，并提供浏览器控制面板和地形编辑工具。

仓库：<https://github.com/uwvwko-zzz/uw-himloco>。本说明对应 **`wheel` 分支**。

## 获取代码

```bash
git clone --branch wheel https://github.com/uwvwko-zzz/uw-himloco.git
cd uw-himloco
```

以下命令均在仓库根目录运行。本地目录名可以是 `himloco_wu`。

## 当前任务

以 [任务注册文件](legged_gym/envs/__init__.py) 为准：

| 任务名 | 功能 | 配置 | 日志目录 |
| --- | --- | --- | --- |
| `w1w` | W1W 轮足机器人复杂地形行走 | [W1W 配置](legged_gym/envs/w1w/w1w_config.py) | `logs/w1w_rough/` |
| `yqz_000` | YQZ-000 复杂地形行走与负载随机化 | [YQZ 配置](legged_gym/envs/yqz_000/yqz_000_config.py) | `logs/yqz_rough/` |
| `yqz_high` | YQZ 高台攀爬课程学习 | [高台配置](legged_gym/envs/yqz_high/yqz_high_config.py) | `logs/yqz_high/` |

高台训练课程覆盖 0.08–0.80 m，实际通过能力取决于训练得到的策略。

## 安装环境

### Isaac Gym 训练环境

训练需要支持 Isaac Gym 的 Linux / NVIDIA GPU 环境。先准备 Isaac Gym 及其配套的 Python、PyTorch 和 CUDA 依赖，再安装算法库与本项目。

**本仓库未包含 `rsl_rl/`。** 需要另行准备 HIMLoco 版本的算法库，包含 `HIMActorCritic` 和 `HIMOnPolicyRunner` 接口。

```bash
# 替换为实际目录
pip install -e /path/to/isaacgym/python
pip install -e /path/to/HIMLoco/rsl_rl
pip install -e .

# ONNX 导出依赖
pip install onnx onnxruntime
```

如果保留原 HIMLoco 多目录布局，算法库在项目同级目录，可使用 `pip install -e ../rsl_rl`。

### MuJoCo 环境

ONNX sim2sim 可以在独立 Python 环境运行，无需启动 Isaac Gym。安装内置工具包会同时安装 MuJoCo、ONNX Runtime、NumPy、Pillow 和 PyYAML 等依赖：

```bash
pip install -e ./mujoco/mujoco-gui
```

详细说明见 [MuJoCo GUI README](mujoco/mujoco-gui/README.md)。

## 训练

```bash
python legged_gym/scripts/train.py --task=w1w --headless
python legged_gym/scripts/train.py --task=yqz_000 --headless
python legged_gym/scripts/train.py --task=yqz_high --headless
```

模型与日志保存在 `logs/<experiment_name>/<run>/`。以下命令中的 `YOUR_RUN` 和 `model_XXXX` 是占位符，请替换为实际训练目录和模型名称。

## Isaac Gym 策略测试

显式指定 checkpoint 与机身高度：

```bash
python legged_gym/scripts/play.py --task=w1w \
  --model logs/w1w_rough/YOUR_RUN/model_XXXX.pt --height 0.45

python legged_gym/scripts/play.py --task=yqz_000 \
  --model logs/yqz_rough/YOUR_RUN/model_XXXX.pt --height 0.47

# 高台任务专用交互入口
python legged_gym/scripts/yqz_high/play.py --task=yqz_high \
  --model logs/yqz_high/YOUR_RUN/model_XXXX.pt
```

## ONNX 导出

导出脚本从 checkpoint 推断网络维度。虽然位于 `onnx/46/`，也支持轮足策略的 58 维单步观测、348 维历史观测与 16 维动作。

```bash
python legged_gym/scripts/onnx/46/pt_to_onnx.py \
  --model logs/yqz_rough/YOUR_RUN/model_XXXX.pt \
  --output logs/yqz_rough/YOUR_RUN/model_XXXX.onnx
```

W1W 和高台任务使用同一入口，替换为对应任务模型路径即可。不指定 `--output` 时，输出到 checkpoint 同目录的同名 `.onnx` 文件。

## MuJoCo sim2sim

使用与机器人和训练任务匹配的 ONNX 模型：

```bash
# W1W 原生查看器
python mujoco/w1w/play_onnx.py \
  --onnx logs/w1w_rough/YOUR_RUN/model_XXXX.onnx

# W1W 浏览器控制面板
python mujoco/w1w/play_gui.py \
  --onnx logs/w1w_rough/YOUR_RUN/model_XXXX.onnx --gui

# YQZ-000 浏览器控制与地形切换
python mujoco/yqz_000/play_onnx.py \
  --onnx logs/yqz_rough/YOUR_RUN/model_XXXX.onnx \
  --map flat --vx 1.0 --height 0.47 --gui

# YQZ 高台：从 0.4 m 高台开始
python mujoco/yqz_high/play_onnx.py \
  --onnx logs/yqz_high/YOUR_RUN/model_XXXX.onnx \
  --platform-height 0.4 --gui
```

浏览器面板默认地址为 `http://127.0.0.1:8765`。YQZ 播放器省略 `--gui` 即使用原生查看器；添加 `--headless --duration 2` 可运行两秒无界面仿真。

常用控制：`W/S` 前后移动，`A/D` 侧移，`Q/E` 转向。高度调节、复位与外力扰动等按键以各播放器说明为准：

- [W1W 仿真说明](mujoco/w1w/README.md)
- [YQZ-000 仿真说明](mujoco/yqz_000/README.md)
- [YQZ 高台仿真说明](mujoco/yqz_high/README.md)

### 地形与模型工具

```bash
# W1W 地形编辑器
bash mujoco/w1w/start_editor.sh

# 查看 W1W 模型、关节和碰撞体
python -m runtime_control.preview mujoco/w1w/w1w.xml

# URDF 转换工具
bash mujoco/mujoco-gui/start_urdf_converter.sh
```

YQZ 网格预处理及 URDF 转 MJCF 命令见 [YQZ-000 仿真说明](mujoco/yqz_000/README.md)。

## 策略接口

三个任务使用混合控制：12 个腿关节目标位置与 4 个轮关节目标速度，最终转换为关节力矩。

| 项目 | 当前配置 |
| --- | --- |
| 单步观测 | 58 维 |
| 历史长度 | 6 步 |
| 策略输入 | 348 维 |
| 策略输出 | 16 维 |
| 物理步长 | 0.005 s |
| 控制间隔 | 4 个物理步，策略频率 50 Hz |

单步观测包含速度指令（3）、机身角速度（3）、重力投影（3）、关节位置特征（16）、关节速度（16）、上一时刻动作（16）和高度指令（1）。

连续轮关节的位置处理、关节顺序、轮子方向、动作缩放和控制增益必须与训练配置一致。各任务的参数不同，部署时应使用对应的 adapter。

## 奖励函数解析与设计

奖励配置沿 `W1W → YQZ-000 → YQZ 高台` 继承。子类未覆盖的权重仍然生效，因此高台任务同时保留速度跟踪、高度、功率和动作平滑等基础项。以下数值为当前代码中的配置权重，`0` 表示关闭该项。

实现入口：[基础奖励](legged_gym/envs/base/legged_robot.py)、[轮足奖励](legged_gym/envs/w1w/wheeled_legged_robot.py)、[高台奖励](legged_gym/envs/yqz_high/yqz_high.py)。

### 总奖励与时间缩放

每个奖励函数先返回未加权特征 `f_i`，再乘以配置权重 `w_i` 与控制步长 `Δt`。当前控制步长为 `0.02 s`：

```text
r_regular = Δt × Σ(w_i × f_i)
r_total   = max(r_regular, 0) + Δt × w_termination × I(非超时终止)
```

三个任务都启用 `only_positive_rewards=True`。裁剪作用于普通奖励之和，负项仍会降低正项形成的收益；终止惩罚在裁剪后添加，因此总奖励仍可为负。代码记录的各项 `episode_sums` 是裁剪前的加权累积值，不能直接相加当作最终回报。

权重不是单步实际奖励。例如高台成功项返回一次 `1`，配置权重为 `100`，实际单次奖励为 `100 × 0.02 = 2`；前轮阶段奖励 `20` 对应单次 `0.4`。进度项返回 `进度增量 / Δt`，抵消时间缩放，完整进度对应的累计收益才等于该项权重。

### 基础行走：跟踪、稳定与能耗

| 奖励项 | W1W 权重 | YQZ-000 权重 | 特征与设计目的 |
| --- | --- | --- | --- |
| `tracking_lin_vel` | 3 | 3 | `exp(-‖v_xy-v_cmd‖²/0.25)`，鼓励平面速度跟踪 |
| `tracking_ang_vel` | 2 | 2 | `exp(-(ω_z-ω_cmd)²/0.25)`，鼓励转向跟踪 |
| `lin_vel_z` | -0.5 | -0.5 | 机身坐标系竖直速度平方，抑制弹跳 |
| `ang_vel_xy` | -0.1 | -0.1 | 横滚、俯仰角速度平方和，抑制摇晃 |
| `orientation` | -0.5 | -0.5 | 重力投影水平分量平方和，鼓励机身保持水平 |
| `base_height` | -2 | -2 | 相对地形的机身高度与高度指令之间的平方误差 |
| `height_tracking` | 1.5 | 1.5 | `exp(-高度误差²/0.0025)`，强化目标高度附近的精细跟踪 |
| `torques` | -5e-6 | -7.5e-6 | 所有执行器力矩平方和，限制持续大力矩 |
| `joint_power` | -2e-6 | -2e-6 | `Σ|关节速度 × 力矩|`，减少机械功率消耗 |
| `leg_dof_acc` | -1e-7 | -5e-8 | 12 个腿关节加速度平方和，抑制腿部抖动 |
| `wheel_dof_acc` | -2.5e-9 | -2.5e-9 | 4 个轮关节加速度平方和，约束轮速突变 |
| `action_rate` | -0.005 | -0.003 | `‖a_t-a_(t-1)‖²`，限制相邻动作变化 |
| `smoothness` | -0.005 | -0.003 | `‖a_t-2a_(t-1)+a_(t-2)‖²`，限制动作二阶差分 |
| `collision` | -0.5 | -0.5 | 配置中的非轮部位接触力超过 0.1 N 的数量，减少机身、腿部碰撞 |
| `termination` | -0.5 | -0.5 | 非超时终止事件惩罚，普通超时不触发 |

高度采用“平方误差惩罚 + 指数跟踪奖励”：偏差较大时仍有误差代价，接近目标时指数项提供更明显的收益。默认站立高度 W1W 为 0.45 m、YQZ 为 0.47 m；实际高度目标读取高度指令。

腿和轮的速度尺度不同，因此关闭通用 `dof_acc`，分别设置加速度惩罚。YQZ 减弱腿加速度和动作平滑代价，给负载下的抬腿越障留出动作空间，同时增加力矩代价约束持续用力。各项原始特征的单位与大小不同，不能仅凭权重绝对值判断贡献大小。

### 轮足专用：平地滚动、越障抬腿与零指令站立

| 奖励项 | W1W 权重 | YQZ-000 权重 | 生效方式与设计目的 |
| --- | --- | --- | --- |
| `stand_still` | -2 | -1 | 低速度指令下惩罚腿关节默认姿态的 L1 偏差，带直立程度缩放 |
| `joint_pos_penalty` | -0.5 | -0.03 | 腿关节默认姿态偏差的 L2 范数；判为静止时乘以 5，并按直立程度缩放 |
| `moving_hip_default` | -1 | -0.1 | 移动且基本直立时，惩罚髋关节偏离各自默认角度，限制轮端异常姿态 |
| `zero_command_leg_default` | 0 | -1 | 接近零指令且直立时，惩罚腿关节默认姿态的 L1 偏差 |
| `wheel_speed` | -2e-4 | -2e-4 | `Σmax(|轮速|-45,0)²`，仅惩罚超速部分 |
| `wheel_vel_penalty` | -0.01 | -0.01 | 移动状态惩罚离地轮空转，静止状态惩罚全部轮速 |
| `wheel_contact_loss` | -0.2 | -0.01 | 有移动指令时惩罚离地轮数量，使用一步接触滞回过滤接触抖动 |
| `zero_command_wheel_action` | 0 | -0.2 | 接近零指令且直立时，惩罚轮动作绝对值和 |
| `zero_command_wheel_velocity` | 0 | -0.02 | 接近零指令且直立时，惩罚实际轮速绝对值和，减少漂移 |
| `foot_clearance` | 0 | -0.2 | 轮心在机身坐标系中的高度误差平方，乘以相对机身的水平运动速度 |

`foot_clearance` 的 YQZ 目标为机身坐标系 `z=-0.325 m`，不是离地高度 0.325 m。它在轮端相对机身移动时增加错误高度的代价，引导换脚时适当抬轮；没有直接规定腾空时间。

轮足设计以“容易地形保持滚动，必要时主动抬腿”为目标，因此关闭 `feet_air_time`、`diagonal_sync`、`hip_mirror_symmetry`、`stand_four_feet`、`stand_orientation` 和 `default_pos_linear` 等项。没有强制传统四足对角小跑接触时序，也没有持续奖励腾空。

YQZ 将接触丢失代价从 `-0.2` 降为 `-0.01`，并减弱默认姿态与髋关节约束，使越障和负载下的换脚更容易获得净收益。零指令姿态与轮速项则单独加强停稳约束。`stand_still` 和 `zero_command_leg_default` 的门限及直立条件不同，但会在部分状态重叠，调参时要看它们的合计贡献。

### 高台任务：分阶段引导完整登台与下台

高台任务继承 YQZ 奖励，并增加以下项。登台专用项主要在非平地、正向直行指令下生效；“前轮/后轮”按世界坐标 X 的领先/落后位置动态选取。

| 奖励项 | 权重 | 触发与设计 |
| --- | --- | --- |
| `platform_progress` | 2 | 机身从起点接近台阶的归一化进度增量，仅突破本回合历史最大值时给奖励 |
| `front_wheel_progress` | 3 | 障碍接触反射触发后，两个领先轮中较慢者跨越台阶边缘的进度增量 |
| `front_platform_stage` | 20 | 两个领先轮稳定在台面接触 5 个控制步后，一次性奖励 |
| `front_contact_regression` | -2 | 前轮阶段已完成、整机尚未成功时，前轮掉出台面接触的持续代价 |
| `rear_wheel_progress` | 4 | 两个落后轮平均 X 位置跨越边缘的进度增量，鼓励后轮跟进 |
| `wheels_on_platform` | 2 | 至少 3 轮在台面且两后轮均在台面时，奖励台面接触轮比例 |
| `base_on_platform` | 1 | 机身越过前缘、达到登台高度，并满足台面轮接触条件时奖励 |
| `platform_success` | 100 | 机身超过前缘 0.35 m、满足高度与轮接触条件并连续保持 5 步后，一次性奖励 |
| `descent_progress` | 3 | 登台成功后，穿过台顶并越过远端边缘的历史最大进度增量 |
| `descent_ground_contact` | 1 | 已开始下台且机身降低至地面附近后，按接触轮比例与直立程度奖励接触恢复 |
| `descent_success` | 100 | 越过远端、落到低地面附近、四轮接触、机身直立且竖直速度稳定，连续保持 10 步后一次性奖励 |
| `backward_motion` | -3 | 惩罚速度与 X 指令方向相反的运动，正常执行倒退指令不会因此受罚 |
| `highplatform_yaw` | -2 | 登台指令下的航向角误差平方，鼓励正对台阶 |
| `rolling_leg_default` | -1 | 普通直线滚动且直立时约束腿默认姿态；障碍反射或下台过渡时释放 |
| `rolling_orientation` | -2 | 普通直线滚动时约束机身倾斜；障碍反射或下台过渡时释放 |
| `zero_command_actor_action` | -1 | 零速度指令时惩罚原始 16 维策略动作平方和 |
| `highplatform_world_z_vel` | -10 | `max(|世界竖直速度|-0.75,0)²`，重点抑制猛烈冲击 |
| `highplatform_leg_dof_vel` | -1 | 腿关节速度超过 16 rad/s 的部分平方和，轮关节不计入 |

高台还把继承的 `collision` 改为 `-1`、`orientation` 改为 `-1`，并将 `wheel_contact_loss` 改为 `-2`。其中接触丢失项的实现也被覆盖：只在普通直线滚动模式下惩罚离地，越障和下台过渡可抬轮。

设计逻辑如下：

1. **解决稀疏成功信号。** 接近、前轮放置、后轮跟进与整机登台提供逐级反馈，让尚未完成全程的策略也能区分更好的动作。
2. **减少来回刷进度。** 进度奖励只支付超过本回合历史最大值的部分，后退再前进不会重复领取同一段收益；阶段和成功奖励由事件锁存，每回合只支付一次。
3. **减少只有前轮上台的局部解。** 台面轮接触、机身登台和成功判定均包含后轮要求；前轮掉落另有回退代价。
4. **允许必要的俯仰与抬腿。** 障碍反射由领先轮持续受到与前进指令相反的接触力触发，门限为 20 N、持续 2 步。普通滚动约束在该模式下释放，但基础倾斜、高度和能耗项仍然生效。
5. **避免转身绕过任务。** 登台指令下线速度跟踪改用世界坐标 +X，并配合航向与反向运动代价；其他指令仍按机身坐标系跟踪。
6. **训练完整上台、过台与下台。** 登台成功后继续训练远端下落与稳定落地，成功条件要求接触和姿态持续稳定，减少瞬间接触误判。
7. **让策略学会零输出。** 高台控制器在零指令下还会将执行动作置零，`zero_command_actor_action` 直接约束门控前的策略输出。这一门控属于控制逻辑，停稳效果不能全部归因于奖励。

台阶位置、高度与接触几何用于训练阶段奖励判定，actor 的 58 维单步观测仍为指令与本体信息，没有直接输入台阶高度或边缘位置。进度历史最大值属于防重复支付机制，并非严格保证策略不变的势函数塑形。

### 调参原则与验证

先保证速度跟踪和任务阶段能产生有效收益，再逐步增加稳定、能耗和动作平滑约束。若策略一直滚动顶住台阶，应检查抬腿动作是否被姿态、接触丢失和加速度代价压制；若高频抖腿，应检查腿加速度、动作差分、轮空转与关节速度超限项。

调参应结合每项加权日志、最终回报、登台/下台成功率和实际动作观察。正奖励裁剪会让多个较差动作都得到零回报；持续台面奖励也可能鼓励停留，因此应同时观察全程完成率与耗时。评价至少覆盖零指令停稳、速度跟踪、不同台阶高度、负载变化与 MuJoCo 验证，不能仅以训练回报升高判断设计有效。

## 项目结构

```text
.
├── legged_gym/
│   ├── envs/
│   │   ├── base/          # 环境与配置基类
│   │   ├── w1w/           # 轮足环境与 W1W 配置
│   │   ├── yqz_000/       # YQZ-000 配置
│   │   └── yqz_high/      # 高台环境与课程配置
│   ├── scripts/          # 训练、测试与 ONNX 导出
│   └── utils/            # 任务注册、地形与辅助工具
├── mujoco/
│   ├── w1w/              # W1W 播放器与 adapter
│   ├── yqz_000/          # YQZ 播放器、网格处理与 adapter
│   ├── yqz_high/         # 高台 sim2sim 播放器与 adapter
│   ├── mujoco-gui/       # 浏览器面板、地形编辑与模型工具
│   └── scripts/          # 模型转换工具
├── resources/robots/    # URDF、网格与 MJCF 资源
├── generated/           # 生成的地形资源
├── setup.py
├── LICENSE
└── README.md
```

`logs/` 为运行时生成目录，已被 `.gitignore` 忽略。新克隆的仓库需要自行训练或准备匹配的模型。

## 来源与许可证

本项目基于 HIMLoco 与 legged_gym 相关工作修改，保留原有版权声明。许可证见 [LICENSE](LICENSE)，第三方组件以各自声明为准。
