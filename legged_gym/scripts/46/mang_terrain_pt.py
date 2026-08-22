# 多障碍环境 - 46维度版本

import os
import sys

# === 始终导入当前脚本所属仓库，避免误用相邻的 himloco_gym ===
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))

import isaacgym

if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

import torch
import numpy as np
from pynput import keyboard
from types import MethodType

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.terrain import Terrain

def pop_custom_args():
    """取出 Isaac Gym 参数解析器不认识的 --model 参数。"""
    argv = sys.argv[1:]
    remaining = [sys.argv[0]]
    model_values = []
    index = 0

    while index < len(argv):
        arg = argv[index]
        if arg == "--model":
            if index + 1 >= len(argv):
                raise SystemExit("--model 后面必须提供模型路径")
            index += 1
            model_values.append(argv[index])
        elif arg.startswith("--model="):
            model_values.append(arg.split("=", 1)[1])
        else:
            remaining.append(arg)
        index += 1

    sys.argv[:] = remaining
    if len(model_values) != 1 or not model_values[0]:
        raise SystemExit(
            f"Usage: {os.path.basename(__file__)} "
            "--task=<task> --model=<path>"
        )

    path = os.path.abspath(os.path.expanduser(model_values[0]))
    if not os.path.isfile(path):
        raise SystemExit(f"MODEL file not found: {path}")
    return path


def get_checkpoint_critic_obs_dim(model_path):
    """读取 checkpoint 的 Critic 输入维度，用于匹配特权观测配置。"""
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit(
            f"Checkpoint does not contain model_state_dict: {model_path}"
        )
    weight = state_dict.get("critic.0.weight")
    if weight is None or weight.ndim != 2:
        raise SystemExit(
            f"Checkpoint does not contain a valid critic.0.weight: {model_path}"
        )
    return int(weight.shape[1])


def t_shaped_stairs_terrain(terrain, horizontal_scale, vertical_scale):
    """
    三侧可攀爬的 T 字形台阶地形。
    尺寸说明 (单位: mm -> m):
    - 顶部平台: 1.0m x 1.0m, 高 0.4m
    - 台阶: 深 0.3m，每级高 0.2m
    - 从地面经 0.2m 台阶到达 0.4m 平台
    """
    
    # 转换物理尺寸为网格索引
    def to_idx(m):
        return int(m / horizontal_scale)
    
    def to_height(m):
        return int(m / vertical_scale)

    # 计算中心点（将台阶放在地形区域的正中心）
    center_x = terrain.length // 2
    center_y = terrain.width // 2

    # 基本参数定义
    platform_size = to_idx(1.0)
    step_width = to_idx(0.3)
    step_height_m = 0.2
    
    # 1. 绘制顶部核心平台 (1000x1000, 高 400mm)
    x_start = center_x - platform_size // 2
    x_end = x_start + platform_size
    y_start = center_y - platform_size // 2
    y_end = y_start + platform_size
    
    terrain.height_field_raw[x_start:x_end, y_start:y_end] = to_height(0.4)

    # 2. 三侧各放一级 20cm 台阶，第二级就是 40cm 顶部平台。
    current_h = to_height(step_height_m)
    terrain.height_field_raw[
        x_start:x_end, y_end:y_end + step_width
    ] = current_h
    terrain.height_field_raw[
        x_start:x_end, y_start - step_width:y_start
    ] = current_h
    terrain.height_field_raw[
        x_end:x_end + step_width, y_start:y_end
    ] = current_h


def long_t_shaped_stairs_terrain(terrain, horizontal_scale, vertical_scale):
    """
    长上下台阶
    """
    def to_idx(m): return int(m / horizontal_scale)
    def to_height(m): return int(m / vertical_scale)

    terrain.height_field_raw[:] = 0

    # 主体参数
    main_step_h = 0.1
    main_step_d = 0.3
    n_main_steps = 3
    platform_length_m = 1.0
    platform_height_m = 0.4  # 用户指定

    # 转换
    step_d_idx = to_idx(main_step_d)
    plat_len_idx = to_idx(platform_length_m)

    # 上部分：3级 + 1级过渡
    up_steps = n_main_steps + 1  # 3级主 + 1级到0.4m
    total_up_len = up_steps * step_d_idx

    # 下部分：1级过渡 + 3级主
    down_steps = 1 + n_main_steps
    total_down_len = down_steps * step_d_idx

    total_len = total_up_len + plat_len_idx + total_down_len
    start_x = (terrain.length - total_len) // 2
    x = start_x

    # --- 上台阶 ---
    for i in range(up_steps):
        h = to_height((i + 1) * main_step_h)
        x0 = x + i * step_d_idx
        x1 = x0 + step_d_idx
        if x0 < terrain.length:
            terrain.height_field_raw[x0:x1, :] = h
    x += total_up_len

    # --- 平台 (0.4m) ---
    h_plat = to_height(platform_height_m)
    x0 = x
    x1 = x + plat_len_idx
    if x0 < terrain.length:
        terrain.height_field_raw[x0:x1, :] = h_plat
    x = x1

    # --- 下台阶 ---
    for i in range(down_steps):
        h = to_height((down_steps - i - 1) * main_step_h)
        x0 = x + i * step_d_idx
        x1 = x0 + step_d_idx
        if x0 < terrain.length:
            terrain.height_field_raw[x0:x1, :] = h


def ramp_platform_full_width_terrain(terrain, horizontal_scale, vertical_scale):
    """
    全宽地形（Y方向全覆盖）：对称三角坡（去掉中间平台，上下坡直接相接于顶点 0.4m）
    - 上斜坡：从 0m 升到 0.4m
    - 下斜坡：从 0.4m 降回 0m
    """
    def to_idx(m):
        return int(m / horizontal_scale)
    
    def to_height(m): 
        return int(m / vertical_scale)

    terrain.height_field_raw[:] = 0

    # === 参数 ===
    target_height_m = 0.4          # 顶点高度
    slope_angle_deg = 14           # 斜坡角度（上下对称）

    angle_rad = np.deg2rad(slope_angle_deg)
    slope_length_m = target_height_m / np.tan(angle_rad)  # 每个斜坡的水平长度

    # 转换为网格索引
    slope_len_idx = to_idx(slope_length_m)

    # 总长度 = 上坡 + 下坡（无平台，两坡直接相接于顶点）
    total_len_idx = slope_len_idx + slope_len_idx
    start_x = (terrain.length - total_len_idx) // 2
    current_x = start_x

    # === 1. 上斜坡：0m → 0.4m ===
    for i in range(slope_len_idx):
        ratio = i / slope_len_idx
        h = to_height(ratio * target_height_m)
        x_idx = current_x + i
        if 0 <= x_idx < terrain.length:
            terrain.height_field_raw[x_idx, :] = h  # 全宽
    current_x += slope_len_idx

    # === 2. 下斜坡：0.4m → 0m ===
    # i=0 时 ratio=1.0，下坡首列即 0.4m 顶点，与上坡自然衔接
    for i in range(slope_len_idx):
        ratio = 1.0 - (i / slope_len_idx)  # 从1降到0
        h = to_height(ratio * target_height_m)
        x_idx = current_x + i
        if 0 <= x_idx < terrain.length:
            terrain.height_field_raw[x_idx, :] = h  # 全宽


def gravel_chipwood_pit_terrain(terrain, horizontal_scale, vertical_scale):
    """
    生成 砂砾/碎木 坑地形。
    物理特性：
    - 左半侧：砂砾（细腻噪点，高摩擦感）
    - 右半侧：碎木（粗糙起伏，低摩擦/打滑感）
    - 四周：150mm 高护栏
    """
    import numpy as np
    
    def to_idx(m): return int(round(m / horizontal_scale))
    def to_height(m): return int(round(m / vertical_scale))

    # === 1. 基础尺寸 (米) ===
    pit_length = 4.0    # 总长 2000mm
    pit_width = 2.0     # 总宽 1000mm
    pit_depth = -0.1    # 坑深 100mm
    wall_height = 0.15  # 护栏高 150mm
    wall_thick = 0.15   # 护栏厚度

    # === 2. 计算网格坐标 ===
    cx, cy = terrain.length // 2, terrain.width // 2
    half_len = to_idx(pit_length / 2)
    half_wid = to_idx(pit_width / 2)
    w_thick = to_idx(wall_thick)
    
    # 坑的边界
    x0, x1 = cx - half_len, cx + half_len
    y0, y1 = cy - half_wid, cy + half_wid
    cy_mid = cy  # 中间分界线

    # === 3. 生成坑底材质 (噪点) ===
    

    # 加大参数以加大高度差
    # --- 左侧：砂砾 ---
    # 生成细腻的高斯噪声 (标准差小，模拟沙石)
    gravel_noise = np.random.normal(0, 0.025, (x1-x0, cy_mid-y0))
    terrain.height_field_raw[x0:x1, y0:cy_mid] = to_height(pit_depth) + (gravel_noise / vertical_scale).astype(int)

    # --- 右侧：碎木 ---
    # 生成粗糙的均匀噪声 (范围大，模拟边角料起伏)
    chipwood_noise = np.random.uniform(-0.06, 0.06, (x1-x0, y1-cy_mid))
    terrain.height_field_raw[x0:x1, cy_mid:y1] = to_height(pit_depth) + (chipwood_noise / vertical_scale).astype(int)

    # # === 4. 生成护栏 (突起) ===
    # h_wall = to_height(wall_height)

    # # 上下外墙
    # terrain.height_field_raw[x0-w_thick:x1+w_thick, y0-w_thick:y0] = h_wall
    # terrain.height_field_raw[x0-w_thick:x1+w_thick, y1:y1+w_thick] = h_wall
    
    # # 左右外墙
    # terrain.height_field_raw[x0-w_thick:x0, y0:y1] = h_wall
    # terrain.height_field_raw[x1:x1+w_thick, y0:y1] = h_wall
    
    # # 中间隔板 (区分两种材质)
    # terrain.height_field_raw[x0:x1, cy_mid:cy_mid+1] = h_wall

def dense_rubble_pit_terrain(terrain, horizontal_scale, vertical_scale):
    """
    独立地形：超密集乱石/碎木坑
    - 几何：2x1m 的深坑，四周有 0.15m 护栏
    - 材质：左侧密集砂砾 (8cm块)，右侧密集碎木 (12cm块)
    - 特点：极度密集且陡峭的台阶状地面
    """
    import numpy as np

    # === 辅助函数 ===
    def to_idx(m): return int(round(m / horizontal_scale))
    def to_height(m): return int(round(m / vertical_scale))

    # === 1. 基础几何设置 (坑与护栏) ===
    pit_length = 4.0    # 2000mm
    pit_width = 2.0     # 1000mm
    pit_depth = -0.1    # 100mm 深
    wall_height = 0.15  # 150mm 护栏
    wall_thick = 0.15   # 护栏厚度

    # 计算网格坐标
    cx, cy = terrain.length // 2, terrain.width // 2
    half_len = to_idx(pit_length / 2)
    half_wid = to_idx(pit_width / 2)
    w_thick = to_idx(wall_thick)
    
    x0, x1 = cx - half_len, cx + half_len
    y0, y1 = cy - half_wid, cy + half_wid
    cy_mid = cy

    # # === 2. 生成护栏 (高度 0.15m) ===
    # h_wall = to_height(wall_height)
    # # 上下外墙
    # terrain.height_field_raw[x0-w_thick:x1+w_thick, y0-w_thick:y0] = h_wall
    # terrain.height_field_raw[x0-w_thick:x1+w_thick, y1:y1+w_thick] = h_wall
    # # 左右外墙
    # terrain.height_field_raw[x0-w_thick:x0, y0:y1] = h_wall
    # terrain.height_field_raw[x1:x1+w_thick, y0:y1] = h_wall
    # # 中间隔板
    # terrain.height_field_raw[x0:x1, cy_mid:cy_mid+1] = h_wall

    # === 3. 生成超密集坑底 (核心逻辑) ===
    def create_rubble_grid(x_start, x_end, y_start, y_end, block_size_m, height_amp_m):
        """
        在指定区域生成密集的随机方块
        """
        bs_idx = to_idx(block_size_m)
        
        # 遍历网格生成随机高度
        for x in range(x_start, x_end, bs_idx):
            for y in range(y_start, y_end, bs_idx):
                # 随机生成当前方块的高度：在坑深基础上上下浮动
                h = np.random.uniform(pit_depth - height_amp_m, pit_depth + height_amp_m)
                
                # 应用高度
                terrain.height_field_raw[x:x+bs_idx, y:y+bs_idx] = to_height(h)

    # --- 左侧：超密集砂砾 ---
    # 块尺寸 0.08米 (8cm)，起伏 +/- 0.04米
    create_rubble_grid(x0, x1, y0, cy_mid, block_size_m=0.08, height_amp_m=0.04)

    # --- 右侧：超密集碎木 ---
    # 块尺寸 0.12米 (12cm)，起伏 +/- 0.07米
    create_rubble_grid(x0, x1, cy_mid, y1, block_size_m=0.12, height_amp_m=0.07)



def high_wall_terrain(terrain, horizontal_scale, vertical_scale):
    """
    生成横跨整个地形的"高墙"。
    特点：
    - 宽度：横跨整个地形（Y轴全覆盖），机器人无法绕行。
    - 高度：0.3m (300mm)。
    - 厚度：0.3m (增加厚度以便于观察和攀爬)。
    """
    def to_idx(m): return int(round(m / horizontal_scale))
    def to_height(m): return int(round(m / vertical_scale))

    # === 尺寸设置 ===
    height = 0.3     # 高度 300mm
    thickness = 0.1  # 厚度 100mm （50mm渲染不出来）

    # === 关键修改：让墙横跨整个 Y 轴 ===
    # 这样无论地形多宽，墙都是一整排，像一道闸门
    y_start = 0
    y_end = terrain.width
    
    # X 轴居中
    start_x_idx = (terrain.length - to_idx(thickness)) // 2
    end_x_idx = start_x_idx + to_idx(thickness)

    # === 生成墙体 ===
    # 将整个条状区域抬升到 0.3m
    terrain.height_field_raw[start_x_idx:end_x_idx, y_start:y_end] = to_height(height)


def grouped_high_platforms_terrain(terrain, horizontal_scale, vertical_scale,
                                   heights=(0.4, 0.5, 0.6),
                                   platform_length=1.5,
                                   platform_width=1.5,
                                   platform_center_offset_x=2.0,
                                   lateral_spacing=2.2):
    """在同一块地形中横向排列多个直壁高台。"""
    def to_size(m): return max(1, int(round(m / horizontal_scale)))
    def to_offset(m): return int(round(m / horizontal_scale))
    def to_height(m): return int(round(m / vertical_scale))

    terrain.height_field_raw[:] = 0
    length_idx = to_size(platform_length)
    width_idx = to_size(platform_width)
    center_x = terrain.length // 2 + to_offset(platform_center_offset_x)
    center_y = terrain.width // 2
    x_start = max(0, center_x - length_idx // 2)
    x_end = min(terrain.length, x_start + length_idx)

    middle_index = (len(heights) - 1) / 2.0
    for index, height in enumerate(heights):
        lateral_offset = (index - middle_index) * lateral_spacing
        platform_center_y = center_y + to_offset(lateral_offset)
        y_start = max(0, platform_center_y - width_idx // 2)
        y_end = min(terrain.width, y_start + width_idx)
        terrain.height_field_raw[
            x_start:x_end, y_start:y_end
        ] = to_height(height)


def rectangular_blocks_with_ramps_terrain(terrain, horizontal_scale, vertical_scale):
    """
    一排独立长方体 + 两端斜坡
    """
    def to_idx(m): return int(round(m / horizontal_scale))
    def to_height(m): return int(round(m / vertical_scale))

    terrain.height_field_raw[:] = 0

    # 尺寸定义
    block_width = 0.4      # 400mm
    block_height = 0.2     # 200mm
    block_length = 1.0     # 1000mm
    gap = 0.15             # 边缘到边缘间隔 150mm
    num_blocks = 6
    ramp_angle = 14        # 度

    # 索引计算
    width_idx = max(1, to_idx(block_width))
    length_idx = max(1, to_idx(block_length))
    height_val = to_height(block_height)
    gap_idx = max(1, to_idx(gap))
    ramp_len = max(1, to_idx(block_height / np.tan(np.deg2rad(ramp_angle))))

    cx, cy = terrain.length // 2, terrain.width // 2
    y0, y1 = cy - length_idx // 2, cy + length_idx // 2

    # 总宽度计算
    blocks_total = num_blocks * width_idx + (num_blocks - 1) * gap_idx
    total_width = ramp_len + blocks_total + ramp_len
    x = cx - total_width // 2  # 当前X位置

    # --- 左斜坡 ---
    for i in range(ramp_len):
        h = int((i + 1) / ramp_len * height_val)
        terrain.height_field_raw[x + i, y0:y1] = h
    x += ramp_len
    print(f"[Left Ramp] end at x={x}")

    # --- 6个独立长方体 ---
    for i in range(num_blocks):
        # 设置当前块
        terrain.height_field_raw[x:x + width_idx, y0:y1] = height_val
        print(f"[Block {i+1}] x[{x}:{x + width_idx}]")
        
        x += width_idx  # 移动到块末尾
        
        # 跳过间隔（最后一块不需要）
        if i < num_blocks - 1:
            x += gap_idx

    # --- 右斜坡 ---
    for i in range(ramp_len):
        h = int((1 - (i + 1) / ramp_len) * height_val)
        terrain.height_field_raw[x + i, y0:y1] = h
    print(f"[Right Ramp] start at x={x}")



def rectangular_blocks_row_terrain(terrain, horizontal_scale, vertical_scale):
    """
    一排长方体障碍物
    单个尺寸：宽400mm, 高200mm, 长1000mm
    沿宽度方向每隔150mm生成一个，共6个
    """
    def to_idx(m): return int(round(m / horizontal_scale))
    def to_height(m): return int(round(m / vertical_scale))

    terrain.height_field_raw[:] = 0

    # 单个长方体尺寸 (米)
    block_width = 0.2      # 宽 400mm (X方向)
    block_height = 0.2     # 高 200mm
    block_length = 1.5     # 长 1000mm (Y方向)
    
    # 间隔 (米)
    gap = 0.10             # 间隔 150mm
    
    # 数量
    num_blocks = 3

    # 中心点
    cx, cy = terrain.length // 2, terrain.width // 2

    # 计算索引
    width_idx = max(1, to_idx(block_width))
    length_idx = max(1, to_idx(block_length))
    height_val = to_height(block_height)
    gap_idx = to_idx(gap)
    
    # 中心到中心距离 = 宽度 + 间隔
    step = width_idx + gap_idx
    
    # 计算起始位置（使整体居中）
    total_width = num_blocks * width_idx + (num_blocks - 1) * gap_idx
    start_x = cx - total_width // 2

    # 生成6个长方体
    for i in range(num_blocks):
        x_start = start_x + i * step
        x_end = x_start + width_idx
        y_start = cy - length_idx // 2
        y_end = cy + length_idx // 2
        
        terrain.height_field_raw[x_start:x_end, y_start:y_end] = height_val



def right_angle_pole_terrain(terrain, horizontal_scale, vertical_scale):
    """
    直角绕杆地形 - 简化版
    
    结构：
    - 4个圆柱体（半径 150mm，高 500mm）
    - 基准杆1个
    - 从基准杆沿 +X 方向：隔 1000mm 放杆2、杆3
    - 从基准杆沿 +Y 方向：隔 1000mm 放杆4
    - 形成直角的 L 形路线
    """
    import numpy as np
    
    def to_idx(m): return int(round(m / horizontal_scale))
    def to_height(m): return int(round(m / vertical_scale))
    
    # === 清空地形 ===
    terrain.height_field_raw[:] = 0
    
    # === 圆柱体参数 ===
    cylinder_radius = 0.15   # 150mm 半径
    cylinder_height = 0.50   # 500mm 高
    spacing = 1.0            # 杆间距 1000mm
    
    # === 计算网格坐标 ===
    cx, cy = terrain.length // 2, terrain.width // 2
    
    # 基准杆位置
    pole_base_x = cx
    pole_base_y = cy
    
    # 杆2、杆3 位置（沿 +X 方向）
    pole2_x = cx + to_idx(spacing)
    pole2_y = cy
    
    pole3_x = cx + to_idx(spacing * 2)
    pole3_y = cy
    
    # 杆4 位置（沿 +Y 方向）
    pole4_x = cx
    pole4_y = cy + to_idx(spacing)
    
    # === 绘制圆柱体函数 ===
    def draw_cylinder(cx_idx, cy_idx, radius_idx, height_val, name=""):
        """绘制圆柱体"""
        for dx in range(-radius_idx, radius_idx + 1):
            for dy in range(-radius_idx, radius_idx + 1):
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= radius_idx:
                    x, y = cx_idx + dx, cy_idx + dy
                    if 0 <= x < terrain.length and 0 <= y < terrain.width:
                        terrain.height_field_raw[x, y] = height_val
        if name:
            print(f"  {name}: pos=({cx_idx}, {cy_idx}), h={height_val}")
    
    # === 绘制4个圆柱体 ===
    radius_idx = to_idx(cylinder_radius)
    height_val = to_height(cylinder_height)
    
    draw_cylinder(pole_base_x, pole_base_y, radius_idx, height_val, 
                  "🟢 Pole 1 (基准杆)")
    draw_cylinder(pole2_x, pole2_y, radius_idx, height_val,
                  "🟢 Pole 2 (+X 方向)")
    draw_cylinder(pole3_x, pole3_y, radius_idx, height_val,
                  "🟢 Pole 3 (+X 方向)")
    draw_cylinder(pole4_x, pole4_y, radius_idx, height_val,
                  "🟢 Pole 4 (+Y 方向)")
    
    print(f"\n[Right Angle Pole Terrain]")
    print(f"  Cylinder radius: {cylinder_radius}m")
    print(f"  Cylinder height: {cylinder_height}m")
    print(f"  Spacing: {spacing}m")
    print(f"  Layout: L-shape with 90° angle")

# ========== Monkey Patch: 让 selected_terrain 生成多种地形（每种一次） ==========


def multi_terrain_selected(self):
    from isaacgym import terrain_utils
    import pprint # 用于美观打印参数

    # --- 1. 定义地形配置 ---
    terrain_configs = [
        # 随机地形台阶
        (terrain_utils.discrete_obstacles_terrain, {
            "max_height": 0.12,
            "min_size": 1.0,       
            "max_size": 2.0,        
            "num_rects": 20,        
            "platform_size": 3.0
        }),

        # 金字塔台阶：每级20cm。
        (terrain_utils.pyramid_stairs_terrain, {
            "step_width": 0.3,
            "step_height": 0.2,
            "platform_size": 1.0,
        }),


        # 深坑
        # (terrain_utils.stepping_stones_terrain, {
        #     "stone_size": 1.2,
        #     "stone_distance": 0.1,
        #     "max_height": 0.0,
        #     "platform_size": 4.0
        # }),
    

        # T型台阶
        (t_shaped_stairs_terrain, {
            "horizontal_scale": self.cfg.horizontal_scale,
            "vertical_scale": self.cfg.vertical_scale
        }),
        # 不平的地面
        (gravel_chipwood_pit_terrain, {
            "horizontal_scale": self.cfg.horizontal_scale,
            "vertical_scale": self.cfg.vertical_scale
        }),
        # 碎木
        (dense_rubble_pit_terrain, {
            "horizontal_scale": self.cfg.horizontal_scale,
            "vertical_scale": self.cfg.vertical_scale
        }),
         # 添加长高墙
        (high_wall_terrain, {
            "horizontal_scale": self.cfg.horizontal_scale,
            "vertical_scale": self.cfg.vertical_scale
        }),
        
        # 无坡木块。
        (rectangular_blocks_row_terrain, {
            "horizontal_scale": self.cfg.horizontal_scale,
            "vertical_scale": self.cfg.vertical_scale
        }),

        # 40cm、50cm、60cm 三个直壁高台集中在同一块地形中。
        (grouped_high_platforms_terrain, {
            "horizontal_scale": self.cfg.horizontal_scale,
            "vertical_scale": self.cfg.vertical_scale,
        }),
    ]

    total_sub = self.cfg.num_rows * self.cfg.num_cols
    num_types = len(terrain_configs)

    # 网格必须和复杂地形数量完全一致，禁止额外生成纯平地。
    if total_sub != num_types:
        raise ValueError(
            f"Terrain grid has {total_sub} slots, but {num_types} complex "
            "terrains are configured. Set num_rows * num_cols exactly equal "
            "to the terrain count."
        )



    # --- 4. 遍历并生成地形 ---
    for k in range(len(terrain_configs)):
        (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
        func, kwargs = terrain_configs[k]

        # 获取函数名
        func_name = getattr(func, '__name__', 'Unknown Function')

        # 打印当前地形信息
        # 格式化参数字典，使其在一行内显示或紧凑显示
        params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        print(f"  [{k:2d}] Grid({i},{j}) | Type: {func_name}")
        print(f"       └─ Params: {params_str}")

        # 初始化 SubTerrain
        sub_terrain = terrain_utils.SubTerrain(
            "sub_terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,  # 修正：使用 length_per_env_pixels
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )

        # 调用地形生成函数
        func(sub_terrain, **kwargs)
        
        # 添加到地图
        self.add_terrain_to_map(sub_terrain, i, j)

  

# 应用 monkey patch
Terrain.selected_terrain = multi_terrain_selected

# ========== 全局键盘状态 ==========
vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
height_cmd = 0.25  
height_cmd_min = 0.20
height_cmd_max = 0.30
height_cmd_default = 0.25
reset_flag = False
exit_flag = False

def on_press(key):
    global vx_cmd, vy_cmd, wz_cmd, height_cmd, reset_flag, exit_flag
    try:
        if key.char == 'w': vx_cmd = 2.0
        elif key.char == 's': vx_cmd = -2.0
        elif key.char == 'a': vy_cmd = 1.0
        elif key.char == 'd': vy_cmd = -1.0
        elif key.char == 'q': wz_cmd = -2.0
        elif key.char == 'e': wz_cmd = 2.0
        elif key.char == 'r':  # R：蹲下（降低）
            height_cmd = max(height_cmd_min, height_cmd - 0.02)
        elif key.char == 'f':  # F：站起（升高）
            height_cmd = min(height_cmd_max, height_cmd + 0.02)
        elif key.char == 'z':  # Z：重置高度
            height_cmd = height_cmd_default
        elif key.char == 'p':  # P：重置机器人位置
            reset_flag = True
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_flag = True

def on_release(key):
    global vx_cmd, vy_cmd, wz_cmd
    try:
        if key.char in 'ws': vx_cmd = 0.0
        elif key.char in 'ad': vy_cmd = 0.0
        elif key.char in 'qe': wz_cmd = 0.0
    except AttributeError:
        pass


def align_camera_with_robot(env):
    """以机器人实际位置为注视中心初始化相机。"""
    robot_position = env.root_states[0, :3].detach().cpu().tolist()
    camera_position = [
        robot_position[0] + 3.0,
        robot_position[1] + 3.0,
        robot_position[2] + 1.5,
    ]
    env.set_camera(camera_position, robot_position)
    print(
        "[INFO] Robot spawn / camera target: "
        f"({robot_position[0]:.2f}, {robot_position[1]:.2f}, "
        f"{robot_position[2]:.2f})"
    )


# ========== 主函数 ==========
def play(args, model_path):
    global reset_flag, exit_flag, height_cmd
    global height_cmd_min, height_cmd_max, height_cmd_default

    if args.headless:
        print("Keyboard control requires GUI. Run without --headless.")
        return

    # --- 1. 获取配置 ---
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    height_cmd_min, height_cmd_max = env_cfg.commands.ranges.range_height
    height_cmd_default = env_cfg.rewards.base_height_target
    height_cmd = height_cmd_default

    env_cfg.terrain.selected = True     # True为从文件中的terrain_configs中选择地形，False则是训练的默认地形     
    # 注意：不再设置 terrain_kwargs，因为 monkey patch 已接管
    # 4x2 正好容纳8种复杂地形，不再保留或补齐纯平地块。
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 2
    
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.env.episode_length_s = 1e6
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.env.test = True
    env_cfg.terrain.border_size = 0.0  # 场景间距
    # 根据任务动作数动态计算观测。普通四足为46维，W1W为58维。
    one_step_obs = 9 + 3 * env_cfg.env.num_actions + 1
    env_cfg.env.num_one_step_observations = one_step_obs
    env_cfg.env.num_observations = one_step_obs * 6

    # 根据 checkpoint 自动匹配 Critic：旧平地模型为 64，rough-terrain
    # 模型额外包含 187 点高度扫描，因此为 251。
    critic_obs_dim = get_checkpoint_critic_obs_dim(model_path)
    proprioceptive_critic_dim = one_step_obs + 3 + 3
    terrain_critic_dim = proprioceptive_critic_dim + 187
    if critic_obs_dim == proprioceptive_critic_dim:
        env_cfg.terrain.measure_heights = False
    elif critic_obs_dim == terrain_critic_dim:
        env_cfg.terrain.measure_heights = True
    else:
        raise SystemExit(
            "Unsupported checkpoint Critic input dimension: "
            f"{critic_obs_dim}. Expected {proprioceptive_critic_dim} "
            f"(without terrain scan) or {terrain_critic_dim} "
            "(with 187-point terrain scan)."
        )
    env_cfg.env.num_one_step_privileged_obs = critic_obs_dim
    env_cfg.env.num_privileged_obs = critic_obs_dim
    env_cfg.commands.num_commands = 5

    print(
        f"[INFO] Checkpoint Critic input: {critic_obs_dim} | "
        f"terrain scan: {env_cfg.terrain.measure_heights}"
    )

    # --- 2. 创建环境 ---
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    align_camera_with_robot(env)

    print(
        f"[INFO] Observation dimension: {env.num_obs} "
        f"({env.num_one_step_obs} per step)"
    )

    # 新增：monkey patch _check_termination 来禁用自动重置（除超时外），如摔倒不重置
    def disable_auto_reset(self):
        self.reset_buf = self.time_out_buf.clone()  # 只用超时作为重置条件

    env._check_termination = MethodType(disable_auto_reset, env)

   # --- 3. 创建 runner 但不自动加载 ---
    train_cfg.runner.resume = False  # 禁用自动加载
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, 
        name=args.task, 
        args=args, 
        train_cfg=train_cfg
    )
    
    print(f"[INFO] Loading policy from: {model_path}")
    
    # 直接调用 runner.load() 传入完整路径
    ppo_runner.load(model_path)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    print(
        f"[✓] HIMActorCritic loaded successfully! "
        f"(Estimator input: {env.num_obs}D)"
    )

    # --- 4. 键盘监听 ---
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\n=== Keyboard Control (Multi-Terrain with Height) ===")
    print("Movement:")
    print("  W/S: forward/backward (±1.5 m/s)")
    print("  A/D: strafe left/right (±1.5 m/s)")
    print("  Q/E: turn left/right (±1.0 rad/s)")
    print("\nHeight Control (站立/蹲下):")
    print(f"  F: 站起 ↑ (最大 {height_cmd_max:.2f}m)")
    print(f"  R: 蹲下 ↓ (最小 {height_cmd_min:.2f}m)")
    print(f"  Z: 重置高度到默认 ({height_cmd_default:.2f}m)")
    print("\nOther:")
    print("  P: 重置机器人位置")
    print("  ESC: 退出\n")
    

    try:
        obs = env.get_observations()
        step_count = 0
        while not exit_flag:
            env.commands[0, 0] = vx_cmd
            env.commands[0, 1] = vy_cmd
            env.commands[0, 2] = wz_cmd
            env.commands[0, 4] = height_cmd  # ✅ 新增：设置高度指令


            with torch.no_grad():
                actions = policy(obs)  # ← 直接调用 policy

            obs, privileged_obs, rewards, dones, infos, timeouts, _ = env.step(actions)

            step_count += 1
            if step_count % 100 == 0:
                base_height = env.root_states[0, 2].item()
                print(f"[Step {step_count}] Height: {base_height:.3f}m | "
                      f"Command: {height_cmd:.3f}m | "
                      f"Velocity: ({vx_cmd:.2f}, {vy_cmd:.2f}, {wz_cmd:.2f})")

            # 仅响应手动复位
            if reset_flag:
                env.reset_idx(torch.tensor([0], device=env.device))
                obs = env.get_observations()
                align_camera_with_robot(env)
                reset_flag = False

            env.render()

    finally:
        listener.stop()
        if hasattr(env, 'viewer') and env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

# ========== 入口 ==========
if __name__ == '__main__':
    model_path = pop_custom_args()
    args = get_args()
    play(args, model_path)
