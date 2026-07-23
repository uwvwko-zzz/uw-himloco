import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

# 这段代码定义了一个 Terrain 类，用于在 Isaac Gym + Legged Gym 框架中动态生成复杂、多样化的仿真地形（height field），
# 支持 课程学习（Curriculum）、随机地形组合 和 指定地形类型 三种模式。 

# 它将整个仿真世界划分为 num_rows × num_cols 个子地形块（sub-terrains），
# 每个块可以是斜坡、台阶、障碍物、坑洞、间隙（gap）、碎石地等，用于训练四足机器人在复杂环境中行走、跳跃、跨越障碍等能力。 

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        # 每个子地形的物理尺寸
        self.env_length = cfg.terrain_length       
        self.env_width = cfg.terrain_width
        # 累计概率，用于随机选择子地形类型
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)

    # 随机地形组合 
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    # 从简单到复杂，从平坦到多样
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # 最后一行 difficulty=1.0（最高墙 30cm），第一行 difficulty=0（最矮墙 5cm）
                difficulty = i / (self.cfg.num_rows - 1) if self.cfg.num_rows > 1 else 1.0
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    # 从配置中读取指定地形（）官方
    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,      
                horizontal_scale=self.cfg.horizontal_scale   
            )
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs)  
            self.add_terrain_to_map(terrain, i, j)

    

    # 地形工厂（专训宽顶平台）
    # difficulty 由课程学习控制：第 0 行最低(5cm)，最后一行最高(30cm)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        # 专训平台：只生成宽顶平台，难度完全由 difficulty 控制
        # （choice 参数保留以兼容调用接口，但不再用于地形选择）
        platform_terrain(
            terrain,
            horizontal_scale=self.cfg.horizontal_scale,
            vertical_scale=self.cfg.vertical_scale,
            difficulty=difficulty,
            wall_x_offsets=getattr(self.cfg, 'wall_x_offsets', None),
            platform_length=getattr(self.cfg, 'platform_length', 0.8),
        )
        return terrain

# 添加地形
    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]



# def t_shaped_stairs_terrain(terrain, horizontal_scale, vertical_scale):
#     """
#     根据图示尺寸自定义 T 字形台阶地形
#     尺寸说明 (单位: mm -> m):
#     - 顶部平台: 1.0m x 1.0m, 高 0.4m
#     - 台阶: 宽 0.3m, 高 0.1m (共3级台阶连接到 0.4m 平台)
#     - 总长度: 2.8m (0.9 + 1.0 + 0.9)
#     - 总宽度: 1.9m (1.0 + 0.9)
#     """
    
#     # 转换物理尺寸为网格索引
#     def to_idx(m):
#         return int(m / horizontal_scale)
    
#     def to_height(m):
#         return int(m / vertical_scale)

#     # 计算中心点（将台阶放在地形区域的正中心）
#     center_x = terrain.length // 2
#     center_y = terrain.width // 2

#     # 基本参数定义
#     platform_size = to_idx(1.0)
#     step_width = to_idx(0.3)
#     step_height = to_height(0.1)
    
#     # 1. 绘制顶部核心平台 (1000x1000, 高 400mm)
#     x_start = center_x - platform_size // 2
#     x_end = x_start + platform_size
#     y_start = center_y - platform_size // 2
#     y_end = y_start + platform_size
    
#     terrain.height_field_raw[x_start:x_end, y_start:y_end] = to_height(0.4)

#     # 2. 绘制三侧台阶 (每侧 3 级)
#     for i in range(1, 4):  # i=1, 2, 3 分别代表从高到低的 3 级台阶
#         current_h = to_height(0.4 - i * 0.1)
#         offset_min = (i - 1) * step_width
#         offset_max = i * step_width
        
#         # --- 右侧台阶 (+Y方向) ---
#         step_y_start = y_end + offset_min
#         step_y_end = y_end + offset_max
#         terrain.height_field_raw[x_start:x_end, step_y_start:step_y_end] = current_h
        
#         # --- 左侧台阶 (-Y方向) ---
#         step_y_start = y_start - offset_max
#         step_y_end = y_start - offset_min
#         terrain.height_field_raw[x_start:x_end, step_y_start:step_y_end] = current_h
        
#         # --- 前侧台阶 (+X方向) ---
#         # 注意：前侧台阶宽度与平台一致 (1000mm)
#         step_x_start = x_end + offset_min
#         step_x_end = x_end + offset_max
#         terrain.height_field_raw[step_x_start:step_x_end, y_start:y_end] = current_h

# def ramp_platform_full_width_terrain(terrain, horizontal_scale, vertical_scale):
#     """
#     去掉两边侧壁的版本：将坡道和平台扩展至整个地形块的宽度
#     """
#     # 辅助转换函数
#     def to_idx(m): return int(m / horizontal_scale)
#     def to_height(m): return int(m / vertical_scale)

#     # 1. 先将整个地形块初始化为平地（高度0）
#     terrain.height_field_raw[:] = 0

#     # 2. 参数设置
#     platform_length_m = 1.0  # 平台长度 1.0m
#     max_h_m = 0.2           # 高度 200mm
#     max_h_raw = to_height(max_h_m)
#     angle_rad = np.deg2rad(14)
    
#     # 计算 14° 斜坡所需的水平长度
#     slope_length_m = max_h_m / np.tan(angle_rad)
    
#     # 转换成索引
#     platform_len_idx = to_idx(platform_length_m)
#     slope_len_idx = to_idx(slope_length_m)
    
#     # 3. 确定在 X 轴上的起始位置（居中放置）
#     # 总占用长度 = 坡道长 + 平台长
#     total_len_idx = slope_len_idx + platform_len_idx
#     start_x = (terrain.length - total_len_idx) // 2
    
#     # 4. 绘制斜坡部分 (全宽渲染，y 轴直接取 0 到 terrain.width)
#     for i in range(slope_len_idx):
#         # 线性插值计算高度
#         h = to_height((i / slope_len_idx) * max_h_m)
#         curr_x = start_x + i
#         if 0 <= curr_x < terrain.length:
#             terrain.height_field_raw[curr_x, :] = h # 注意这里用 ":" 代表覆盖全宽

#     # 5. 绘制平台部分 (全宽渲染)
#     plat_start_x = start_x + slope_len_idx
#     plat_end_x = plat_start_x + platform_len_idx
#     if plat_start_x < terrain.length:
#         terrain.height_field_raw[plat_start_x:plat_end_x, :] = max_h_raw

#     # 这样修改后，坡道会横跨整个“赛道”，侧面就没有落差产生的三角形面片了


# # 自定义地形函数
# # 在地形中央挖一个 “十字形”深坑（-1000），但保留中心小平台（高度=0）
# # 模拟 跨越沟壑 的场景
# def gap_terrain(terrain, gap_size, platform_size=1.):
#     gap_size = int(gap_size / terrain.horizontal_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale)

#     center_x = terrain.length // 2
#     center_y = terrain.width // 2
#     x1 = (terrain.length - platform_size) // 2
#     x2 = x1 + gap_size
#     y1 = (terrain.width - platform_size) // 2
#     y2 = y1 + gap_size
   
#     terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
#     terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

# # 在中央挖一个 方形坑洞，深度由 depth 控制
# # 模拟 掉入坑中 的挑战
# def pit_terrain(terrain, depth, platform_size=1.):
#     depth = int(depth / terrain.vertical_scale)
#     platform_size = int(platform_size / terrain.horizontal_scale / 2)
#     x1 = terrain.length // 2 - platform_size
#     x2 = terrain.length // 2 + platform_size
#     y1 = terrain.width // 2 - platform_size
#     y2 = terrain.width // 2 + platform_size
#     terrain.height_field_raw[x1:x2, y1:y2] = -depth


def t_shaped_stairs_terrain(terrain, horizontal_scale, vertical_scale):
    # 长台阶
    """
    - 3级上台阶（每级0.1m高、0.3m深） → 总高0.3m
    - 紧接着一个0.1m高的过渡台阶（达到0.4m）
    - 0.4m高平台（长1.0m）
    - 紧接着一个0.1m低的过渡台阶（降到0.3m）
    - 3级下台阶（每级0.1m） → 回到0m
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


def gravel_chipwood_pit_terrain(terrain,  horizontal_scale, vertical_scale):
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

  
    max_pit_length = 5.0    
    max_pit_width = 5.0     
    pit_length = max_pit_length
    pit_width = max_pit_width
    pit_depth = -0.1    # 坑深 100mm
    wall_height = 0.15  # 护栏高 150mm
    wall_thick = 0.15   # 护栏厚度

    # === 2. 计算网格坐标，clip到terrain边界 ===
    cx, cy = terrain.length // 2, terrain.width // 2
    half_len = to_idx(pit_length / 2)
    half_wid = to_idx(pit_width / 2)
    w_thick = to_idx(wall_thick)
    
    # 坑的理论边界
    theo_x0, theo_x1 = cx - half_len, cx + half_len
    theo_y0, theo_y1 = cy - half_wid, cy + half_wid
    theo_cy_mid = cy  # 中间分界线

    # 有效边界（clip到[0, length/width]）
    x0 = max(0, theo_x0)
    x1 = min(terrain.length, theo_x1)
    y0 = max(0, theo_y0)
    y1 = min(terrain.width, theo_y1)
    cy_mid = min(max(theo_cy_mid, 0), terrain.width)  # clip cy_mid

    # 左侧有效范围
    y0_left = max(0, theo_y0)
    cy_mid_left = min(terrain.width, theo_cy_mid)
    eff_shape_x = x1 - x0
    eff_shape_y_left = max(0, cy_mid_left - y0_left)  # 避免负数

    # 右侧有效范围
    cy_mid_right = max(0, theo_cy_mid)
    y1_right = min(terrain.width, theo_y1)
    eff_shape_y_right = max(0, y1_right - cy_mid_right)

    if eff_shape_x <= 0 or (eff_shape_y_left <= 0 and eff_shape_y_right <= 0):
        return  # 如果有效区域为空，跳过

    # === 3. 生成坑底材质 (噪点)，用有效形状生成 ===
    
    # --- 左侧：砂砾 ---
    if eff_shape_y_left > 0:
        gravel_noise = np.random.normal(0, 0.025, (eff_shape_x, eff_shape_y_left))
        terrain.height_field_raw[x0:x1, y0_left:cy_mid_left] = to_height(pit_depth) + (gravel_noise / vertical_scale).astype(int)

    # --- 右侧：碎木 ---
    if eff_shape_y_right > 0:
        chipwood_noise = np.random.uniform(-0.06, 0.06, (eff_shape_x, eff_shape_y_right))
        terrain.height_field_raw[x0:x1, cy_mid_right:y1_right] = to_height(pit_depth) + (chipwood_noise / vertical_scale).astype(int)

    # === 4. 生成护栏 (突起)，添加clip ===
    # h_wall = to_height(wall_height)

    # # 上下外墙（clip范围）
    # wall_y0_low = max(0, theo_y0 - w_thick)
    # wall_y0_up = min(terrain.width, theo_y0)
    # if wall_y0_up > wall_y0_low:
    #     terrain.height_field_raw[max(0, theo_x0 - w_thick):min(terrain.length, theo_x1 + w_thick), wall_y0_low:wall_y0_up] = h_wall

    # wall_y1_low = max(0, theo_y1)
    # wall_y1_up = min(terrain.width, theo_y1 + w_thick)
    # if wall_y1_up > wall_y1_low:
    #     terrain.height_field_raw[max(0, theo_x0 - w_thick):min(terrain.length, theo_x1 + w_thick), wall_y1_low:wall_y1_up] = h_wall
    
    # # 左右外墙
    # wall_x0_low = max(0, theo_x0 - w_thick)
    # wall_x0_up = min(terrain.length, theo_x0)
    # if wall_x0_up > wall_x0_low:
    #     terrain.height_field_raw[wall_x0_low:wall_x0_up, max(0, theo_y0):min(terrain.width, theo_y1)] = h_wall

    # wall_x1_low = max(0, theo_x1)
    # wall_x1_up = min(terrain.length, theo_x1 + w_thick)
    # if wall_x1_up > wall_x1_low:
    #     terrain.height_field_raw[wall_x1_low:wall_x1_up, max(0, theo_y0):min(terrain.width, theo_y1)] = h_wall
    
    # # 中间隔板
    # mid_low = max(0, theo_cy_mid)
    # mid_up = min(terrain.width, theo_cy_mid + 1)
    # if mid_up > mid_low:
    #     terrain.height_field_raw[max(0, theo_x0):min(terrain.length, theo_x1), mid_low:mid_up] = h_wall

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

   
    max_pit_length = 5.0    
    max_pit_width = 5.0     
    pit_length = max_pit_length
    pit_width = max_pit_width
    pit_depth = -0.1    # 100mm 深
    wall_height = 0.15  # 150mm 护栏
    wall_thick = 0.15   # 护栏厚度

    # 计算网格坐标，clip到边界
    cx, cy = terrain.length // 2, terrain.width // 2
    half_len = to_idx(pit_length / 2)
    half_wid = to_idx(pit_width / 2)
    w_thick = to_idx(wall_thick)
    
    theo_x0, theo_x1 = cx - half_len, cx + half_len
    theo_y0, theo_y1 = cy - half_wid, cy + half_wid
    theo_cy_mid = cy

    x0 = max(0, theo_x0)
    x1 = min(terrain.length, theo_x1)
    y0 = max(0, theo_y0)
    y1 = min(terrain.width, theo_y1)
    cy_mid = min(max(theo_cy_mid, 0), terrain.width)

    if x1 <= x0 or y1 <= y0:
        return  # 有效区域为空，跳过

    # === 2. 生成护栏 (高度 0.15m)，添加clip ===
    # h_wall = to_height(wall_height)
    # # 上下外墙
    # terrain.height_field_raw[max(0, theo_x0 - w_thick):min(terrain.length, theo_x1 + w_thick), max(0, theo_y0 - w_thick):min(terrain.width, theo_y0)] = h_wall
    # terrain.height_field_raw[max(0, theo_x0 - w_thick):min(terrain.length, theo_x1 + w_thick), max(0, theo_y1):min(terrain.width, theo_y1 + w_thick)] = h_wall
    # # 左右外墙
    # terrain.height_field_raw[max(0, theo_x0 - w_thick):min(terrain.length, theo_x0), max(0, theo_y0):min(terrain.width, theo_y1)] = h_wall
    # terrain.height_field_raw[max(0, theo_x1):min(terrain.length, theo_x1 + w_thick), max(0, theo_y0):min(terrain.width, theo_y1)] = h_wall
    # # 中间隔板
    # terrain.height_field_raw[max(0, theo_x0):min(terrain.length, theo_x1), max(0, theo_cy_mid):min(terrain.width, theo_cy_mid + 1)] = h_wall

    # === 3. 生成超密集坑底 (核心逻辑) ===
    def create_rubble_grid(x_start, x_end, y_start, y_end, block_size_m, height_amp_m):
        """
        在指定区域生成密集的随机方块，添加范围clip
        """
        bs_idx = max(1, to_idx(block_size_m))  # 避免bs_idx=0
        # clip循环范围
        clipped_x_start = max(x_start, 0)
        clipped_x_end = min(x_end, terrain.length)
        clipped_y_start = max(y_start, 0)
        clipped_y_end = min(y_end, terrain.width)
        
        for x in range(clipped_x_start, clipped_x_end, bs_idx):
            for y in range(clipped_y_start, clipped_y_end, bs_idx):
                # clip方块边界
                block_x_end = min(x + bs_idx, clipped_x_end)
                block_y_end = min(y + bs_idx, clipped_y_end)
                if block_x_end <= x or block_y_end <= y:
                    continue
                # 随机生成当前方块的高度：在坑深基础上上下浮动
                h = np.random.uniform(pit_depth - height_amp_m, pit_depth + height_amp_m)
                
                # 应用高度
                terrain.height_field_raw[x:block_x_end, y:block_y_end] = to_height(h)

    # --- 左侧：超密集砂砾 ---
    # 块尺寸 0.08米 (8cm)，起伏 +/- 0.04米
    create_rubble_grid(x0, x1, y0, cy_mid, block_size_m=0.08, height_amp_m=0.04)

    # --- 右侧：超密集碎木 ---
    # 块尺寸 0.12米 (12cm)，起伏 +/- 0.07米
    create_rubble_grid(x0, x1, cy_mid, y1, block_size_m=0.12, height_amp_m=0.07)


# def high_wall_terrain(terrain, horizontal_scale, vertical_scale):
#     """
#         生成横跨整个地形的“高墙”。
#         特点：
#         - 宽度：横跨整个地形（Y轴全覆盖），机器人无法绕行。
#         - 高度：0.3m (300mm)。
#         - 厚度：0.3m (增加厚度以便于观察和攀爬)。
#     """
#     def to_idx(m): return int(round(m / horizontal_scale))
#     def to_height(m): return int(round(m / vertical_scale))

#     # === 尺寸设置 ===
#     height = 0.3     # 高度 300mm
#     thickness = 0.1  # 厚度 100mm （50mm渲染不出来）

#     # === 关键修改：让墙横跨整个 Y 轴 ===
#     # 这样无论地形多宽，墙都是一整排，像一道闸门
#     y_start = 0
#     y_end = terrain.width
    
#     # X 轴居中
#     start_x_idx = (terrain.length - to_idx(thickness)) // 2
#     end_x_idx = start_x_idx + to_idx(thickness)

#     # === 生成墙体 ===
#     # 将整个条状区域抬升到 0.3m
#     terrain.height_field_raw[start_x_idx:end_x_idx, y_start:y_end] = to_height(height)

def platform_terrain(terrain, horizontal_scale, vertical_scale, difficulty=0.5,
                     wall_x_offsets=None, platform_length=1.0):
    """
    生成多个横跨整个 Y 轴的宽顶平台。

    特点：
    - 每个平台横跨整个 Y 轴，机器人无法绕行。
    - 高度随 difficulty 从 0.05m 渐变到 0.30m（课程学习）。
    - 平台沿 X 方向默认长 0.8m，顶部可稳定落足。
    - 位置：wall_x_offsets 给出每个平台相对块中心的 X 偏移（米）。
            默认在 spawn 前方 2m 放置 1 个高台。
            ⚠️ 必须与 _reward_wall_crossing 读取的 cfg.terrain.wall_x_offsets 完全一致，
            否则奖励判定的墙位置与实际墙位置错位。
    """
    def to_idx(m):
        return int(round(m / horizontal_scale))

    def to_height(m):
        return int(round(m / vertical_scale))

    # === 尺寸设置 ===
    height = 0.05 + 0.25 * difficulty   # 5cm → 30cm

    # 默认布局：每个子地形只放置 1 个高台。
    if wall_x_offsets is None:
        wall_x_offsets = [2.0]

    platform_length_idx = max(1, to_idx(platform_length))
    platform_height_idx = to_height(height)

    # 横跨整个 Y 轴
    y_start = 0
    y_end = terrain.width
    cx = terrain.length // 2  # 块中心（网格索引）

    # === 按 wall_x_offsets 放置每道墙 ===
    for offset_m in wall_x_offsets:
        center_x = cx + to_idx(offset_m)
        start_x_idx = center_x - platform_length_idx // 2
        end_x_idx = start_x_idx + platform_length_idx

        # 防止越界
        start_x_idx = max(0, start_x_idx)
        end_x_idx = min(terrain.length, end_x_idx)

        terrain.height_field_raw[start_x_idx:end_x_idx, y_start:y_end] = platform_height_idx


# 兼容仍然导入旧函数名的外部测试脚本。
high_wall_terrain = platform_terrain


def speed_bump_terrain(terrain, horizontal_scale, vertical_scale, difficulty=0.5):
    """
    生成 减速带 地形。
    特点：
    - 每条减速带横跨整个 Y 轴（全宽），机器人无法绕行。
    - 梯形横截面：边缘 1cm → 3cm → 顶 5cm(平顶) → 3cm → 1cm，共 6 格宽（0.6m）。
    - 高度固定 5cm，不随难度缩放；难度只控制减速带的条数。
    - 条数随课程难度：num_bumps = 1 + round(difficulty * 3)，简单行 1 条、困难行最多 4 条。
    - 各条减速带沿 X 方向均匀分布。
    """
    def to_idx(m):
        return int(round(m / horizontal_scale))

    def to_height(m):
        return int(round(m / vertical_scale))

    # === 先将整个地形块初始化为平地（高度0）===
    terrain.height_field_raw[:] = 0

    # === 几何参数（固定不缩放）===
    # 梯形横截面：单位 [m]，从一侧边缘到另一侧
    # [1cm, 3cm, 5cm, 5cm, 3cm, 1cm] → 共 6 格宽（0.6m）
    profile_m = [0.01, 0.03, 0.05, 0.05, 0.03, 0.01]
    profile_idx = [to_height(h) for h in profile_m]

    # 横跨整个 Y 轴
    y_start = 0
    y_end = terrain.width

    # 条数随课程难度（每条几何不变，仍是固定 5cm 梯形）
    num_bumps = 1 + int(round(difficulty * 3))

    # === 沿 X 方向均匀放置减速带 ===
    bump_width = len(profile_idx)  # 6
    for k in range(num_bumps):
        center_x = int(terrain.length * (k + 1) / (num_bumps + 1))
        start_x_idx = center_x - bump_width // 2

        # 按梯形轮廓逐列写入高度
        for i, h_idx in enumerate(profile_idx):
            x_idx = start_x_idx + i
            # 防止越界
            if 0 <= x_idx < terrain.length:
                terrain.height_field_raw[x_idx, y_start:y_end] = h_idx
