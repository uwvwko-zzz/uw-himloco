#!/usr/bin/env python3
"""
测试机器人初始关节角度的脚本 - 动态调整版本
用于可视化和实时调整 default_joint_angles 配置下的机器人姿态

使用示例：
    python legged_gym/scripts/test_angle.py --robot jqg

视角控制：
- 鼠标左键拖拽：旋转视角
- 鼠标中键拖拽：平移视角
- 鼠标滚轮：缩放
- 鼠标右键拖拽：缩放（上下移动）

关节控制：
- 按键 [ / ]：选择上一个/下一个关节
- 按键 + / -：增加/减少当前关节角度
- 按键 0：重置当前关节到默认角度
- 按键 P：打印当前所有关节角度（可复制到代码中）
"""

import os
import math
import sys
from pathlib import Path

# 必须先导入 isaacgym
import isaacgym
from isaacgym import gymapi, gymutil

# 支持从 legged_gym/scripts 或任意其他目录直接运行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legged_gym import LEGGED_GYM_ROOT_DIR
import legged_gym.envs  # noqa: F401，导入时完成所有任务注册
from legged_gym.utils import task_registry

def main():
    # ==================== 加载机器人配置 ====================
    args = gymutil.parse_arguments(
        description="查看机器人默认关节角度",
        custom_parameters=[
            {
                "name": "--robot",
                "type": str,
                "default": "dog",
                "help": "已在 legged_gym.envs 中注册的机器人名称，例如 dog、go1、jqg",
            },
        ],
    )

    available_robots = sorted(task_registry.env_cfgs)
    if args.robot not in task_registry.env_cfgs:
        raise ValueError(
            f"机器人 '{args.robot}' 未注册，可选机器人: "
            f"{', '.join(available_robots)}"
        )

    robot_cfg, _ = task_registry.get_cfgs(name=args.robot)
    asset_path = robot_cfg.asset.file.format(
        LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
    )
    asset_root = os.path.dirname(asset_path)
    urdf_file = os.path.basename(asset_path)
    default_joint_angles = dict(robot_cfg.init_state.default_joint_angles)

    
    # 角度调整步长（弧度）
    angle_step = 0.05  # 约 2. 86 度
    angle_step_fine = 0.01  # 精细调整，约 0.57 度
    
    # 初始位置高度
    init_height = robot_cfg.init_state.pos[2]
    
    # 是否固定基座
    fix_base = True
    
    # ==================== 初始化 Isaac Gym ====================
    gym = gymapi.acquire_gym()
    
    # 仿真参数
    sim_params = gymapi. SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")

    # 创建 viewer
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    viewer = gym.create_viewer(sim, camera_props)
    if viewer is None: 
        raise RuntimeError("Failed to create viewer")

    # 订阅键盘事件 - 视角控制
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_F, "toggle_fix")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "view_front")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "view_side")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "view_top")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi. KEY_4, "view_back")
    
    # 订阅键盘事件 - 关节控制
    gym. subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT_BRACKET, "prev_joint")   # [
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT_BRACKET, "next_joint")  # ]
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_EQUAL, "increase_angle")      # = (+ 键)
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_MINUS, "decrease_angle")      # -
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_0, "reset_joint")             # 0
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "print_angles")            # P
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "increase_angle_fine")    # 上箭头 - 精细增加
    gym. subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "decrease_angle_fine")  # 下箭头 - 精细减少
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "prev_joint")           # 左箭头
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "next_joint")          # 右箭头

    # 添加地面
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = 0
    gym.add_ground(sim, plane_params)

    # 设置初始相机位置
    cam_pos = gymapi.Vec3(1.5, 1.5, init_height + 0.6)
    cam_target = gymapi.Vec3(0, 0, init_height * 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # ==================== 加载机器人 ====================
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fix_base
    # 查看模型时保留所有 link，方便确认每个 visual 和关节的位置。
    asset_options.collapse_fixed_joints = False
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.flip_visual_attachments = robot_cfg.asset.flip_visual_attachments
    # SolidWorks 导出的 STL 在 Isaac Gym 中可能因面法线/顶点法线处理而
    # 出现部分 link 不显示，加载时统一重算法线。
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

    print(f"机器人配置: {args.robot}")
    print(f"Loading asset from: {asset_path}")
    asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
    if asset is None:
        raise RuntimeError(f"Failed to load asset: {urdf_file}")

    num_dofs = gym.get_asset_dof_count(asset)
    dof_names = gym.get_asset_dof_names(asset)
    num_bodies = gym.get_asset_rigid_body_count(asset)
    body_names = gym.get_asset_rigid_body_names(asset)
    
    print(f"\n===== 机器人关节信息 =====")
    print(f"刚体数量: {num_bodies}")
    print(f"刚体名称: {body_names}")
    print(f"关节数量: {num_dofs}")
    print(f"关节名称: {dof_names}")

    # 创建环境
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
    
    pose = gymapi.Transform()
    pose.p = gymapi. Vec3(0, 0, init_height)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    
    actor = gym.create_actor(env, asset, pose, "robot", 0, 1)

    # 调试显示：不同 link 使用不同颜色，便于区分重叠的 visual。
    debug_colors = (
        gymapi.Vec3(0.85, 0.85, 0.85),
        gymapi.Vec3(0.95, 0.45, 0.15),
        gymapi.Vec3(0.20, 0.65, 0.95),
        gymapi.Vec3(0.25, 0.85, 0.35),
        gymapi.Vec3(0.95, 0.85, 0.15),
    )
    for body_index in range(num_bodies):
        gym.set_rigid_body_color(
            env,
            actor,
            body_index,
            gymapi.MESH_VISUAL,
            debug_colors[body_index % len(debug_colors)],
        )

    # ==================== 设置关节角度 ====================
    dof_props = gym.get_actor_dof_properties(env, actor)
    
    for i in range(num_dofs):
        dof_props['stiffness'][i] = 100.0
        dof_props['damping'][i] = 10.0
        dof_props['driveMode'][i] = int(gymapi.DOF_MODE_POS)
    gym.set_actor_dof_properties(env, actor, dof_props)

    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    
    # 当前关节角度（可动态修改）
    current_angles = {}
    
    print(f"\n===== 设置关节角度 =====")
    for i, name in enumerate(dof_names):
        if name in default_joint_angles:
            angle = default_joint_angles[name]
            dof_states['pos'][i] = angle
            current_angles[name] = angle
            print(f"  {name}: {angle:.3f} rad ({angle * 180 / math.pi:.1f} deg)")
        else: 
            print(f"  {name}: 未在配置中找到，使用默认值 0")
            dof_states['pos'][i] = 0.0
            current_angles[name] = 0.0

    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    
    # 当前选中的关节索引
    selected_joint_idx = 0
    
    def update_targets():
        """更新所有关节的目标位置"""
        targets = []
        for i, name in enumerate(dof_names):
            targets.append(current_angles. get(name, 0.0))
        gym.set_actor_dof_position_targets(env, actor, targets)
        return targets
    
    def print_current_angles():
        """打印当前所有关节角度（可复制格式）"""
        print(f"\n{'='*50}")
        print("    当前关节角度配置（可复制到代码中）")
        print('='*50)
        print("default_joint_angles = {")
        for name in dof_names: 
            angle = current_angles. get(name, 0.0)
            print(f"    '{name}': {angle:.4f},")
        print("}")
        print('='*50 + "\n")
    
    def print_selected_joint():
        """打印当前选中的关节信息"""
        name = dof_names[selected_joint_idx]
        angle = current_angles. get(name, 0.0)
        print(f">>> 选中关节 [{selected_joint_idx + 1}/{num_dofs}]:  {name} = {angle:.3f} rad ({angle * 180 / math.pi:.1f} deg)")
    
    targets = update_targets()

    # 让 Isaac Gym 更新一次运动学状态，然后输出各 link 的世界坐标。
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    body_states = gym.get_actor_rigid_body_states(
        env, actor, gymapi.STATE_POS
    )
    body_positions = body_states["pose"]["p"]
    print("\n===== 刚体世界坐标 =====")
    for name, position in zip(body_names, body_positions):
        print(
            f"  {name:14s}: "
            f"x={position['x']:.3f}, "
            f"y={position['y']:.3f}, "
            f"z={position['z']:.3f}"
        )

    # ==================== 打印控制说明 ====================
    print(f"\n{'='*60}")
    print("                    控制说明")
    print('='*60)
    print("  【视角控制】")
    print("  鼠标左键拖拽   : 旋转视角")
    print("  鼠标中键拖拽   : 平移视角")
    print("  鼠标滚轮       : 缩放")
    print("  按键 1/2/3/4   : 前/侧/俯/后视图")
    print("-"*60)
    print("  【关节控制】")
    print("  按键 [ �� ←    : 选择上一个关节")
    print("  按键 ] 或 →    : 选择下一个关节")
    print("  按键 + (=)     : 增加角度 (+0.05 rad)")
    print("  按键 -         : 减少角度 (-0.05 rad)")
    print("  按键 ↑         : 精细增加角度 (+0.01 rad)")
    print("  按键 ↓         : 精细减少角度 (-0.01 rad)")
    print("  按键 0         :  重置当前关节到默认值")
    print("  按键 R         : 重置所有关节到默认值")
    print("  按键 P         : 打印当前所有角度（可复制）")
    print("-"*60)
    print("  按键 ESC       : 退出")
    print('='*60)
    print(f"  初始高度: {init_height} m")
    print(f"  固定基座: {fix_base}")
    print('='*60 + "\n")
    
    print_selected_joint()

    # ==================== 主循环 ====================
    frame = 0
    
    while not gym. query_viewer_has_closed(viewer):
        # 处理键盘事件
        for evt in gym.query_viewer_action_events(viewer):
            if evt. value > 0:  # 按��按下
                if evt.action == "reset":
                    # 重置所有关节到默认值
                    for name in dof_names:
                        if name in default_joint_angles:
                            current_angles[name] = default_joint_angles[name]
                        else:
                            current_angles[name] = 0.0
                    targets = update_targets()
                    # 同时重置状态
                    for i, name in enumerate(dof_names):
                        dof_states['pos'][i] = current_angles.get(name, 0.0)
                        dof_states['vel'][i] = 0.0
                    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
                    print("所有关节已重置到默认值")
                    print_selected_joint()
                    
                elif evt.action == "prev_joint":
                    selected_joint_idx = (selected_joint_idx - 1) % num_dofs
                    print_selected_joint()
                    
                elif evt.action == "next_joint": 
                    selected_joint_idx = (selected_joint_idx + 1) % num_dofs
                    print_selected_joint()
                    
                elif evt. action == "increase_angle":
                    name = dof_names[selected_joint_idx]
                    current_angles[name] = current_angles.get(name, 0.0) + angle_step
                    targets = update_targets()
                    print_selected_joint()
                    
                elif evt.action == "decrease_angle": 
                    name = dof_names[selected_joint_idx]
                    current_angles[name] = current_angles.get(name, 0.0) - angle_step
                    targets = update_targets()
                    print_selected_joint()
                    
                elif evt.action == "increase_angle_fine": 
                    name = dof_names[selected_joint_idx]
                    current_angles[name] = current_angles. get(name, 0.0) + angle_step_fine
                    targets = update_targets()
                    print_selected_joint()
                    
                elif evt.action == "decrease_angle_fine": 
                    name = dof_names[selected_joint_idx]
                    current_angles[name] = current_angles. get(name, 0.0) - angle_step_fine
                    targets = update_targets()
                    print_selected_joint()
                    
                elif evt.action == "reset_joint":
                    name = dof_names[selected_joint_idx]
                    if name in default_joint_angles: 
                        current_angles[name] = default_joint_angles[name]
                    else:
                        current_angles[name] = 0.0
                    targets = update_targets()
                    print(f"关节 {name} 已重置到默认值")
                    print_selected_joint()
                    
                elif evt.action == "print_angles":
                    print_current_angles()
                    
                elif evt.action == "view_front":
                    gym.viewer_camera_look_at(viewer, None, 
                        gymapi.Vec3(0, -2, init_height),
                        gymapi.Vec3(0, 0, init_height * 0.5))
                    print("切换到:  前视图")
                    
                elif evt.action == "view_side": 
                    gym. viewer_camera_look_at(viewer, None, 
                        gymapi.Vec3(2, 0, init_height),
                        gymapi.Vec3(0, 0, init_height * 0.5))
                    print("切换到: 侧视图")
                    
                elif evt.action == "view_top": 
                    gym.viewer_camera_look_at(viewer, None, 
                        gymapi.Vec3(0, 0, 2), gymapi.Vec3(0, 0, 0))
                    print("切换到: 俯视图")
                    
                elif evt.action == "view_back":
                    gym.viewer_camera_look_at(viewer, None, 
                        gymapi.Vec3(0, 2, init_height),
                        gymapi.Vec3(0, 0, init_height * 0.5))
                    print("切换到: 后视图")
                    
                elif evt.action == "toggle_fix":
                    print("注意: 切换固定基座需要重新加载机器人，请重启脚本并修改 fix_base 参数")

        # 仿真步进
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 持续应用目标位置
        gym. set_actor_dof_position_targets(env, actor, targets)
        
        # 更新图形
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        
        frame += 1

    # 清理
    gym. destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done!")


if __name__ == "__main__": 
    main()
