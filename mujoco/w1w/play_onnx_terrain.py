#!/usr/bin/env python3
"""Run the W1W ONNX policy on the complex MuJoCo obstacle course."""

from copy import deepcopy
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from play_onnx import KeyboardController, main


SCRIPT_DIR = Path(__file__).resolve().parent
W1W_XML = SCRIPT_DIR / "w1w.xml"
TERRAIN_XML = SCRIPT_DIR / "terrain_source.xml"
FRONT_CAMERA = "robot_front_camera"
TOP_CAMERA = "robot_top_camera"
MIN_BOX_HALF_THICKNESS = 0.05
# Isaac Gym trains with zero restitution and limits depenetration velocity.
# MuJoCo has no direct equivalent for the latter, so use a softer,
# over-damped contact transition to prevent stair edges from launching wheels.
TERRAIN_CONTACT_PARAMS = {
    "condim": "3",
    "friction": "0.6 0.005 0.0001",
    "solref": "0.03 2.0",
    "solimp": "0.8 0.95 0.01 0.5 2",
    "margin": "0.005",
}
VALIDATION_PLATFORM_SPECS = (
    ("validation_platform_30cm", 8.0, 0.30, "0.30 0.75 0.95 1"),
    ("validation_platform_40cm", 10.0, 0.40, "0.35 0.85 0.55 1"),
    ("validation_platform_50cm", 12.0, 0.50, "0.95 0.75 0.25 1"),
    ("validation_platform_60cm", 14.0, 0.60, "0.90 0.35 0.25 1"),
)
NOMINAL_BASE_HEIGHT = 0.48
FALL_CLEARANCE = 0.01


def vector(text):
    return np.asarray([float(value) for value in text.split()])


def vector_text(values):
    return " ".join(f"{value:.6g}" for value in values)


def quat_rotation(quat):
    w, x, y, z = quat
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def quat_multiply(left, right):
    """Multiply MuJoCo-format quaternions (w, x, y, z)."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray([
        lw*rw - lx*rx - ly*ry - lz*rz,
        lw*rx + lx*rw + ly*rz - lz*ry,
        lw*ry - lx*rz + ly*rw + lz*rx,
        lw*rz + lx*ry - ly*rx + lz*rw,
    ])


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half_angle = angle / 2.0
    return np.concatenate((
        [np.cos(half_angle)], axis * np.sin(half_angle)
    ))


def yaw_only_quat(quat):
    """Extract heading while discarding the current roll and pitch."""
    w, x, y, z = quat
    yaw = np.arctan2(
        2.0 * (w*z + x*y),
        1.0 - 2.0 * (y*y + z*z),
    )
    return axis_angle_quat((0.0, 0.0, 1.0), yaw)


def thicken_collision_boxes(root):
    """Prevent thin ramps and steps from being tunneled through by wheels."""
    for geom in root.findall(".//geom[@type='box']"):
        if geom.get("contype") == "0" or geom.get("conaffinity") == "0":
            continue
        if geom.get("size") is None or geom.get("pos") is None:
            continue
        size = vector(geom.get("size"))
        if size.size != 3 or size[2] >= MIN_BOX_HALF_THICKNESS:
            continue
        old = size[2]
        size[2] = MIN_BOX_HALF_THICKNESS
        geom.set("size", vector_text(size))
        rotation = quat_rotation(vector(geom.get("quat", "1 0 0 0")))
        position = vector(geom.get("pos"))
        position -= (size[2] - old) * rotation[:, 2]
        geom.set("pos", vector_text(position))


def configure_terrain_contacts(robot_root, terrain_root):
    """Apply the same damped contact model to wheels and terrain surfaces."""
    contact_geoms = []

    # terrain_source.xml's <default> section is not copied into the combined
    # model, therefore set parameters explicitly on every terrain collision.
    for geom in terrain_root.findall("./worldbody//geom"):
        if geom.get("contype") == "0" or geom.get("conaffinity") == "0":
            continue
        contact_geoms.append(geom)

    # Set the other side of wheel-terrain contacts explicitly as well. This
    # avoids MuJoCo mixing the tuned terrain parameters with wheel defaults.
    for foot_name in ("fl_foot", "fr_foot", "rl_foot", "rr_foot"):
        wheel_geom = robot_root.find(
            f".//body[@name='{foot_name}']/geom[@type='cylinder']"
        )
        if wheel_geom is None:
            raise ValueError(f"w1w.xml does not contain wheel geom '{foot_name}'")
        contact_geoms.append(wheel_geom)

    for geom in contact_geoms:
        for attribute, value in TERRAIN_CONTACT_PARAMS.items():
            geom.set(attribute, value)


def add_validation_obstacles(worldbody):
    """Add isolated ledges and a staircase with 25 cm risers."""
    # Four independent ledges are arranged in parallel lanes on the open side
    # of the original course. MuJoCo box sizes are half extents.
    for name, x, height, rgba in VALIDATION_PLATFORM_SPECS:
        ET.SubElement(worldbody, "geom", {
            "name": name,
            "type": "box",
            "pos": vector_text((x, -7.0, height / 2.0)),
            "size": vector_text((0.65, 0.65, height / 2.0)),
            "rgba": rgba,
        })

    # Three 25 cm risers up to a 75 cm platform, followed by two descending
    # blocks. The final 25 cm drop returns directly to the ground.
    stair_x = 8.0
    stair_half_width = 0.75
    tread_half_depth = 0.225
    stair_sections = (
        ("up_1", -3.80, 0.25),
        ("up_2", -3.35, 0.50),
        ("up_3", -2.90, 0.75),
        ("down_1", -1.45, 0.50),
        ("down_2", -1.00, 0.25),
    )
    for suffix, y, height in stair_sections:
        ET.SubElement(worldbody, "geom", {
            "name": f"validation_stairs_25cm_{suffix}",
            "type": "box",
            "pos": vector_text((stair_x, y, height / 2.0)),
            "size": vector_text((
                stair_half_width, tread_half_depth, height / 2.0
            )),
            "rgba": "0.62 0.48 0.28 1",
        })
    ET.SubElement(worldbody, "geom", {
        "name": "validation_stairs_25cm_top",
        "type": "box",
        "pos": vector_text((stair_x, -2.175, 0.375)),
        "size": vector_text((stair_half_width, 0.5, 0.375)),
        "rgba": "0.72 0.57 0.34 1",
    })


def build_combined_xml(output_path):
    """Combine the local W1W robot with the obstacle course only."""
    robot_root = ET.parse(W1W_XML).getroot()
    terrain_root = ET.parse(TERRAIN_XML).getroot()
    thicken_collision_boxes(terrain_root)
    add_validation_obstacles(terrain_root.find("worldbody"))
    configure_terrain_contacts(robot_root, terrain_root)

    compiler = robot_root.find("compiler")
    compiler.set("meshdir", str(W1W_XML.parent.parent.parent /
                                "resources/robots/w1w_urdf1/meshes"))
    option = robot_root.find("option")
    option.set("iterations", "100")
    option.set("integrator", "implicitfast")

    robot_asset = robot_root.find("asset")
    for item in list(robot_asset):
        if item.tag in {"texture", "material"}:
            robot_asset.remove(item)
    for item in terrain_root.find("asset"):
        if item.tag in {"texture", "material"}:
            robot_asset.insert(0, deepcopy(item))

    robot_world = robot_root.find("worldbody")
    robot_body = robot_world.find("./body[@name='base']")
    if robot_body is None:
        raise ValueError("w1w.xml does not contain body 'base'")
    ET.SubElement(robot_body, "camera", {
        "name": FRONT_CAMERA, "mode": "fixed", "pos": "0.25 0 0.5",
        "xyaxes": "0 -1 0 0.242536 0 0.970143", "fovy": "75",
    })
    ET.SubElement(robot_body, "camera", {
        "name": TOP_CAMERA, "mode": "fixed", "pos": "-0.8 0 1.0",
        "xyaxes": "0 -1 0 0.707107 0 0.707107", "fovy": "70",
    })
    for item in list(robot_world):
        robot_world.remove(item)
    for item in terrain_root.find("worldbody"):
        if item.tag == "body" and item.get("name") == "trunk":
            continue
        robot_world.append(deepcopy(item))
    robot_world.append(robot_body)
    ET.ElementTree(robot_root).write(
        output_path, encoding="utf-8", xml_declaration=True
    )


class TerrainKeyboardController(KeyboardController):
    GLFW_KEY_O = 79
    GLFW_KEY_P = 80
    GLFW_KEY_I = 73
    GLFW_KEY_SPACE = 32
    GLFW_KEY_M = 77
    GLFW_KEY_N = 78
    GLFW_KEY_B = 66

    def __init__(self, command, height):
        super().__init__(command, height)
        self.fixed_camera = None
        self.camera_changed = False
        self.push_requested = False
        self.push_force = np.zeros(3)
        self.push_steps = 0
        self.base_id = None
        self.fall_requested = None

    def key_callback(self, keycode):
        if keycode == self.GLFW_KEY_SPACE:
            self.push_requested = True
        elif keycode == self.GLFW_KEY_O:
            self.fixed_camera = FRONT_CAMERA
            self.camera_changed = True
        elif keycode == self.GLFW_KEY_I:
            self.fixed_camera = TOP_CAMERA
            self.camera_changed = True
        elif keycode == self.GLFW_KEY_P:
            self.fixed_camera = None
            self.camera_changed = True
        elif keycode == self.GLFW_KEY_M:
            self.fall_requested = "prone"
        elif keycode == self.GLFW_KEY_N:
            self.fall_requested = "side"
        elif keycode == self.GLFW_KEY_B:
            self.fall_requested = "random"
        else:
            super().key_callback(keycode)

    def reset_simulation_state(self):
        self.push_requested = False
        self.push_force.fill(0.0)
        self.push_steps = 0
        self.fall_requested = None

    def _is_robot_geom(self, model, geom_id):
        body_id = model.geom_bodyid[geom_id]
        return (
            body_id == self.base_id
            or model.body_rootid[body_id] == self.base_id
        )

    def _support_height(self, model, data):
        """Estimate the local supporting surface from vertical contacts."""
        contact_heights = []
        for index in range(data.ncon):
            contact = data.contact[index]
            first_is_robot = self._is_robot_geom(model, contact.geom1)
            second_is_robot = self._is_robot_geom(model, contact.geom2)
            if first_is_robot == second_is_robot:
                continue
            if abs(contact.frame[2]) < 0.5:
                continue
            contact_heights.append(float(contact.pos[2]))
        if contact_heights:
            return max(contact_heights)
        return float(data.qpos[2] - NOMINAL_BASE_HEIGHT)

    def _lowest_robot_collision_point(self, model, data):
        lowest = np.inf
        for geom_id in range(model.ngeom):
            if not self._is_robot_geom(model, geom_id):
                continue
            if model.geom_contype[geom_id] == 0:
                continue

            geom_type = model.geom_type[geom_id]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            size = model.geom_size[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                vertical_extent = np.dot(np.abs(rotation[2]), size)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                axis_z = abs(rotation[2, 2])
                vertical_extent = (
                    size[0] * np.sqrt(max(0.0, 1.0 - axis_z*axis_z))
                    + size[1] * axis_z
                )
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                axis_z = abs(rotation[2, 2])
                vertical_extent = size[1] * axis_z + size[0]
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                vertical_extent = size[0]
            else:
                vertical_extent = model.geom_rbound[geom_id]
            lowest = min(
                lowest,
                float(data.geom_xpos[geom_id, 2] - vertical_extent),
            )
        return lowest

    def _apply_fall_pose(self, model, data, mode):
        ground_height = self._support_height(model, data)
        heading = yaw_only_quat(data.qpos[3:7])

        if mode == "prone":
            tilt = axis_angle_quat((0.0, 1.0, 0.0), np.pi / 2.0)
            description = "趴倒"
        elif mode == "side":
            tilt = axis_angle_quat((1.0, 0.0, 0.0), np.pi / 2.0)
            description = "侧翻"
        else:
            tilt_axis_angle = np.random.uniform(0.0, 2.0 * np.pi)
            tilt_axis = (
                np.cos(tilt_axis_angle), np.sin(tilt_axis_angle), 0.0
            )
            tilt_angle = np.random.uniform(
                np.deg2rad(90.0), np.deg2rad(170.0)
            )
            tilt = axis_angle_quat(tilt_axis, tilt_angle)
            description = (
                f"随机倾倒（倾角 {np.degrees(tilt_angle):.1f}°）"
            )

        data.qpos[3:7] = quat_multiply(heading, tilt)
        data.qpos[2] = ground_height + NOMINAL_BASE_HEIGHT
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        self.push_requested = False
        self.push_force.fill(0.0)
        self.push_steps = 0

        mujoco.mj_forward(model, data)
        lowest = self._lowest_robot_collision_point(model, data)
        if np.isfinite(lowest):
            data.qpos[2] += ground_height + FALL_CLEARANCE - lowest
        mujoco.mj_forward(model, data)
        print(
            f"[FALL TEST] {description} | "
            f"surface z={ground_height:.3f} m"
        )

    def apply_external_forces(self, model, data):
        if self.base_id is None:
            self.base_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "base"
            )
        data.xfrc_applied[self.base_id, :] = 0.0
        if self.fall_requested is not None:
            mode = self.fall_requested
            self.fall_requested = None
            self._apply_fall_pose(model, data, mode)
            return True
        if self.push_requested:
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            magnitude = np.random.uniform(80.0, 180.0)
            duration = np.random.uniform(0.10, 0.25)
            self.push_force[:] = magnitude * np.array(
                [np.cos(angle), np.sin(angle), 0.0]
            )
            self.push_steps = max(1, int(np.ceil(duration / model.opt.timestep)))
            self.push_requested = False
            print(f"[PUSH] {magnitude:.1f} N, {np.degrees(angle):.1f} deg, "
                  f"{duration:.3f} s")
        if self.push_steps > 0:
            data.xfrc_applied[self.base_id, :3] = self.push_force
            self.push_steps -= 1
        return False


def update_camera(viewer, model, data, keyboard):
    if keyboard.fixed_camera is not None:
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, keyboard.fixed_camera
        )
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        keyboard.camera_changed = False
    elif keyboard.camera_changed:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        keyboard.camera_changed = False


def setup_viewer(viewer, model, data, keyboard):
    print("O: 机器人前视相机，I: 后上方相机，P: 返回跟随相机")
    print("空格: 施加一次随机水平推力")
    print("M: 趴倒，N: 侧翻，B: 随机倾倒（测试倾倒恢复）")
    print("新增高台: y=-7.0，x=8/10/12/14，高度=30/40/50/60 cm")
    print("新增楼梯: x=8.0，沿+y方向，每级高度=25 cm\n")
    print("地形接触: 过阻尼柔化参数已启用，减少轮子撞击棱角时的弹起\n")


def run():
    with tempfile.TemporaryDirectory(prefix="w1w_mujoco_terrain_") as directory:
        combined = Path(directory) / "w1w_terrain.xml"
        build_combined_xml(combined)
        main(
            xml_override=combined,
            keyboard_cls=TerrainKeyboardController,
            initial_position=(3.7, -9.0, 0.48),
            initial_quaternion=(-0.7071068, 0.0, 0.0, 0.7071068),
            camera_update=update_camera,
            setup_viewer=setup_viewer,
        )


if __name__ == "__main__":
    run()
