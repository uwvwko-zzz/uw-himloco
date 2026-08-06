#!/usr/bin/env python3
"""URDF → MJCF (MuJoCo) 转换脚本.

用法:
    python urdf_to_xml.py [urdf_path] [xml_path] [--spawn Z] [--effort-mult X]

会把 legged_gym 风格的 URDF 转成可直接给 mujoco 用的 MJCF：
  - 保留 URDF 中的 link/joint 命名
  - 自动生成 compiler / option / default / asset / worldbody / actuator / sensor
  - 视觉 geom 用 mesh, 碰撞 geom 用 box/cylinder/sphere (尺寸 URDF→MJCF 自动换算)
  - 自动检测 imu_link 并放置 site
  - 自动按关节树生成 motor 与 jointpos/jointvel 传感器
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom


# ----------------- 数学工具 -----------------

def rpy_to_quat(rpy):
    """URDF 的 rpy (roll-pitch-yaw, intrinsic ZYX) → MuJoCo quat [w x y z]."""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y_, z]


def fmt_vec(v, prec=6):
    """格式化向量, 去掉 -0 与 极小值."""
    out = []
    for x in v:
        if abs(x) < 1e-9:
            x = 0.0
        out.append(f"{x:.{prec}g}")
    return " ".join(out)


def parse_origin(elem):
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if elem is not None:
        if "xyz" in elem.attrib:
            xyz = [float(v) for v in elem.attrib["xyz"].split()]
        if "rpy" in elem.attrib:
            rpy = [float(v) for v in elem.attrib["rpy"].split()]
    return xyz, rpy


def set_pos_quat(elem, xyz, rpy):
    if any(abs(v) > 1e-9 for v in xyz):
        elem.set("pos", fmt_vec(xyz))
    if any(abs(v) > 1e-9 for v in rpy):
        elem.set("quat", fmt_vec(rpy_to_quat(rpy)))


# ----------------- URDF 解析 -----------------

def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    robot_name = root.attrib.get("name", "robot")

    links = {l.attrib["name"]: l for l in root.findall("link")}
    joints = list(root.findall("joint"))

    parent_of = {}      # child_link -> (joint, parent_link)
    children_of = {}    # parent_link -> [joint, ...]
    for j in joints:
        p = j.find("parent").attrib["link"]
        c = j.find("child").attrib["link"]
        parent_of[c] = (j, p)
        children_of.setdefault(p, []).append(j)

    root_links = [n for n in links if n not in parent_of]
    if not root_links:
        raise ValueError("URDF 中没有根 link (出现环?)")
    base_link = root_links[0]
    return root, robot_name, links, joints, parent_of, children_of, base_link


# ----------------- 几何转换 -----------------

def box_size(urdf_size):
    sx, sy, sz = [float(v) for v in urdf_size.split()]
    return [sx / 2, sy / 2, sz / 2]


def cylinder_size(radius, length):
    return [float(radius), float(length) / 2]


def sphere_size(radius):
    return [float(radius)]


def geometry_to_mjcf(geom):
    """URDF <geometry> → (mjcf_type, size_list, mesh_name_or_None)."""
    box = geom.find("box")
    if box is not None:
        return "box", box_size(box.attrib["size"]), None
    cyl = geom.find("cylinder")
    if cyl is not None:
        return "cylinder", cylinder_size(cyl.attrib["radius"], cyl.attrib["length"]), None
    sph = geom.find("sphere")
    if sph is not None:
        return "sphere", sphere_size(sph.attrib["radius"]), None
    mesh = geom.find("mesh")
    if mesh is not None:
        fn = mesh.attrib["filename"].split("/")[-1]
        name = fn.replace(".STL", "").replace(".stl", "").replace(".obj", "")
        return "mesh", None, name
    return None, None, None


# ----------------- 材质 -----------------

DEFAULT_MATERIALS = {
    "black": "0.10196 0.10196 0.10196 1",
    "silver": "0.79216 0.81961 0.93333 1",
    "gray": "0.69804 0.69804 0.69804 1",
    "grey": "0.69804 0.69804 0.69804 1",
    "white": "1 1 1 1",
    "red": "0.8 0.2 0.2 1",
    "blue": "0.2 0.2 0.8 1",
    "green": "0.2 0.8 0.2 1",
}


def collect_materials(urdf_root):
    """读取 URDF 顶层 <material> 定义, 与内置默认值合并."""
    mats = dict(DEFAULT_MATERIALS)
    for m in urdf_root.findall("material"):
        name = m.attrib.get("name")
        color = m.find("color")
        if name and color is not None and "rgba" in color.attrib:
            mats[name] = color.attrib["rgba"]
    return mats


def visual_color(visual, materials):
    m = visual.find("material")
    if m is None:
        return None
    color = m.find("color")
    if color is not None and "rgba" in color.attrib:
        return color.attrib["rgba"]
    name = m.attrib.get("name")
    return materials.get(name)


def hide(rgba):
    """把 alpha 置 0, 用于碰撞 geom."""
    parts = rgba.split()
    parts[3] = "0"
    return " ".join(parts)


# ----------------- mesh 资源 -----------------

def collect_meshes(links):
    seen = set()
    order = []
    for link in links.values():
        for v in link.findall("visual"):
            mesh = v.find("geometry/mesh")
            if mesh is not None:
                fn = mesh.attrib["filename"].split("/")[-1]
                if fn not in seen:
                    seen.add(fn)
                    order.append(fn)
    return order


# ----------------- 关节 -----------------

def joint_axis(joint):
    a = joint.find("axis")
    if a is not None and "xyz" in a.attrib:
        return a.attrib["xyz"]
    return "0 0 1"


def joint_limits(joint):
    lim = joint.find("limit")
    if lim is None:
        return None
    return {
        "lower": float(lim.attrib.get("lower", 0)),
        "upper": float(lim.attrib.get("upper", 0)),
        "effort": float(lim.attrib.get("effort", 0)),
        "velocity": float(lim.attrib.get("velocity", 0)),
    }


# ----------------- 构造 body -----------------

def build_body(parent_elem, link_name, link_elem, parent_joint,
               links, children_of, materials, is_base, settings):
    body = ET.SubElement(parent_elem, "body")
    body.set("name", link_name)

    # body 位置/姿态来自父关节的 origin
    if parent_joint is not None:
        xyz, rpy = parse_origin(parent_joint.find("origin"))
        set_pos_quat(body, xyz, rpy)
    elif is_base:
        # base spawn 高度
        body.set("pos", f"0 0 {settings['spawn_height']}")

    # 关节
    if is_base:
        ET.SubElement(body, "joint", {"type": "free"})
    else:
        jtype = parent_joint.attrib.get("type", "fixed")
        if jtype in ("revolute", "continuous"):
            j = ET.SubElement(body, "joint")
            j.set("name", parent_joint.attrib["name"])
            j.set("pos", "0 0 0")
            j.set("axis", joint_axis(parent_joint))
            lim = joint_limits(parent_joint)
            if lim is not None and jtype == "revolute":
                j.set("range", f"{lim['lower']} {lim['upper']}")
                if lim["effort"] > 0:
                    j.set("actuatorfrcrange", f"{-lim['effort']} {lim['effort']}")
        # fixed joint 不输出 (MuJoCo 里子 body 已经挂在父 body 下)

    # inertial
    inertial = link_elem.find("inertial")
    if inertial is not None:
        ie = ET.SubElement(body, "inertial")
        xyz, rpy = parse_origin(inertial.find("origin"))
        ie.set("pos", fmt_vec(xyz))
        if any(abs(v) > 1e-9 for v in rpy):
            ie.set("quat", fmt_vec(rpy_to_quat(rpy)))
        mass = inertial.find("mass")
        if mass is not None:
            ie.set("mass", mass.attrib["value"])
        inertia = inertial.find("inertia")
        if inertia is not None:
            ixx = inertia.attrib.get("ixx", "0")
            iyy = inertia.attrib.get("iyy", "0")
            izz = inertia.attrib.get("izz", "0")
            ixy = inertia.attrib.get("ixy", "0")
            ixz = inertia.attrib.get("ixz", "0")
            iyz = inertia.attrib.get("iyz", "0")
            if any(abs(float(v)) > 1e-12 for v in (ixy, ixz, iyz)):
                ie.set(
                    "fullinertia",
                    f"{ixx} {iyy} {izz} {ixy} {ixz} {iyz}",
                )
            else:
                ie.set("diaginertia", f"{ixx} {iyy} {izz}")

    # visual geom (mesh 优先, 否则用 primitive)
    for visual in link_elem.findall("visual"):
        geom_elem = visual.find("geometry")
        if geom_elem is None:
            continue
        gtype, size, mesh_name = geometry_to_mjcf(geom_elem)
        if gtype is None:
            continue
        g = ET.SubElement(body, "geom")
        if gtype == "mesh":
            g.set("type", "mesh")
            g.set("mesh", mesh_name)
        else:
            g.set("type", gtype)
            g.set("size", fmt_vec(size))
        g.set("contype", "0")
        g.set("conaffinity", "0")
        g.set("group", "1")
        set_pos_quat(g, *parse_origin(visual.find("origin")))
        rgba = visual_color(visual, materials)
        if rgba:
            g.set("rgba", rgba)

    # collision geom
    for collision in link_elem.findall("collision"):
        geom_elem = collision.find("geometry")
        if geom_elem is None:
            continue
        gtype, size, mesh_name = geometry_to_mjcf(geom_elem)
        if gtype is None:
            continue
        g = ET.SubElement(body, "geom")
        if gtype == "mesh":
            # mesh 碰撞: 让 MuJoCo 用 convex hull
            g.set("type", "mesh")
            g.set("mesh", mesh_name)
        else:
            g.set("type", gtype)
            g.set("size", fmt_vec(size))
        set_pos_quat(g, *parse_origin(collision.find("origin")))
        # 碰撞默认隐藏 (alpha=0)
        rgba = visual_color(collision, materials)
        g.set("rgba", hide(rgba) if rgba else "0.5 0.5 0.5 0")

    # 递归子 body
    for child_joint in children_of.get(link_name, []):
        child_link_name = child_joint.find("child").attrib["link"]
        # 跳过 imu_link (后面单独处理)
        if child_link_name == "imu_link":
            continue
        build_body(body, child_link_name, links[child_link_name], child_joint,
                   links, children_of, materials, False, settings)


def find_imu_offset(base_link, links, children_of):
    """在 base 的子关节里找连到 imu_link 的 fixed joint, 返回相对偏移."""
    for j in children_of.get(base_link, []):
        child = j.find("child").attrib["link"]
        if child == "imu_link":
            xyz, rpy = parse_origin(j.find("origin"))
            return xyz, rpy
    return None


# ----------------- 主流程 -----------------

def build_mjcf(urdf_root, robot_name, links, joints, parent_of, children_of, base_link,
               settings):
    mujoco_elem = ET.Element("mujoco", {"model": robot_name})
    materials = collect_materials(urdf_root)

    # compiler
    ET.SubElement(mujoco_elem, "compiler", {
        "angle": "radian",
        "meshdir": settings["meshdir"],
    })

    # size
    ET.SubElement(mujoco_elem, "size", {"njmax": "500", "nconmax": "100"})

    # option
    ET.SubElement(mujoco_elem, "option", {
        "gravity": "0 0 -9.806",
        "iterations": "50",
        "solver": "Newton",
        "timestep": "0.005",
    })

    # default
    default = ET.SubElement(mujoco_elem, "default")
    ET.SubElement(default, "geom", {
        "contype": "1", "conaffinity": "1",
        "friction": "0.6 0.3 0.3", "margin": "0.001", "group": "0",
    })
    ET.SubElement(default, "light", {"castshadow": "false", "diffuse": "1 1 1"})
    ET.SubElement(default, "joint", {
        "damping": "0.01", "armature": "0.01", "frictionloss": "0.2",
    })
    ET.SubElement(default, "camera", {"fovy": "60"})

    # asset
    asset = ET.SubElement(mujoco_elem, "asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient",
        "rgb1": "1.0 1.0 1.0", "rgb2": "0.6 0.8 1.0",
        "width": "512", "height": "512",
    })
    ET.SubElement(asset, "texture", {
        "name": "plane", "type": "2d", "builtin": "flat",
        "rgb1": "1 1 1", "rgb2": "1 1 1",
        "width": "512", "height": "512",
        "mark": "cross", "markrgb": "0 0 0",
    })
    ET.SubElement(asset, "material", {
        "name": "plane", "reflectance": "0.0",
        "texture": "plane", "texrepeat": "3 3", "texuniform": "true",
    })
    for mesh_file in collect_meshes(links):
        name = mesh_file.replace(".STL", "").replace(".stl", "").replace(".obj", "")
        ET.SubElement(asset, "mesh", {"name": name, "file": mesh_file})

    # visual
    visual = ET.SubElement(mujoco_elem, "visual")
    ET.SubElement(visual, "rgba", {
        "com": "0.502 1.0 0 0.5",
        "contactforce": "0.98 0.4 0.4 0.7",
        "contactpoint": "1.0 1.0 0.6 0.4",
    })
    ET.SubElement(visual, "scale", {
        "com": "0.2", "forcewidth": "0.035",
        "contactwidth": "0.10", "contactheight": "0.04",
    })

    # worldbody
    worldbody = ET.SubElement(mujoco_elem, "worldbody")
    ET.SubElement(worldbody, "light", {
        "directional": "true", "diffuse": ".8 .8 .8",
        "pos": "0 0 10", "dir": "0 0 -10",
    })
    ET.SubElement(worldbody, "camera", {
        "name": "track", "mode": "trackcom",
        "pos": "0 -1.3 1.6", "xyaxes": "1 0 0 0 0.707 0.707",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane",
        "conaffinity": "1", "condim": "3", "contype": "1",
        "rgba": "0.5 0.9 0.9 0.1", "material": "plane",
        "pos": "0 0 0", "size": "0 0 1",
    })

    # base body + 整棵腿树
    base_link_elem = links[base_link]
    build_body(worldbody, base_link, base_link_elem, None,
               links, children_of, materials, True, settings)

    # 在 base body 上放 imu site
    base_body = worldbody.find(f"body[@name='{base_link}']")
    if base_body is not None:
        site = ET.SubElement(base_body, "site", {"name": "imu"})
        off = find_imu_offset(base_link, links, children_of)
        if off is not None:
            xyz, _ = off
            site.set("pos", fmt_vec(xyz))
        else:
            site.set("pos", "0 0 0")

    # actuator: 对所有 revolute/continuous joint 生成 motor
    actuator = ET.SubElement(mujoco_elem, "actuator")
    rev_joints = []
    for j in joints:
        if j.attrib.get("type") in ("revolute", "continuous"):
            jname = j.attrib["name"]
            rev_joints.append(jname)
            lim = joint_limits(j) or {}
            effort = lim.get("effort", 0) * settings["effort_mult"]
            ET.SubElement(actuator, "motor", {
                "name": jname.replace("_joint", ""),
                "gear": "1",
                "joint": jname,
                "ctrlrange": f"{-effort} {effort}",
                "ctrllimited": "true",
            })

    # sensor: 每个关节 jointpos / jointvel + IMU
    sensor = ET.SubElement(mujoco_elem, "sensor")
    for jname in rev_joints:
        short = jname.replace("_joint", "")
        ET.SubElement(sensor, "jointpos", {"name": f"{short}_pos", "joint": jname})
    for jname in rev_joints:
        short = jname.replace("_joint", "")
        ET.SubElement(sensor, "jointvel", {"name": f"{short}_vel", "joint": jname})
    ET.SubElement(sensor, "accelerometer", {"name": "Body_Acc", "site": "imu"})
    ET.SubElement(sensor, "gyro", {"name": "Body_Gyro", "site": "imu"})
    ET.SubElement(sensor, "framepos", {
        "name": "Body_Pos", "objtype": "site", "objname": "imu",
    })
    ET.SubElement(sensor, "framequat", {
        "name": "Body_Quat", "objtype": "site", "objname": "imu",
    })

    return mujoco_elem


def prettify(elem):
    rough = ET.tostring(elem, "utf-8")
    parsed = minidom.parseString(rough)
    lines = [l for l in parsed.toprettyxml(indent="  ").split("\n") if l.strip()]
    return "\n".join(lines) + "\n"


def main():
    default_urdf = "/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/resources/robots/dog_t/dog_t/urdf/dog_t.urdf"
    default_xml = "/home/zhy/桌面/IsaacGym_Preview_4_Package/HIMLoco-main/himloco_gym/resources/robots/dog_t/dog_t/xml/dog_t.xml"

    parser = argparse.ArgumentParser(description="URDF → MJCF (MuJoCo) 转换")
    parser.add_argument("urdf", nargs="?", default=default_urdf)
    parser.add_argument("xml", nargs="?", default=default_xml)
    parser.add_argument("--spawn", type=float, default=0.45,
                        help="base 初始高度 (m), 默认 0.45")
    parser.add_argument("--effort-mult", type=float, default=1.0,
                        help="ctrlrange = URDF effort × 此系数, 默认 1.0")
    parser.add_argument(
        "--meshdir",
        default=None,
        help="MJCF meshdir；默认根据 URDF 的相邻 meshes 目录自动计算",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.xml))
    urdf_mesh_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(args.urdf)), "..", "meshes")
    )
    meshdir = args.meshdir or os.path.relpath(urdf_mesh_dir, out_dir)

    urdf_root, robot_name, links, joints, parent_of, children_of, base_link = parse_urdf(args.urdf)
    settings = {
        "spawn_height": args.spawn,
        "effort_mult": args.effort_mult,
        "meshdir": meshdir,
    }
    mjcf = build_mjcf(urdf_root, robot_name, links, joints, parent_of, children_of,
                      base_link, settings)

    os.makedirs(out_dir, exist_ok=True)
    with open(args.xml, "w") as f:
        f.write(prettify(mjcf))
    print(f"[OK] {args.urdf}\n -> {args.xml}")


if __name__ == "__main__":
    main()
