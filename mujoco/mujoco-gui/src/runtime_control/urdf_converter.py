"""Convert a URDF robot description into a standalone MuJoCo MJCF file."""

import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


SUPPORTED_JOINTS = {"fixed", "revolute", "continuous", "prismatic"}
MUJOCO_MESH_SUFFIXES = {".obj", ".stl", ".msh"}


def _floats(text, count, label):
    values = tuple(float(item) for item in str(text).split())
    if len(values) != count:
        raise ValueError("%s must contain %d numbers" % (label, count))
    return values


def _fmt(values):
    return " ".join("%.9g" % (0.0 if abs(value) < 1e-12 else value) for value in values)


def _indent(element, level=0):
    """Pretty-print an ElementTree on Python 3.8 and newer."""
    whitespace = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = whitespace + "  "
        for child in element:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = whitespace
    if level and (not element.tail or not element.tail.strip()):
        element.tail = whitespace


def _origin(parent):
    element = None if parent is None else parent.find("origin")
    if element is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return (
        _floats(element.get("xyz", "0 0 0"), 3, "origin xyz"),
        _floats(element.get("rpy", "0 0 0"), 3, "origin rpy"),
    )


def rpy_to_quat(rpy):
    """Convert URDF fixed-axis roll/pitch/yaw to MuJoCo ``wxyz``."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _set_pose(target, parent):
    xyz, rpy = _origin(parent)
    if any(abs(value) > 1e-12 for value in xyz):
        target.set("pos", _fmt(xyz))
    if any(abs(value) > 1e-12 for value in rpy):
        target.set("quat", _fmt(rpy_to_quat(rpy)))


def _safe_name(value):
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    return cleaned.strip("_") or "mesh"


@dataclass(frozen=True)
class MeshAsset:
    name: str
    path: Path
    scale: tuple


def _tag_name(element):
    return element.tag.rsplit("}", 1)[-1]


def _children(element, name):
    return [child for child in element if _tag_name(child) == name]


def _child(element, name):
    return next((child for child in element if _tag_name(child) == name), None)


def _collada_to_obj(source, destination):
    """Convert COLLADA triangle/polylist geometry to a MuJoCo-readable OBJ.

    This deliberately handles geometry only. URDF supplies the link transform,
    while common robot DAE files keep their mesh node transform at identity.
    """
    root = ET.parse(source).getroot()
    asset = _child(root, "asset")
    unit_scale = 1.0
    up_axis = "Z_UP"
    if asset is not None:
        unit = _child(asset, "unit")
        if unit is not None:
            unit_scale = float(unit.get("meter", "1"))
        axis = _child(asset, "up_axis")
        if axis is not None and axis.text:
            up_axis = axis.text.strip().upper()
    if up_axis not in {"X_UP", "Y_UP", "Z_UP"}:
        raise ValueError("unsupported COLLADA up_axis %r in %s" % (up_axis, source))

    vertices_out = []
    faces_out = []
    library = next((item for item in root.iter() if _tag_name(item) == "library_geometries"), None)
    if library is None:
        raise ValueError("COLLADA has no library_geometries: %s" % source)

    for geometry in _children(library, "geometry"):
        mesh = _child(geometry, "mesh")
        if mesh is None:
            continue
        arrays = {}
        for item in _children(mesh, "source"):
            array = _child(item, "float_array")
            technique = _child(item, "technique_common")
            accessor = None if technique is None else _child(technique, "accessor")
            if array is None or not array.text:
                continue
            stride = 1 if accessor is None else int(accessor.get("stride", "1"))
            values = [float(value) for value in array.text.split()]
            arrays[item.get("id")] = (values, stride)

        vertex_sources = {}
        for item in _children(mesh, "vertices"):
            for entry in _children(item, "input"):
                if entry.get("semantic") == "POSITION":
                    vertex_sources[item.get("id")] = entry.get("source", "").lstrip("#")

        position_cache = {}

        def output_vertex(source_id, source_index):
            key = (source_id, source_index)
            if key in position_cache:
                return position_cache[key]
            if source_id not in arrays:
                raise ValueError("missing COLLADA position source %s in %s" % (source_id, source))
            values, stride = arrays[source_id]
            start = source_index * stride
            if start + 2 >= len(values):
                raise ValueError("COLLADA position index out of range in %s" % source)
            x, y, z = (values[start + index] * unit_scale for index in range(3))
            if up_axis == "Y_UP":
                x, y, z = x, -z, y
            elif up_axis == "X_UP":
                x, y, z = -z, y, x
            vertices_out.append((x, y, z))
            result = len(vertices_out)
            position_cache[key] = result
            return result

        for primitive in list(mesh):
            primitive_name = _tag_name(primitive)
            if primitive_name not in {"triangles", "polylist", "polygons"}:
                continue
            inputs = _children(primitive, "input")
            if not inputs:
                continue
            stride = max(int(item.get("offset", "0")) for item in inputs) + 1
            vertex_input = next((item for item in inputs if item.get("semantic") in {"VERTEX", "POSITION"}), None)
            if vertex_input is None:
                raise ValueError("COLLADA primitive has no vertex input: %s" % source)
            offset = int(vertex_input.get("offset", "0"))
            source_id = vertex_input.get("source", "").lstrip("#")
            if vertex_input.get("semantic") == "VERTEX":
                source_id = vertex_sources.get(source_id)
            if not source_id:
                raise ValueError("cannot resolve COLLADA vertex source in %s" % source)

            streams = []
            if primitive_name == "polygons":
                streams = [([len(item.text.split()) // stride], item.text.split())
                           for item in _children(primitive, "p") if item.text]
            else:
                p = _child(primitive, "p")
                if p is None or not p.text:
                    continue
                if primitive_name == "triangles":
                    counts = [3] * int(primitive.get("count", "0"))
                else:
                    vcount = _child(primitive, "vcount")
                    if vcount is None or not vcount.text:
                        raise ValueError("COLLADA polylist has no vcount: %s" % source)
                    counts = [int(value) for value in vcount.text.split()]
                streams = [(counts, p.text.split())]

            for counts, raw_indices in streams:
                indices = [int(value) for value in raw_indices]
                cursor = 0
                for count in counts:
                    polygon = []
                    for vertex_index in range(count):
                        index = indices[cursor + vertex_index * stride + offset]
                        polygon.append(output_vertex(source_id, index))
                    cursor += count * stride
                    for index in range(1, len(polygon) - 1):
                        faces_out.append((polygon[0], polygon[index], polygon[index + 1]))

    if len(vertices_out) < 3 or not faces_out:
        raise ValueError("COLLADA contains no supported triangle geometry: %s" % source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii") as output:
        output.write("# Generated from %s by mujoco-gui\n" % source.name)
        for vertex in vertices_out:
            output.write("v %.9g %.9g %.9g\n" % vertex)
        for face in faces_out:
            output.write("f %d %d %d\n" % face)
    return destination


class UrdfConverter:
    """Stateful converter used by both the Python API and CLI."""

    def __init__(
        self, urdf_path, *, package_dirs=None, mesh_dir=None,
        spawn_height=0.45, effort_multiplier=1.0, collision_only=False,
    ):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError("URDF not found: %s" % self.urdf_path)
        self.package_dirs = {
            str(name): Path(path).expanduser().resolve()
            for name, path in dict(package_dirs or {}).items()
        }
        self.mesh_dir = None if mesh_dir is None else Path(mesh_dir).expanduser().resolve()
        self.spawn_height = float(spawn_height)
        self.effort_multiplier = float(effort_multiplier)
        self.collision_only = bool(collision_only)
        if self.effort_multiplier <= 0:
            raise ValueError("effort_multiplier must be positive")
        self.root = ET.parse(self.urdf_path).getroot()
        if self.root.tag != "robot":
            raise ValueError("URDF root element must be <robot>")
        self.links = {link.get("name"): link for link in self.root.findall("link")}
        self.joints = list(self.root.findall("joint"))
        if None in self.links or not self.links:
            raise ValueError("every URDF link must have a non-empty name")
        self.children = {}
        child_names = set()
        for joint in self.joints:
            self._validate_joint(joint)
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in self.links or child not in self.links:
                raise ValueError("joint references an unknown link: %s" % joint.get("name"))
            if child in child_names:
                raise ValueError("link has more than one parent: %s" % child)
            child_names.add(child)
            self.children.setdefault(parent, []).append(joint)
        roots = sorted(set(self.links) - child_names)
        if len(roots) != 1:
            raise ValueError("URDF must contain exactly one root link; found %d" % len(roots))
        self.root_link = roots[0]
        self.mesh_assets = {}
        self.materials = self._materials()

    def _prepare_mesh_assets(self, asset_base_dir):
        if asset_base_dir is None:
            destination_dir = self.urdf_path.parent / "mujoco_meshes"
        else:
            destination_dir = Path(asset_base_dir).resolve() / "mujoco_meshes"
        converted = {}
        for key, item in self.mesh_assets.items():
            suffix = item.path.suffix.lower()
            if suffix in MUJOCO_MESH_SUFFIXES:
                converted[key] = item
                continue
            if suffix != ".dae":
                raise ValueError(
                    "MuJoCo does not support mesh format %s: %s; use OBJ/STL/MSH"
                    % (suffix or "<none>", item.path)
                )
            digest = hashlib.sha1(item.path.read_bytes()).hexdigest()[:8]
            destination = destination_dir / ("%s_%s.obj" % (_safe_name(item.path.stem), digest))
            # The content hash in the filename makes this cache safe across edits.
            if not destination.is_file():
                _collada_to_obj(item.path, destination)
            converted[key] = MeshAsset(item.name, destination, item.scale)
        self.mesh_assets = converted

    def _validate_joint(self, joint):
        name = joint.get("name")
        kind = joint.get("type", "fixed")
        if not name:
            raise ValueError("every URDF joint must have a name")
        if kind not in SUPPORTED_JOINTS:
            raise ValueError("unsupported joint type %r for %s" % (kind, name))
        if joint.find("parent") is None or joint.find("child") is None:
            raise ValueError("joint %s requires parent and child links" % name)
        if joint.find("mimic") is not None:
            raise ValueError("mimic joint is not supported: %s" % name)

    def _materials(self):
        result = {}
        for material in self.root.findall("material"):
            color = material.find("color")
            if material.get("name") and color is not None and color.get("rgba"):
                result[material.get("name")] = color.get("rgba")
        return result

    def _resolve_mesh(self, filename):
        if filename.startswith("package://"):
            package_path = filename[len("package://"):]
            package, separator, relative = package_path.partition("/")
            if not separator or package not in self.package_dirs:
                raise ValueError(
                    "cannot resolve %s; pass --package-dir %s=/path/to/package"
                    % (filename, package)
                )
            path = self.package_dirs[package] / relative
        else:
            raw = Path(filename).expanduser()
            if raw.is_absolute():
                path = raw
            elif self.mesh_dir is not None:
                path = self.mesh_dir / raw.name
            else:
                path = self.urdf_path.parent / raw
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError("mesh not found: %s (from %s)" % (path, filename))
        return path

    def _mesh_asset(self, mesh):
        path = self._resolve_mesh(mesh.get("filename", ""))
        scale = _floats(mesh.get("scale", "1 1 1"), 3, "mesh scale")
        key = (str(path), scale)
        if key not in self.mesh_assets:
            digest = hashlib.sha1((str(path) + repr(scale)).encode("utf-8")).hexdigest()[:8]
            name = "%s_%s" % (_safe_name(path.stem), digest)
            self.mesh_assets[key] = MeshAsset(name, path, scale)
        return self.mesh_assets[key]

    def _geometry(self, source, target):
        geometry = source.find("geometry")
        if geometry is None:
            raise ValueError("visual/collision element has no geometry")
        box = geometry.find("box")
        cylinder = geometry.find("cylinder")
        sphere = geometry.find("sphere")
        mesh = geometry.find("mesh")
        if box is not None:
            target.set("type", "box")
            target.set("size", _fmt(value / 2 for value in _floats(box.get("size"), 3, "box size")))
        elif cylinder is not None:
            target.set("type", "cylinder")
            target.set("size", _fmt((float(cylinder.get("radius")), float(cylinder.get("length")) / 2)))
        elif sphere is not None:
            target.set("type", "sphere")
            target.set("size", _fmt((float(sphere.get("radius")),)))
        elif mesh is not None:
            asset = self._mesh_asset(mesh)
            target.set("type", "mesh")
            target.set("mesh", asset.name)
        else:
            raise ValueError("unsupported or empty URDF geometry")
        _set_pose(target, source)

    def _color(self, visual):
        material = visual.find("material")
        if material is None:
            return None
        color = material.find("color")
        if color is not None and color.get("rgba"):
            return color.get("rgba")
        return self.materials.get(material.get("name"))

    def _add_inertial(self, body, link):
        source = link.find("inertial")
        if source is None:
            return
        mass = source.find("mass")
        inertia = source.find("inertia")
        if mass is None or inertia is None:
            raise ValueError("inertial for %s requires mass and inertia" % link.get("name"))
        target = ET.SubElement(body, "inertial", {
            "mass": mass.get("value"), "pos": "0 0 0",
        })
        _set_pose(target, source)
        values = tuple(float(inertia.get(name, "0")) for name in (
            "ixx", "iyy", "izz", "ixy", "ixz", "iyz"
        ))
        target.set("fullinertia", _fmt(values))

    def _add_joint(self, body, source):
        if source is None:
            ET.SubElement(body, "freejoint")
            return
        kind = source.get("type", "fixed")
        if kind == "fixed":
            return
        attributes = {"name": source.get("name")}
        attributes["type"] = "slide" if kind == "prismatic" else "hinge"
        axis = source.find("axis")
        attributes["axis"] = "0 0 1" if axis is None else axis.get("xyz", "0 0 1")
        limit = source.find("limit")
        if kind != "continuous":
            if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                raise ValueError("joint %s requires lower/upper limits" % source.get("name"))
            attributes["range"] = "%s %s" % (limit.get("lower"), limit.get("upper"))
        dynamics = source.find("dynamics")
        if dynamics is not None:
            if dynamics.get("damping") is not None:
                attributes["damping"] = dynamics.get("damping")
            if dynamics.get("friction") is not None:
                attributes["frictionloss"] = dynamics.get("friction")
        ET.SubElement(body, "joint", attributes)

    def _add_body(self, parent, link_name, parent_joint=None):
        link = self.links[link_name]
        body = ET.SubElement(parent, "body", {"name": link_name})
        if parent_joint is None:
            body.set("pos", "0 0 %.9g" % self.spawn_height)
        else:
            _set_pose(body, parent_joint)
        self._add_joint(body, parent_joint)
        self._add_inertial(body, link)
        if not self.collision_only:
            for visual in link.findall("visual"):
                geom = ET.SubElement(body, "geom", {
                    "contype": "0", "conaffinity": "0", "group": "1",
                })
                self._geometry(visual, geom)
                color = self._color(visual)
                if color:
                    geom.set("rgba", color)
        for collision in link.findall("collision"):
            geom = ET.SubElement(body, "geom")
            self._geometry(collision, geom)
            if self.collision_only:
                geom.set("rgba", "0.35 0.45 0.65 1")
            else:
                geom.set("rgba", "0.5 0.5 0.5 0")
        for joint in self.children.get(link_name, ()):
            self._add_body(body, joint.find("child").get("link"), joint)

    def build(self, asset_base_dir=None):
        model = ET.Element("mujoco", {"model": self.root.get("name", "robot")})
        ET.SubElement(model, "compiler", {"angle": "radian"})
        ET.SubElement(model, "option", {
            "gravity": "0 0 -9.806", "timestep": "0.005", "solver": "Newton",
        })
        defaults = ET.SubElement(model, "default")
        ET.SubElement(defaults, "geom", {
            "friction": "0.6 0.005 0.0001", "condim": "3", "margin": "0.001",
        })
        ET.SubElement(defaults, "joint", {
            "damping": "0.01", "armature": "0.01", "frictionloss": "0.0",
        })
        asset = ET.SubElement(model, "asset")
        world = ET.SubElement(model, "worldbody")
        ET.SubElement(world, "light", {"directional": "true", "pos": "0 0 4", "dir": "0 0 -1"})
        ET.SubElement(world, "geom", {
            "name": "floor", "type": "plane", "size": "0 0 0.05", "rgba": "0.25 0.3 0.35 1",
        })
        self._add_body(world, self.root_link)
        self._prepare_mesh_assets(asset_base_dir)
        for item in self.mesh_assets.values():
            if asset_base_dir is None:
                mesh_file = str(item.path)
            else:
                mesh_file = Path(os.path.relpath(
                    str(item.path), str(Path(asset_base_dir).resolve())
                )).as_posix()
            attributes = {"name": item.name, "file": mesh_file}
            if item.scale != (1.0, 1.0, 1.0):
                attributes["scale"] = _fmt(item.scale)
            ET.SubElement(asset, "mesh", attributes)
        actuator = ET.SubElement(model, "actuator")
        sensors = ET.SubElement(model, "sensor")
        for joint in self.joints:
            if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
                continue
            name = joint.get("name")
            attributes = {"name": name + "_motor", "joint": name, "gear": "1"}
            limit = joint.find("limit")
            effort = 0.0 if limit is None else float(limit.get("effort", "0"))
            if effort > 0:
                effort *= self.effort_multiplier
                attributes.update({"ctrllimited": "true", "ctrlrange": _fmt((-effort, effort))})
            ET.SubElement(actuator, "motor", attributes)
            ET.SubElement(sensors, "jointpos", {"name": name + "_pos", "joint": name})
            ET.SubElement(sensors, "jointvel", {"name": name + "_vel", "joint": name})
        return model


def convert_urdf(
    urdf_path, output_path, *, package_dirs=None, mesh_dir=None,
    spawn_height=0.45, effort_multiplier=1.0, collision_only=False,
    validate=True,
):
    """Convert one URDF, write MJCF and return compiled model dimensions."""
    converter = UrdfConverter(
        urdf_path, package_dirs=package_dirs, mesh_dir=mesh_dir,
        spawn_height=spawn_height, effort_multiplier=effort_multiplier,
        collision_only=collision_only,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    root = converter.build(asset_base_dir=output.parent)
    _indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    result = {"path": str(output)}
    if validate:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(output))
        result.update({key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "nbody", "ngeom")})
    return result


def _package_mapping(value):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=/path/to/package")
    return name, path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert URDF to MuJoCo MJCF")
    parser.add_argument("urdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--package-dir", action="append", type=_package_mapping, default=[])
    parser.add_argument("--spawn-height", type=float, default=0.45)
    parser.add_argument("--effort-multiplier", type=float, default=1.0)
    parser.add_argument("--collision-only", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)
    result = convert_urdf(
        args.urdf, args.output, package_dirs=dict(args.package_dir),
        mesh_dir=args.mesh_dir, spawn_height=args.spawn_height,
        effort_multiplier=args.effort_multiplier,
        collision_only=args.collision_only, validate=not args.no_validate,
    )
    dimensions = "" if args.no_validate else " nq={nq} nv={nv} nu={nu}".format(**result)
    print("[OK] %s%s" % (result["path"], dimensions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
