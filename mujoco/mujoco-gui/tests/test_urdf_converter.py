from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco

from runtime_control import UrdfConverter, convert_urdf


REPOSITORY = Path(__file__).resolve().parents[1]
DOG_URDF = REPOSITORY / "eg" / "dog" / "urdf" / "dog.urdf"


class UrdfConverterTests(unittest.TestCase):
    def test_dog_converts_and_compiles(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dog.xml"
            result = convert_urdf(DOG_URDF, output)
            self.assertEqual((result["nq"], result["nv"], result["nu"]), (19, 18, 12))
            self.assertGreater(len(output.read_text(encoding="utf-8").splitlines()), 100)
            model = mujoco.MjModel.from_xml_path(str(output))
            self.assertEqual(model.nbody, 19)
            root = ET.parse(output).getroot()
            self.assertIsNotNone(root.find(".//inertial[@fullinertia]"))
            mesh_files = [mesh.get("file") for mesh in root.findall("./asset/mesh")]
            self.assertTrue(all(not Path(path).is_absolute() for path in mesh_files))
            self.assertTrue(all((output.parent / path).resolve().is_file()
                                for path in mesh_files))

    def test_mesh_scale_and_collision_mesh_are_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mesh = directory / "part.obj"
            # Minimal tetrahedron accepted by MuJoCo's mesh loader.
            mesh.write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n",
                encoding="ascii",
            )
            urdf = directory / "mesh.urdf"
            urdf.write_text(
                '<robot name="mesh"><link name="base">'
                '<inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
                '<collision><geometry><mesh filename="part.obj" scale="2 3 4"/></geometry></collision>'
                '</link></robot>', encoding="utf-8",
            )
            output = directory / "mesh.xml"
            convert_urdf(urdf, output, collision_only=True)
            asset = ET.parse(output).getroot().find("./asset/mesh")
            self.assertEqual(asset.get("scale"), "2 3 4")
            self.assertEqual(asset.get("file"), "part.obj")

    def test_rejects_unsupported_joint_instead_of_dropping_it(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf = Path(directory) / "bad.urdf"
            urdf.write_text(
                '<robot name="bad"><link name="a"/><link name="b"/>'
                '<joint name="floating" type="floating"><parent link="a"/><child link="b"/></joint>'
                '</robot>', encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported joint type"):
                UrdfConverter(urdf)

    def test_collada_mesh_is_converted_to_relative_obj_and_compiles(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mesh = directory / "part.dae"
            mesh.write_text(
                '''<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="part"><mesh>
    <source id="positions"><float_array id="positions-array" count="12">0 0 0 1 0 0 0 1 0 0 0 1</float_array><technique_common><accessor source="#positions-array" count="4" stride="3"/></technique_common></source>
    <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
    <triangles count="4"><input semantic="VERTEX" source="#vertices" offset="0"/><p>0 2 1 0 1 3 0 3 2 1 2 3</p></triangles>
  </mesh></geometry></library_geometries>
</COLLADA>''', encoding="utf-8")
            urdf = directory / "mesh.urdf"
            urdf.write_text(
                '<robot name="mesh"><link name="base">'
                '<inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
                '<visual><geometry><mesh filename="part.dae"/></geometry></visual>'
                '</link></robot>', encoding="utf-8")
            output = directory / "mesh.xml"
            result = convert_urdf(urdf, output)
            asset_path = ET.parse(output).getroot().find("./asset/mesh").get("file")
            self.assertEqual(Path(asset_path).suffix, ".obj")
            self.assertFalse(Path(asset_path).is_absolute())
            self.assertTrue((output.parent / asset_path).is_file())
            self.assertEqual(result["nq"], 7)


if __name__ == "__main__":
    unittest.main()
