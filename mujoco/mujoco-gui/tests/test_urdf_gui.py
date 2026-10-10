import json
from pathlib import Path
import tempfile
import unittest
import unittest.mock
from urllib.request import Request, urlopen

from runtime_control.urdf_gui import PAGE, UrdfGuiServer


REPOSITORY = Path(__file__).resolve().parents[1]


class UrdfGuiTests(unittest.TestCase):
    def test_page_contains_conversion_controls(self):
        self.assertIn("URDF → MuJoCo MJCF", PAGE)
        self.assertIn("转换后自动打开 MuJoCo", PAGE)
        self.assertIn("选择 URDF…", PAGE)
        self.assertIn("打开转换结果文件夹", PAGE)
        self.assertIn("/api/convert", PAGE)
        self.assertIn("退出转换器", PAGE)

    def test_http_conversion_without_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dog.xml"
            server = UrdfGuiServer(port=0).start()
            try:
                payload = {
                    "urdf": str(REPOSITORY / "eg/dog/urdf/dog.urdf"),
                    "output": str(output),
                    "preview": False,
                }
                request = Request(
                    server.url + "api/convert",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                result = json.loads(urlopen(request, timeout=5).read())
                self.assertEqual((result["nq"], result["nv"], result["nu"]), (19, 18, 12))
                self.assertFalse(result["preview"])
                self.assertTrue(output.is_file())
            finally:
                server.close()

    def test_picker_suggestions_use_default_output_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            server = UrdfGuiServer(port=0, output_dir=directory)
            try:
                with unittest.mock.patch(
                    "runtime_control.urdf_gui._dialog",
                    return_value=str(REPOSITORY / "eg/dog/urdf/dog.urdf"),
                ):
                    result = server.pick_urdf()
                self.assertTrue(result["suggested_output"].endswith("/dog.xml"))
                self.assertEqual(
                    Path(result["suggested_output"]),
                    REPOSITORY / "eg/dog/urdf/dog.xml",
                )
                self.assertTrue(result["suggested_mesh_dir"].endswith("/dog/meshes"))
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
