#!/usr/bin/env python3
"""在 dog_terrain.xml 的复杂地形中运行 00000JQG ONNX 策略。"""

from copy import deepcopy
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from play_onnx import main


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
JQG_XML = PROJECT_ROOT / "resources/robots/00000JQG/jqg.xml"
DOG_TERRAIN_XML = (
    PROJECT_ROOT / "resources/robots/dog/xml/dog_terrain.xml"
)
TERRAIN_CONFIG = SCRIPT_DIR / "config_terrain.yaml"


def build_combined_xml(output_path):
    """组合 JQG 机器人和 dog 复杂地形，不复制 dog 机器人本体。"""
    jqg_root = ET.parse(JQG_XML).getroot()
    terrain_root = ET.parse(DOG_TERRAIN_XML).getroot()

    compiler = jqg_root.find("compiler")
    compiler.set("meshdir", str(JQG_XML.parent / "meshes"))

    jqg_asset = jqg_root.find("asset")
    for element in list(jqg_asset):
        if element.tag in {"texture", "material"}:
            jqg_asset.remove(element)

    terrain_asset = terrain_root.find("asset")
    for element in terrain_asset:
        if element.tag in {"texture", "material"}:
            jqg_asset.insert(0, deepcopy(element))

    jqg_worldbody = jqg_root.find("worldbody")
    jqg_robot = jqg_worldbody.find("./body[@name='base_link']")
    if jqg_robot is None:
        raise ValueError("jqg.xml 中找不到 base_link")

    for element in list(jqg_worldbody):
        jqg_worldbody.remove(element)

    terrain_worldbody = terrain_root.find("worldbody")
    for element in terrain_worldbody:
        # dog_terrain.xml 中唯一的动态机器人根节点是 trunk。
        if element.tag == "body" and element.get("name") == "trunk":
            continue
        jqg_worldbody.append(deepcopy(element))
    jqg_worldbody.append(jqg_robot)

    ET.ElementTree(jqg_root).write(
        output_path, encoding="utf-8", xml_declaration=True
    )


def run():
    with tempfile.TemporaryDirectory(prefix="jqg_mujoco_terrain_") as temp_dir:
        combined_xml = Path(temp_dir) / "jqg_terrain.xml"
        build_combined_xml(combined_xml)
        main(
            default_config=TERRAIN_CONFIG,
            model_override=combined_xml,
        )


if __name__ == "__main__":
    run()
