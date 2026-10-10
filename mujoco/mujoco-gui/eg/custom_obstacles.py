"""Example trusted obstacle plugin: a cylindrical pedestal."""

import xml.etree.ElementTree as ET

from runtime_control import register_obstacle


def build_pedestal(worldbody, obstacle, index):
    radius = float(obstacle.params["radius"])
    height = float(obstacle.params["height"])
    if radius <= 0 or height <= 0:
        raise ValueError("radius and height must be positive")
    ET.SubElement(worldbody, "geom", {
        "name": obstacle.name or "pedestal_%d" % index,
        "type": "cylinder",
        "pos": "%s %s %s" % (obstacle.x, obstacle.y, obstacle.z + height / 2),
        "size": "%s %s" % (radius, height / 2),
        "rgba": "0.3 0.7 0.5 1",
    })


register_obstacle(
    "pedestal", "圆柱台", build_pedestal,
    defaults={"radius": .4, "height": .3},
    parameters={
        "radius": ["半径", "m", .05, 2, .05],
        "height": ["高度", "m", .05, 2, .05],
    },
)
