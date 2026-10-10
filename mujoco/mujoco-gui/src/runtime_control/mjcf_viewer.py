"""Open an MJCF model in MuJoCo's native passive viewer."""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def view_mjcf(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("MJCF not found: %s" % path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = max(1.2, float(model.stat.extent) * 1.8)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        while viewer.is_running():
            started = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Open an MJCF in MuJoCo viewer")
    parser.add_argument("xml", type=Path)
    args = parser.parse_args(argv)
    view_mjcf(args.xml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
