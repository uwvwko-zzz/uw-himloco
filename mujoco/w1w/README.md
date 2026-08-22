# W1W MuJoCo validation

Run from the repository root in a Python environment containing MuJoCo and
ONNX Runtime:

```bash
python mujoco/w1w/play_onnx.py --onnx=logs/w1w_rough/model_2500.onnx
```

Set a constant velocity and body-height command with:

```bash
python mujoco/w1w/play_onnx.py \
  --onnx=logs/w1w_rough/model_2500.onnx \
  --vx=1.0 --vy=0.0 --wz=0.0 --height=0.45
```

Use `--duration=10` for a ten-second smoke test. The default duration is
unlimited and the simulation exits when the viewer is closed.

Keyboard controls match `mujoco/jqg/play_onnx.py`:

- `W/S` or up/down arrows: forward/backward
- `A/D` or left/right arrows: move left/right
- `Q/E`: turn left/right
- `R/F`: lower/raise target body height
- `Z`: reset target height to 0.45 m
- `T`: reset the robot
- `X`: emergency stop; releasing movement keys also stops the command

## Browser GUI / runtime-control plugin

The `mujoco/mujoco-gui` plugin is integrated through a separate W1W entry
point, so the original players remain unchanged. Start the browser-only GUI
with:

```bash
python mujoco/w1w/play_gui.py \
  --onnx=logs/w1w_rough/terrain_6/model_4500.onnx \
  --gui
```

Use `--map=stairs` (or select a map in the browser) for terrain tests. Without
`--gui`, the same script opens the native MuJoCo viewer. For an automated smoke
test use `--headless --duration=1`.

The panel has a separate wheel-velocity Kp slider, reproducible randomization,
save/load parameter presets, live torque/contact/slip telemetry, and browser
buttons/shortcuts for `M` prone, `N` side and `B` random fall tests. Closing the
panel also stops the browser-only simulation and releases its port.

The flat map is extracted directly from the selected robot XML, so its skybox,
floor texture, material and lighting match the original native MuJoCo scene.
Browser rendering uses a 960x720 framebuffer and JPEG quality 92; the plugin
automatically expands the MuJoCo offscreen framebuffer to the requested size.

W1W-specific joint lookup, 348-value observation history, reset behavior,
leg position PD, wheel velocity control and telemetry live in
`w1w_adapter.py`. It implements the plugin's public `RobotAdapter` contract;
`play_gui.py` only coordinates policy inference, keyboard/fall tests and the
simulation loop. This file is the 16-DOF mixed-control reference for porting a
new wheeled quadruped.

Inspect the W1W collision bodies, centers of mass, inertia boxes and joint/body
axes without running the policy:

```bash
python -m runtime_control.preview mujoco/w1w/w1w.xml
```

This opens a RobotsFan-style local browser inspector with a body tree, physical
properties, joint sliders and MuJoCo-rendered visual/collision layers. Add
`--native` for MuJoCo's GLFW viewer. Use `--collision-only` to hide mesh-only visuals, or use
`--no-viewer --json=/tmp/w1w_report.json` to export all body inertia matrices,
local/world centers of mass, joint axes and collision volumes.

## Complex-terrain sim-to-sim

```bash
python mujoco/w1w/play_onnx_terrain.py \
  --onnx=logs/w1w_rough/model_2500.onnx
```

The terrain contains stairs, ramps, narrow bridges, poles, low-clearance bars
and uneven obstacles. Additional terrain controls are `O` (front camera), `I`
(rear/top camera), `P` (tracking camera), and Space (random horizontal push).
