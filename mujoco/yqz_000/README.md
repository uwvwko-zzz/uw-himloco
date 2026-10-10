# YQZ-000 MuJoCo sim2sim

## Plugin player

Start the browser GUI implemented by `mujoco-gui/runtime_control`:

```bash
python mujoco/yqz_000/play_onnx.py \
  --onnx logs/yqz_rough/model_3200.onnx \
  --map flat --vx 1.0 --height 0.47 --gui
```

The panel is available at `http://127.0.0.1:8765` and can switch among all
12 bundled maps, tune motor and randomization parameters, change cameras,
reset the robot and apply external pushes. Hold `W/S` for forward/backward,
`A/D` for lateral motion and `Q/E` for yaw. Their default magnitudes are
1.0 m/s, 0.6 m/s and 1.0 rad/s and can be changed with `--key-vx`,
`--key-vy` and `--key-wz`.

Run with MuJoCo's native viewer by omitting `--gui`:

```bash
python mujoco/yqz_000/play_onnx.py \
  --onnx logs/yqz_rough/model_3200.onnx --map stairs --vx 1.0
```

Plugin-path smoke test without a viewer:

```bash
python mujoco/yqz_000/play_onnx.py \
  --onnx logs/yqz_rough/model_3200.onnx \
  --map flat --vx 1.0 --headless --duration 2
```

Generate MuJoCo-safe visual STL files and regenerate the visual MJCF after
changing the URDF or source meshes:

```bash
python mujoco/yqz_000/prepare_visual_meshes.py
python mujoco/scripts/urdf_to_xml.py \
  resources/robots/YQZ-000/urdf/YQZ-000.urdf \
  resources/robots/YQZ-000/xml/YQZ-000.xml \
  --spawn 0.50 --meshdir ../meshes_mujoco/
```

The source base and thigh STL files exceed MuJoCo's 200,000-face-per-STL
limit. The preparation command leaves those originals untouched and creates
display-only copies capped at 120,000 faces. Physics still uses the URDF's
primitive collision geoms rather than the visual mesh.

The adapter matches the Isaac Gym policy contract: 348 observation values,
16 actions, 80/2 leg PD gains, 30 rad/s wheel action scaling, 0.5 wheel
velocity gain, and the YQZ-specific mirrored joint order and default pose.
