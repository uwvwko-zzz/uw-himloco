# YQZ high-platform MuJoCo sim2sim

This player evaluates the `yqz_high` ONNX policy on a compact sequence of
vertical-faced platforms ordered from 80 cm down to 10 cm. Each platform is
1.2 x 2.0 m and carries a yellow height label on its top surface. The first
ledge is 0.8 m in front of the spawn, matching the interactive PyTorch demo.
`--platform-height` optionally starts at a later course segment.

```bash
python mujoco/yqz_high/play_onnx.py \
  --onnx logs/yqz_high/<run>/model.onnx \
  --platform-height 0.4 --vx 0.8
```

Available starting heights are `0.8`, `0.7`, `0.6`, `0.5`, `0.4`, `0.3`,
`0.2`, and `0.1` m. The default is `0.8` m.

Native-viewer controls are `↑/↓` or `W/S` for forward/backward,
`←/→` or `A/D` for strafing, and `Q/E` for yaw. `X` sends an exact zero
command, `T` resets, and `Space` applies a push. The
movement command is active only while a key is held and returns to exact zero
on release. The browser GUI accepts the same arrow keys.

Browser GUI:

```bash
python mujoco/yqz_high/play_onnx.py \
  --onnx logs/yqz_high/<run>/model.onnx \
  --platform-height 0.4 --gui
```

Headless smoke test:

```bash
python mujoco/yqz_high/play_onnx.py \
  --onnx logs/yqz_high/<run>/model.onnx \
  --platform-height 0.1 --headless --duration 2
```

The adapter matches the training task's 348-value observation history and 16
actions. It also uses the high-platform task's `0.35` leg action scale, 30
rad/s wheel scale, and zero normalized fixed-height command.
