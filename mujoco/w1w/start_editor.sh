#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

exec "${PROJECT_ROOT}/mujoco/mujoco-gui/start_editor.sh" \
    --preview-player "${SCRIPT_DIR}/play_gui.py" \
    --robot-xml "${SCRIPT_DIR}/w1w.xml" \
    --policy "${PROJECT_ROOT}/logs/w1w_rough/terrain_7/model_4500.onnx" \
    "$@"
