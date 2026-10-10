#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MUJOCO_GUI_PYTHON:-${HIMLOCO_EDITOR_PYTHON:-python}}"
HOST="127.0.0.1"
PORT="8770"

arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
    case "${arguments[index]}" in
        --host=*) HOST="${arguments[index]#*=}" ;;
        --port=*) PORT="${arguments[index]#*=}" ;;
        --host|--port)
            if ((index + 1 >= ${#arguments[@]})); then
                echo "错误：${arguments[index]} 后面缺少参数。" >&2; exit 2
            fi
            [[ "${arguments[index]}" == "--host" ]] && HOST="${arguments[index+1]}" || PORT="${arguments[index+1]}"
            ((index+=1))
            ;;
    esac
done

PYTHON_PATH="$(command -v -- "${PYTHON_BIN}" 2>/dev/null || true)"
if [[ -z "${PYTHON_PATH}" || ! -x "${PYTHON_PATH}" ]]; then
    echo "错误：找不到 Python：${PYTHON_BIN}" >&2; exit 1
fi
if ! "${PYTHON_PATH}" -c 'import mujoco, mujoco.viewer' 2>/dev/null; then
    echo "错误：${PYTHON_PATH} 缺少 MuJoCo Python 包。" >&2; exit 1
fi
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
    echo "错误：端口必须是 1 到 65535：${PORT}" >&2; exit 2
fi
CONNECT_HOST="${HOST}"
[[ "${CONNECT_HOST}" == "0.0.0.0" || "${CONNECT_HOST}" == "::" ]] && CONNECT_HOST="127.0.0.1"
if "${PYTHON_PATH}" -c 'import socket,sys;s=socket.socket();s.settimeout(.2);sys.exit(0 if s.connect_ex((sys.argv[1],int(sys.argv[2])))==0 else 1)' "${CONNECT_HOST}" "${PORT}"; then
    url="http://${CONNECT_HOST}:${PORT}/"
    if command -v curl >/dev/null 2>&1 && curl --max-time 1 --silent "${url}" | grep -q 'URDF → MuJoCo MJCF'; then
        echo "URDF 转换器已经在运行：${url}"; exit 0
    fi
    echo "错误：${HOST}:${PORT} 已被其他程序占用。" >&2; exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${SCRIPT_DIR}"
exec "${PYTHON_PATH}" -m runtime_control.urdf_gui --port 8770 --python "${PYTHON_PATH}" "$@"
