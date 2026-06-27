#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${SPAR3D_REPO_PATH:-$ROOT_DIR/third_party/stable-point-aware-3d}"
VENV_DIR="${SPAR3D_VENV_DIR:-$ROOT_DIR/.venv-spar3d}"
PYTHON_BIN="${SPAR3D_PYTHON_BIN:-$VENV_DIR/bin/python3}"
RUNTIME_DIR="${SPAR3D_RUNTIME_DIR:-$ROOT_DIR/.runtime/spar3d}"
LOG_DIR="$RUNTIME_DIR/logs"
PID_FILE="$RUNTIME_DIR/service.pid"
LOG_FILE="$LOG_DIR/service.log"
HOST="${SPAR3D_HOST:-127.0.0.1}"
PORT="${SPAR3D_PORT:-8091}"
DEVICE="${SPAR3D_DEVICE:-cpu}"
MODEL_ID="${SPAR3D_MODEL_ID:-stabilityai/stable-point-aware-3d}"
LOW_VRAM="${SPAR3D_LOW_VRAM_MODE:-1}"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$REPO_DIR")"

echo "==> ROOT_DIR: $ROOT_DIR"
echo "==> REPO_DIR: $REPO_DIR"
echo "==> DEVICE: $DEVICE"
echo "==> MODEL_ID: $MODEL_ID"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "==> Cloning Stability-AI/stable-point-aware-3d"
  git clone --depth 1 https://github.com/Stability-AI/stable-point-aware-3d.git "$REPO_DIR"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "==> Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "==> Upgrading pip/setuptools/wheel"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install "setuptools<81"

echo "==> Installing PyTorch for local $DEVICE mode"
if [[ "$DEVICE" == "cpu" ]]; then
  "$PYTHON_BIN" -m pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
else
  "$PYTHON_BIN" -m pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
fi

echo "==> Installing SPAR3D requirements"
(
  cd "$REPO_DIR"
  echo "==> Installing CLIP and AlphaCLIP without build isolation"
  "$PYTHON_BIN" -m pip install --no-build-isolation "git+https://github.com/openai/CLIP.git"
  "$PYTHON_BIN" -m pip install --no-build-isolation "git+https://github.com/SunzeY/AlphaCLIP.git"

  echo "==> Installing remaining SPAR3D requirements"
  grep -v "github.com/openai/CLIP.git" requirements.txt | grep -v "github.com/SunzeY/AlphaCLIP.git" > /tmp/spar3d_requirements_filtered.txt
  "$PYTHON_BIN" -m pip install --no-build-isolation -r /tmp/spar3d_requirements_filtered.txt
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "==> Logging into Hugging Face with HF_TOKEN"
  "$PYTHON_BIN" -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN"
else
  cat <<'EOF'
==> HF_TOKEN not set.
SPAR3D weights are gated on Hugging Face. If the first run fails with permission errors, do:
    export HF_TOKEN=hf_xxx
and rerun this script.
EOF
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "==> Stopping existing SPAR3D service pid=$OLD_PID"
    kill "$OLD_PID" || true
    sleep 1
  fi
fi

SERVICE_ARGS=(
  "$ROOT_DIR/scripts/spar3d_local_service.py"
  "--host" "$HOST"
  "--port" "$PORT"
  "--repo-path" "$REPO_DIR"
  "--python-bin" "$PYTHON_BIN"
  "--device" "$DEVICE"
  "--pretrained-model" "$MODEL_ID"
  "--runtime-dir" "$RUNTIME_DIR"
)

if [[ "$LOW_VRAM" == "1" ]]; then
  SERVICE_ARGS+=("--low-vram-mode")
fi

echo "==> Starting SPAR3D local service on http://$HOST:$PORT"
nohup "$PYTHON_BIN" "${SERVICE_ARGS[@]}" >"$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "==> Service pid: $NEW_PID"
echo "==> Log file: $LOG_FILE"
echo "==> Health check: curl http://$HOST:$PORT/health"
