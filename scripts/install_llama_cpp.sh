#!/usr/bin/env bash
# Build llama.cpp's rpc-server and llama-server into vendor/llama.cpp/build/bin
# (LocalMesh finds them there automatically). Metal is used on Apple Silicon,
# CUDA when an NVIDIA driver is present, CPU otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."

need() { command -v "$1" >/dev/null 2>&1 || { echo "✗ '$1' is missing. $2"; exit 1; }; }
need git   "Install it (macOS: xcode-select --install, Ubuntu: sudo apt install git)."
need cmake "Install it (macOS: brew install cmake, Ubuntu: sudo apt install cmake build-essential)."

mkdir -p vendor
if [ ! -d vendor/llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp vendor/llama.cpp
else
  git -C vendor/llama.cpp pull --ff-only || true
fi

FLAGS=(-DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "→ NVIDIA GPU found: building with CUDA"
  FLAGS+=(-DGGML_CUDA=ON)
fi
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build "${FLAGS[@]}"
JOBS=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)
# The RPC worker target was renamed from rpc-server to ggml-rpc-server in newer llama.cpp.
cmake --build vendor/llama.cpp/build --config Release -j "$JOBS" --target llama-server ggml-rpc-server \
  || cmake --build vendor/llama.cpp/build --config Release -j "$JOBS" --target llama-server rpc-server

echo
echo "✓ Built:"
ls -1 vendor/llama.cpp/build/bin/ | grep -E "^(ggml-rpc-server|rpc-server|llama-server)$"
echo "Restart LocalMesh on this device; the Devices page should show ✓ rpc-server ✓ llama-server."
