#!/usr/bin/env bash
# Sync this repo to the GPU pod and run a command there.
#
#   scripts/gpu.sh                         # sync, then open a shell on the pod
#   scripts/gpu.sh nvidia-smi
#   scripts/gpu.sh 'make -C SGEMM && SGEMM/build/sgemm 1'
#
# Override the target with GPU_HOST (an ~/.ssh/config alias) and GPU_DIR.
# Needs SSH over exposed TCP (Host runpod). The ssh.runpod.io proxy cannot
# run remote commands or rsync.
set -euo pipefail

HOST="${GPU_HOST:-runpod}"
REMOTE_DIR="${GPU_DIR:-/workspace/kernels}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Non-interactive SSH skips the image's login PATH; nvcc lives here.
REMOTE_ENV='export PATH="/usr/local/cuda/bin:$PATH"; export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"'

# Fresh RunPod images often lack rsync; install it on first use.
ssh "$HOST" "$REMOTE_ENV; command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)"
rsync -az --delete --exclude .git --exclude build/ --exclude .DS_Store \
    "$ROOT/" "$HOST:$REMOTE_DIR/"

# Always leave an interactive shell so the TCP session stays up.
# Run the user's command first when given; a failure still drops you in.
if [ $# -eq 0 ]; then
    ssh -t "$HOST" "$REMOTE_ENV; cd $REMOTE_DIR && exec bash"
else
    ssh -t "$HOST" "$REMOTE_ENV; cd $REMOTE_DIR && { $*; }; exec bash"
fi
