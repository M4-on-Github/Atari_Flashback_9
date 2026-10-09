#!/bin/bash
# Run a command inside container/fb9.sif with GPU access, from the repo root.
# --no-mount /opt: the cluster's apptainer.conf binds host /opt, which would hide the image's /opt/conda (Python).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
exec apptainer exec --nv --no-mount /opt "$REPO/container/fb9.sif" "$@"
