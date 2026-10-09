#!/bin/bash
# Build container/fb9.sif. Run from inside a SLURM session (works with 2 CPUs / 4 GB).
# Cache and temp stay inside the repo (.apptainer/, gitignored).
# mksquashfs is memory-capped (it gets OOM-killed at 4 GB otherwise) and skips NFS ACL xattrs.
set -euo pipefail
cd "$(dirname "$0")/.."
export APPTAINER_CACHEDIR=$PWD/.apptainer/cache APPTAINER_TMPDIR=$PWD/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build --fakeroot --force \
    --mksquashfs-args "-mem 1G -processors 2 -no-xattrs" \
    container/fb9.sif container/fb9.def
