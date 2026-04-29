#!/usr/bin/env bash
# Preflight checks for vid3d studio local dev. Exits 0 if the host is
# ready to `make up`, non-zero on any required component missing.
#
# By default skips the heavy "docker can see the GPU" probe (which pulls
# a ~150 MB CUDA image on first run). Set `DOCTOR_GPU_PROBE=1` to enable
# it once you've installed nvidia-container-toolkit and want to confirm
# the wiring end-to-end.

set -uo pipefail

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

fail=0

check() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        green "  ok  $label"
    else
        red "  err $label"
        fail=$((fail + 1))
    fi
}

check_optional() {
    local label="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        green "  ok  $label"
    else
        yellow "  --  $label (optional)"
    fi
}

echo "host:"
check "docker installed"             command -v docker
check "docker daemon reachable"      docker info
check "docker compose v2 available"  docker compose version

echo
echo "gpu:"
check "nvidia-smi installed"         command -v nvidia-smi
check "gpu visible to host"          nvidia-smi
# Look for the nvidia runtime in `docker info`. The runtime registers
# itself there once nvidia-container-toolkit is installed and the
# daemon's been restarted.
check "docker has nvidia runtime"    bash -c "docker info 2>/dev/null | grep -Eq 'Runtimes:.*nvidia|nvidia[[:space:]]*runc'"

if [ "${DOCTOR_GPU_PROBE:-0}" = "1" ]; then
    echo
    echo "container gpu access (probing — pulls ~150 MB on first run):"
    if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        green "  ok  containerized nvidia-smi sees the gpu"
    else
        red "  err containerized nvidia-smi failed — check nvidia-container-toolkit"
        fail=$((fail + 1))
    fi
else
    echo
    yellow "  --  containerized GPU probe skipped (set DOCTOR_GPU_PROBE=1 to run)"
fi

echo
if [ $fail -gt 0 ]; then
    red "$fail check(s) failed. Fix the above before running 'make up'."
    exit 1
fi
green "all required checks passed. ready to run 'make up'."
