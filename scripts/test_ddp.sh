#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${1:-configs/default.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

NUM_GPUS="${NUM_GPUS:-3}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  "${PROJECT_ROOT}/test.py" \
  --config "${CONFIG_PATH}" \
  "$@"
