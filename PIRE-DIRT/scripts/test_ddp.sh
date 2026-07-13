#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"
NUM_GPUS="${NUM_GPUS:-3}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  test.py --config "${CONFIG_PATH}"
