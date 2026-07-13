from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl", timeout_minutes: int = 60) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current DDP configuration")

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.clone()
    if get_world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result
