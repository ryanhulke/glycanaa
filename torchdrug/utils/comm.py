import contextlib
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "get_rank",
    "get_world_size",
    "synchronize",
    "init_process_group",
    "reduce",
]


def init_process_group(*args, **kwargs) -> None:
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(*args, **kwargs)


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def synchronize() -> None:
    if get_world_size() == 1:
        return
    dist.barrier()


def reduce(value: torch.Tensor, op: str = "mean") -> torch.Tensor:
    if get_world_size() == 1:
        return value
    value = value.clone()
    if op == "mean":
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= get_world_size()
    elif op == "sum":
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    else:
        raise ValueError(f"Unsupported reduce operation `{op}`")
    return value
