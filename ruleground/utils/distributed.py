"""
Distributed Training Utilities

Thin wrappers around PyTorch DDP and Accelerate for multi-GPU training.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """Check if we are in a distributed training context."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> None:
    """Initialize distributed training if environment variables are set."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        logger.info(f"Distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        logger.info("Running in single-process mode")


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all processes."""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all ranks into a single tensor on rank 0."""
    if not is_distributed():
        return tensor

    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)
