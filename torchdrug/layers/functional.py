from typing import Tuple

import torch


def variadic_to_padded(tensor: torch.Tensor, sizes: torch.Tensor, value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(sizes.shape[0])
    max_len = int(sizes.max().item()) if batch_size > 0 else 0
    padded = tensor.new_full((batch_size, max_len) + tensor.shape[1:], value)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=tensor.device)
    start = 0
    for i, size in enumerate(sizes.tolist()):
        if size == 0:
            continue
        end = start + size
        padded[i, :size] = tensor[start:end]
        mask[i, :size] = True
        start = end
    return padded, mask


def padded_to_variadic(tensor: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    chunks = []
    for i, size in enumerate(sizes.tolist()):
        chunks.append(tensor[i, :size])
    return torch.cat(chunks, dim=0) if chunks else tensor.new_empty((0,) + tensor.shape[2:])


def masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int = 0) -> torch.Tensor:
    mask = mask.float()
    return (value * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-9)


def as_mask(indices: torch.Tensor, size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=indices.device)
    mask[indices] = True
    return mask
