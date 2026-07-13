from __future__ import annotations

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    """Low-rank residual updates for query, key, and value projections."""

    def __init__(self, qkv_linear: nn.Linear, dim: int, rank: int = 2, alpha: int = 2) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")

        self.qkv = qkv_linear
        self.scaling = alpha / rank

        self.q_A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.q_B = nn.Parameter(torch.zeros(dim, rank))
        self.k_A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.k_B = nn.Parameter(torch.zeros(dim, rank))
        self.v_A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.v_B = nn.Parameter(torch.zeros(dim, rank))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(inputs)
        query, key, value = qkv.chunk(3, dim=-1)

        delta_q = (inputs @ self.q_A.T @ self.q_B.T) * self.scaling
        delta_k = (inputs @ self.k_A.T @ self.k_B.T) * self.scaling
        delta_v = (inputs @ self.v_A.T @ self.v_B.T) * self.scaling
        return torch.cat([query + delta_q, key + delta_k, value + delta_v], dim=-1)


def add_lora_to_dinov3(model: nn.Module, rank: int = 2, alpha: int = 2) -> int:
    """Replace every ``attn.qkv`` linear layer and return the replacement count."""
    target_names = [name for name, _ in model.named_modules() if name.endswith("attn.qkv")]

    for name in target_names:
        parent = model
        components = name.split(".")
        for component in components[:-1]:
            parent = getattr(parent, component)

        original = getattr(parent, components[-1])
        if not isinstance(original, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {name}, found {type(original).__name__}")
        dim = original.weight.shape[1]
        setattr(parent, components[-1], LoRAQKV(original, dim, rank=rank, alpha=alpha))

    if not target_names:
        raise RuntimeError("No attention QKV layers ending with 'attn.qkv' were found")
    return len(target_names)
