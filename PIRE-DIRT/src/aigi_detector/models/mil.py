from __future__ import annotations

import torch
import torch.nn as nn


class TopKMIL(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int = 2,
        hidden_dim: int = 256,
        k_ratio: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0 < k_ratio <= 1:
            raise ValueError("k_ratio must be in the interval (0, 1]")

        self.k_ratio = k_ratio
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.fc1(patch_tokens))
        patch_logits = self.fc2(hidden)
        k = max(1, int(patch_tokens.size(1) * self.k_ratio))
        topk_logits = patch_logits.topk(k, dim=1).values
        return topk_logits.mean(dim=1)
