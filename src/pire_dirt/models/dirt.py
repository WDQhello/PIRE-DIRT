from __future__ import annotations

import torch
import torch.nn as nn


class DIRT(nn.Module):
    """Dynamic Important Region Token filtering head.

    DIRT predicts patch-level logits, retains the most suspicious token regions,
    and averages their logits to obtain the regional prediction.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int = 2,
        hidden_dim: int = 256,
        retain_ratio: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0 < retain_ratio <= 1:
            raise ValueError("retain_ratio must be in the interval (0, 1]")

        self.retain_ratio = retain_ratio
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.fc1(patch_tokens))
        patch_logits = self.fc2(hidden)

        num_tokens = patch_tokens.size(1)
        num_retained = max(1, int(num_tokens * self.retain_ratio))
        important_logits = patch_logits.topk(num_retained, dim=1).values
        return important_logits.mean(dim=1)
