from __future__ import annotations

import torch
import torch.nn as nn


class PIRE(nn.Module):
    """Prior Information Region Enhancement.

    PIRE enhances patch-level representations through complementary style,
    frequency, and noise-statistical paths. It is enabled only during training
    by the detector, matching the behavior of the original implementation.
    """

    def __init__(self, embed_dim: int, strength: float = 0.20) -> None:
        super().__init__()
        self.strength = strength

        self.style_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.freq_mask = nn.Parameter(torch.randn(embed_dim) * 0.01)
        self.noise_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        style_features = self.style_fc(patch_tokens)

        spectrum = torch.fft.rfft(patch_tokens, dim=-1)
        frequency_mask = self.freq_mask[: spectrum.shape[-1]].view(1, 1, -1)
        spectrum = spectrum * (1 + frequency_mask * self.strength)
        frequency_features = torch.fft.irfft(
            spectrum,
            n=patch_tokens.shape[-1],
            dim=-1,
        )

        noise_features = self.noise_proj(torch.randn_like(patch_tokens))
        enhanced_features = style_features + frequency_features + noise_features
        return patch_tokens + self.strength * enhanced_features
