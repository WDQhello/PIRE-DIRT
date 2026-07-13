from __future__ import annotations

import torch
import torch.nn as nn


class FPSM(nn.Module):
    """Feature perturbation using style, frequency, and random-noise paths."""

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
        style = self.style_fc(patch_tokens)

        spectrum = torch.fft.rfft(patch_tokens, dim=-1)
        mask = self.freq_mask[: spectrum.shape[-1]].view(1, 1, -1)
        spectrum = spectrum * (1 + mask * self.strength)
        frequency = torch.fft.irfft(spectrum, n=patch_tokens.shape[-1], dim=-1)

        noise = self.noise_proj(torch.randn_like(patch_tokens))
        return patch_tokens + self.strength * (style + frequency + noise)
