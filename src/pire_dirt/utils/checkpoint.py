from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# Compatibility aliases for checkpoints produced before the public module
# names were standardized as PIRE and DIRT.
_LEGACY_PREFIX_MAP = {
    "fpsm.": "pire.",
    "mil_head.": "dirt.",
}


def normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
    remap_legacy_names: bool = True,
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]

        if remap_legacy_names:
            for old_prefix, new_prefix in _LEGACY_PREFIX_MAP.items():
                if key.startswith(old_prefix):
                    key = new_prefix + key[len(old_prefix) :]
                    break

        normalized[key] = value

    return normalized


def extract_state_dict(
    checkpoint: Any,
    remap_legacy_names: bool = True,
) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a state_dict or a dictionary containing one"
        )

    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    return normalize_state_dict(
        state_dict,
        remap_legacy_names=remap_legacy_names,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
    remap_legacy_names: bool = True,
) -> dict[str, Any] | None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(
        checkpoint,
        remap_legacy_names=remap_legacy_names,
    )
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else None


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )
