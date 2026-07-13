from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from safetensors.torch import load_file

from .fpsm import FPSM
from .lora import add_lora_to_dinov3
from .mil import TopKMIL


class DINOv3LoRAMILFPSM(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        backbone_checkpoint: str | Path,
        num_classes: int = 2,
        lora_rank: int = 2,
        lora_alpha: int = 2,
        mil_hidden_dim: int = 256,
        k_ratio: float = 0.10,
        use_fpsm: bool = True,
        fpsm_strength: float = 0.20,
    ) -> None:
        super().__init__()
        self.use_fpsm = use_fpsm

        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        checkpoint_path = Path(backbone_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Backbone checkpoint does not exist: {checkpoint_path}")

        backbone_state = load_file(str(checkpoint_path), device="cpu")
        self.backbone.load_state_dict(backbone_state, strict=True)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        lora_count = add_lora_to_dinov3(
            self.backbone,
            rank=lora_rank,
            alpha=lora_alpha,
        )

        embed_dim = int(getattr(self.backbone, "num_features", 1024))
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.mil_head = TopKMIL(
            dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=mil_hidden_dim,
            k_ratio=k_ratio,
        )
        self.fpsm = FPSM(embed_dim=embed_dim, strength=fpsm_strength)

        self.backbone_checkpoint = str(checkpoint_path)
        self.lora_layer_count = lora_count

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(images)
        if isinstance(tokens, dict):
            if "x_norm_clstoken" in tokens and "x_norm_patchtokens" in tokens:
                cls_token = tokens["x_norm_clstoken"]
                patch_tokens = tokens["x_norm_patchtokens"]
            else:
                raise KeyError("Unsupported dictionary output from backbone.forward_features")
        else:
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]

        if self.training and self.use_fpsm:
            patch_tokens = self.fpsm(patch_tokens)

        global_logits = self.cls_head(cls_token)
        regional_logits = self.mil_head(patch_tokens)
        return (global_logits + regional_logits) / 2
