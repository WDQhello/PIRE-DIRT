#!/usr/bin/env python
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

from torch.nn.parallel import DistributedDataParallel as DDP

from aigi_detector.engine import run_training
from aigi_detector.models import DINOv3LoRAMILFPSM
from aigi_detector.utils.config import apply_overrides, build_common_parser, load_config
from aigi_detector.utils.distributed import cleanup_distributed, init_distributed, is_main_process
from aigi_detector.utils.seed import set_seed


def build_model(config: dict) -> DINOv3LoRAMILFPSM:
    model_cfg = config["model"]
    return DINOv3LoRAMILFPSM(
        backbone_name=model_cfg["backbone_name"],
        backbone_checkpoint=config["paths"]["backbone_checkpoint"],
        num_classes=model_cfg["num_classes"],
        lora_rank=model_cfg["lora_rank"],
        lora_alpha=model_cfg["lora_alpha"],
        mil_hidden_dim=model_cfg["mil_hidden_dim"],
        k_ratio=model_cfg["k_ratio"],
        use_fpsm=model_cfg["use_fpsm"],
        fpsm_strength=model_cfg["fpsm_strength"],
    )


def main() -> None:
    parser = build_common_parser("Distributed training for DINOv3-LoRA-MIL-FPSM")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)

    set_seed(config["seed"])
    device = init_distributed(
        backend=config["distributed"]["backend"],
        timeout_minutes=config["distributed"]["timeout_minutes"],
    )

    try:
        model = build_model(config).to(device)
        if is_main_process():
            print(f"[Device] {device}: {device.index}")
            print(f"[Backbone] {model.backbone_checkpoint}")
            print(f"[LoRA] inserted into {model.lora_layer_count} QKV layers")

        local_rank = device.index
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config["train"]["find_unused_parameters"],
        )
        run_training(model, device, config)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
