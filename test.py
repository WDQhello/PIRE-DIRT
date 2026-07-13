#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow direct execution from a cloned repository without requiring
# `pip install -e .` first.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

from torch.nn.parallel import DistributedDataParallel as DDP

from pire_dirt.engine.evaluator import evaluate_all_domains, save_test_results
from pire_dirt.models import DINOv3LoRAPIREDIRT
from pire_dirt.utils.checkpoint import load_checkpoint
from pire_dirt.utils.config import apply_overrides, build_common_parser, load_config
from pire_dirt.utils.distributed import (
    cleanup_distributed,
    init_distributed,
    is_main_process,
)
from pire_dirt.utils.seed import set_seed


def build_model(config: dict) -> DINOv3LoRAPIREDIRT:
    model_cfg = config["model"]
    return DINOv3LoRAPIREDIRT(
        backbone_name=model_cfg["backbone_name"],
        backbone_checkpoint=config["paths"]["backbone_checkpoint"],
        num_classes=model_cfg["num_classes"],
        lora_rank=model_cfg["lora_rank"],
        lora_alpha=model_cfg["lora_alpha"],
        dirt_hidden_dim=model_cfg["dirt_hidden_dim"],
        dirt_retain_ratio=model_cfg["dirt_retain_ratio"],
        use_pire=model_cfg["use_pire"],
        pire_strength=model_cfg["pire_strength"],
    )


def main() -> None:
    parser = build_common_parser("Distributed testing for PIRE-DIRT")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)

    set_seed(config["seed"])
    device = init_distributed(
        backend=config["distributed"]["backend"],
        timeout_minutes=config["distributed"]["timeout_minutes"],
    )

    try:
        model = build_model(config).to(device)
        load_checkpoint(
            model,
            config["paths"]["trained_checkpoint"],
            strict=True,
            remap_legacy_names=True,
        )
        if is_main_process():
            print(f"[Checkpoint] Loaded {config['paths']['trained_checkpoint']}")

        local_rank = device.index
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config["train"]["find_unused_parameters"],
        )
        results = evaluate_all_domains(
            model=model,
            test_root=config["paths"]["test_root"],
            device=device,
            config=config,
        )
        save_test_results(results, config["paths"]["output_dir"], config)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
