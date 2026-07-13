from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from aigi_detector.data import AIGIBenchDataset, collate_drop_none, discover_domains, make_rank_subset
from aigi_detector.utils.distributed import get_rank, get_world_size, is_main_process
from aigi_detector.utils.metrics import compute_binary_metrics
from aigi_detector.utils.seed import make_generator, seed_worker


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> dict[str, Any] | None:
    model.eval()
    rank = get_rank()
    # Evaluation subsets can contain different numbers of batches on each rank.
    # Bypass the DDP wrapper during forward passes to avoid collective buffer
    # synchronization and safely gather predictions only after local inference.
    inference_model = model.module if isinstance(model, DDP) else model

    local_labels: list[int] = []
    local_predictions: list[int] = []
    local_scores: list[float] = []

    iterator = tqdm(loader, desc=description, disable=not is_main_process())
    for batch in iterator:
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = inference_model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        local_labels.extend(labels.cpu().tolist())
        local_predictions.extend(predictions.cpu().tolist())
        local_scores.extend(probabilities[:, 1].cpu().tolist())

    payload = {
        "labels": local_labels,
        "predictions": local_predictions,
        "scores": local_scores,
    }

    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, payload)

    if not is_main_process():
        return None

    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    for item in gathered:
        labels.extend(item["labels"])
        predictions.extend(item["predictions"])
        scores.extend(item["scores"])

    return compute_binary_metrics(
        np.asarray(labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
    )


def evaluate_all_domains(
    model: DDP,
    test_root: str,
    device: torch.device,
    config: dict,
) -> list[dict[str, Any]]:
    rank = get_rank()
    world_size = get_world_size()
    domains = discover_domains(test_root)

    if is_main_process():
        print(f"[Info] Found {len(domains)} test domains")
        for domain in domains:
            print(f"  - {domain}")

    input_cfg = config["input"]
    test_cfg = config["test"]
    results: list[dict[str, Any]] = []

    for domain in domains:
        dataset = AIGIBenchDataset(
            root=test_root,
            domain_name=domain,
            crop_size=input_cfg["crop_size"],
            sdv_quality=input_cfg["sdv_quality"],
            jpeg_domains=tuple(input_cfg["jpeg_domains"]),
            image_extensions=tuple(input_cfg["image_extensions"]),
            is_train=False,
            return_none_on_failure=True,
            verbose=is_main_process(),
        )
        rank_dataset = make_rank_subset(dataset, rank, world_size)
        loader = DataLoader(
            rank_dataset,
            batch_size=test_cfg["batch_size_per_gpu"],
            shuffle=False,
            num_workers=test_cfg["num_workers"],
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=make_generator(config["seed"]),
            collate_fn=collate_drop_none,
        )

        metrics = evaluate_loader(model, loader, device, f"[Test] {domain}")
        if is_main_process() and metrics is not None:
            result = {"domain": domain, **metrics}
            results.append(result)
            print(
                f"[{domain}] ACC: {result['acc']:.2f}% | AP: {result['ap']:.2f}% | "
                f"Correct: {result['correct']} / {result['total']}"
            )
        dist.barrier()

    return results


def save_test_results(results: list[dict[str, Any]], output_dir: str | Path, config: dict) -> None:
    if not is_main_process():
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / config["test"]["save_csv"]
    summary_path = output_dir / config["test"]["save_summary"]

    fieldnames = ["domain", "acc", "ap", "correct", "total"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    valid_acc = [item["acc"] for item in results if not np.isnan(item["acc"])]
    valid_ap = [item["ap"] for item in results if not np.isnan(item["ap"])]
    summary = {
        "num_domains": len(results),
        "mean_acc": float(np.mean(valid_acc)) if valid_acc else float("nan"),
        "mean_ap": float(np.mean(valid_ap)) if valid_ap else float("nan"),
        "csv_path": str(csv_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Mean ACC: {summary['mean_acc']:.2f}%")
    print(f"Mean AP : {summary['mean_ap']:.2f}%")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print("=" * 80)
