from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from aigi_detector.data import AIGIBenchDataset, make_rank_subset
from aigi_detector.engine.evaluator import evaluate_loader
from aigi_detector.utils.checkpoint import save_checkpoint
from aigi_detector.utils.distributed import get_rank, get_world_size, is_main_process, reduce_sum
from aigi_detector.utils.seed import make_generator, seed_worker


def build_train_and_val_loaders(config: dict) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    input_cfg = config["input"]
    paths = config["paths"]
    train_cfg = config["train"]
    val_cfg = config["validation"]
    rank = get_rank()
    world_size = get_world_size()

    train_dataset = AIGIBenchDataset(
        root=paths["train_root"],
        crop_size=input_cfg["crop_size"],
        sdv_quality=input_cfg["sdv_quality"],
        jpeg_domains=tuple(input_cfg["jpeg_domains"]),
        image_extensions=tuple(input_cfg["image_extensions"]),
        is_train=True,
        return_none_on_failure=False,
        verbose=is_main_process(),
    )
    val_dataset = AIGIBenchDataset(
        root=paths["val_root"],
        crop_size=input_cfg["crop_size"],
        sdv_quality=input_cfg["sdv_quality"],
        jpeg_domains=tuple(input_cfg["jpeg_domains"]),
        image_extensions=tuple(input_cfg["image_extensions"]),
        is_train=False,
        return_none_on_failure=False,
        verbose=is_main_process(),
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config["seed"],
        drop_last=False,
    )
    val_rank_dataset = make_rank_subset(val_dataset, rank, world_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size_per_gpu"],
        shuffle=False,
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=train_cfg["num_workers"] > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(config["seed"]),
    )
    val_loader = DataLoader(
        val_rank_dataset,
        batch_size=val_cfg["batch_size_per_gpu"],
        shuffle=False,
        num_workers=val_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=make_generator(config["seed"]),
    )
    return train_loader, val_loader, train_sampler


def train_one_epoch(
    model: DDP,
    loader: DataLoader,
    sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_grad_norm: float,
) -> dict[str, float]:
    sampler.set_epoch(epoch)
    model.train()

    local_loss_sum = 0.0
    local_correct = 0
    local_total = 0
    iterator = tqdm(loader, desc=f"[Train] Epoch {epoch + 1}", disable=not is_main_process())

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_grad_norm,
        )
        optimizer.step()

        batch_size = labels.size(0)
        local_loss_sum += float(loss.item()) * batch_size
        local_correct += int((logits.argmax(dim=1) == labels).sum().item())
        local_total += batch_size

        if is_main_process():
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * local_correct / max(local_total, 1):.2f}%",
            )

    stats = torch.tensor(
        [local_loss_sum, float(local_correct), float(local_total)],
        dtype=torch.float64,
        device=device,
    )
    stats = reduce_sum(stats)
    global_loss = stats[0].item() / max(stats[2].item(), 1.0)
    global_acc = 100.0 * stats[1].item() / max(stats[2].item(), 1.0)
    return {"loss": global_loss, "acc": global_acc}


def run_training(model: DDP, device: torch.device, config: dict) -> None:
    train_loader, val_loader, train_sampler = build_train_and_val_loaders(config)
    train_cfg = config["train"]

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    if is_main_process():
        count = sum(parameter.numel() for parameter in trainable_parameters)
        print(f"[Model] Trainable parameters: {count:,}")

    best_metric = float("-inf")
    output_path = Path(config["paths"]["output_dir"]) / "AIGIbench_best.pth"

    for epoch in range(train_cfg["epochs"]):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            sampler=train_sampler,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_grad_norm=train_cfg["max_grad_norm"],
        )

        should_validate = (epoch + 1) % train_cfg["validate_every"] == 0
        if should_validate:
            val_metrics = evaluate_loader(
                model=model,
                loader=val_loader,
                device=device,
                description=f"[Val] Epoch {epoch + 1}",
            )
        else:
            val_metrics = None

        if is_main_process():
            if val_metrics is None:
                print(
                    f"[Epoch {epoch + 1:03d}] train_loss={train_metrics['loss']:.4f} "
                    f"train_acc={train_metrics['acc']:.2f}%"
                )
            else:
                print(
                    f"[Epoch {epoch + 1:03d}] train_loss={train_metrics['loss']:.4f} "
                    f"train_acc={train_metrics['acc']:.2f}% | "
                    f"val_acc={val_metrics['acc']:.2f}% val_ap={val_metrics['ap']:.2f}%"
                )

                selection_metric = train_cfg["selection_metric"]
                current_metric = float(val_metrics[selection_metric])
                if current_metric > best_metric:
                    best_metric = current_metric
                    save_checkpoint(
                        output_path,
                        model=model.module,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={"train": train_metrics, "validation": val_metrics},
                        config=config,
                    )
                    print(f"[Checkpoint] Saved best model to {output_path}")

        dist.barrier()
