from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset

from .transforms import build_image_transform, jpeg_compress

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AIGIBenchDataset(Dataset):
    """Dataset for ``root/domain/{0_real,1_fake}`` directory layouts."""

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 224,
        sdv_quality: int = 90,
        jpeg_domains: tuple[str, ...] = ("sdv1.4", "1.4", "sd15"),
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        domain_name: str | None = None,
        is_train: bool = False,
        max_retry: int = 10,
        return_none_on_failure: bool = False,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.sdv_quality = sdv_quality
        self.jpeg_domains = set(jpeg_domains)
        self.image_extensions = {ext.lower() for ext in image_extensions}
        self.is_train = is_train
        self.max_retry = max_retry
        self.return_none_on_failure = return_none_on_failure
        self.transform = build_image_transform(crop_size)
        self.samples: list[dict[str, Any]] = []

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        if domain_name is None:
            domain_paths = [path for path in sorted(self.root.iterdir()) if path.is_dir()]
        else:
            requested = self.root / domain_name
            if requested.is_dir():
                domain_paths = [requested]
            elif self.root.name == domain_name:
                domain_paths = [self.root]
            else:
                raise FileNotFoundError(f"Domain directory does not exist: {requested}")

        for domain_path in domain_paths:
            domain = domain_path.name
            for class_dir, label in (("0_real", 0), ("1_fake", 1)):
                image_dir = domain_path / class_dir
                if not image_dir.is_dir():
                    continue

                for image_path in sorted(image_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in self.image_extensions:
                        self.samples.append(
                            {
                                "path": image_path,
                                "label": label,
                                "domain": domain,
                            }
                        )

        if not self.samples:
            raise RuntimeError(f"No supported images were found under: {self.root}")

        if verbose:
            label = domain_name if domain_name is not None else str(self.root)
            print(f"[Dataset] {label}: {len(self.samples)} images")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int):
        last_error: Exception | None = None

        for _ in range(self.max_retry):
            sample = self.samples[index]
            try:
                image = self._load_rgb(sample["path"])
                if self.is_train and sample["domain"] in self.jpeg_domains:
                    image = jpeg_compress(image, quality=self.sdv_quality)
                image = self.transform(image)
                return image, int(sample["label"])
            except Exception as error:  # corrupted images should not stop long evaluations
                last_error = error
                index = random.randint(0, len(self.samples) - 1)

        if self.return_none_on_failure:
            return None
        raise RuntimeError("Failed to load an image after repeated retries") from last_error


def collate_drop_none(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    images, labels = zip(*valid)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def discover_domains(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Test root does not exist: {root}")

    domains = []
    for domain_path in sorted(root.iterdir()):
        if not domain_path.is_dir():
            continue
        if (domain_path / "0_real").is_dir() or (domain_path / "1_fake").is_dir():
            domains.append(domain_path.name)
    return domains


def make_rank_subset(dataset: Dataset, rank: int, world_size: int) -> Subset:
    """Partition without padding, avoiding duplicated evaluation samples."""
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)
