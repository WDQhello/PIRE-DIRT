from __future__ import annotations

import io
from collections.abc import Sequence

from PIL import Image
from torchvision import transforms


def translate_duplicate(image: Image.Image, crop_size: int) -> Image.Image:
    """Tile an image until both dimensions are at least ``crop_size``."""
    width, height = image.size
    if width >= crop_size and height >= crop_size:
        return image

    new_width = max(width, crop_size)
    new_height = max(height, crop_size)
    canvas = Image.new("RGB", (new_width, new_height))

    for x in range(0, new_width, width):
        for y in range(0, new_height, height):
            canvas.paste(image, (x, y))

    return canvas


def jpeg_compress(image: Image.Image, quality: int = 90) -> Image.Image:
    """Apply in-memory JPEG compression and return an RGB image."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


class TranslateDuplicateTransform:
    """Pickle-safe transform replacing a lambda used by multiprocessing loaders."""

    def __init__(self, crop_size: int) -> None:
        self.crop_size = crop_size

    def __call__(self, image: Image.Image) -> Image.Image:
        return translate_duplicate(image, self.crop_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size={self.crop_size})"


def build_image_transform(
    crop_size: int,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    return transforms.Compose(
        [
            TranslateDuplicateTransform(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
