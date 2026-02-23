from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PatchIndex:
    image_path: Path
    y: int
    x: int
    padded_h: int
    padded_w: int


def load_rgb_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def pad_to_patch_multiple(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = image.shape
    padded_h = ceil(height / patch_size) * patch_size
    padded_w = ceil(width / patch_size) * patch_size
    pad_h = padded_h - height
    pad_w = padded_w - width
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def iter_patch_coords(height: int, width: int, patch_size: int) -> Iterable[tuple[int, int]]:
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            yield y, x


def extract_patches(image: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, list[tuple[int, int]], tuple[int, int]]:
    _, original_h, original_w = image.shape
    padded = pad_to_patch_multiple(image, patch_size)
    _, padded_h, padded_w = padded.shape
    coords = list(iter_patch_coords(padded_h, padded_w, patch_size))
    patches = [padded[:, y:y + patch_size, x:x + patch_size] for y, x in coords]
    return torch.stack(patches), coords, (original_h, original_w)


def stitch_patch_maps(
    patch_maps: torch.Tensor,
    coords: list[tuple[int, int]],
    original_hw: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    original_h, original_w = original_hw
    padded_h = ceil(original_h / patch_size) * patch_size
    padded_w = ceil(original_w / patch_size) * patch_size

    full = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    counts = torch.zeros((padded_h, padded_w), dtype=torch.float32)

    for patch_map, (y, x) in zip(patch_maps, coords):
        full[y:y + patch_size, x:x + patch_size] += patch_map
        counts[y:y + patch_size, x:x + patch_size] += 1.0

    full /= torch.clamp_min(counts, 1.0)
    full = full[:original_h, :original_w]
    return full.numpy()


class TrainGoodPatchDataset(Dataset):
    def __init__(
        self,
        dataset_base_dir: Path,
        object_name: str,
        patch_size: int,
        image_limit: int | None = None,
        cache_size: int = 8,
    ):
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        self.patch_size = patch_size
        self.cache_size = max(int(cache_size), 1)
        image_dir = dataset_base_dir / object_name / "train" / "good"
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Training directory not found: {image_dir}. "
                "Expected structure: <dataset_base_dir>/<object_name>/train/good/*.png"
            )
        image_paths = sorted(image_dir.glob("*.png"))
        if image_limit is not None:
            image_paths = image_paths[:image_limit]

        self.index: list[PatchIndex] = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                width, height = img.size
            padded_h = ceil(height / patch_size) * patch_size
            padded_w = ceil(width / patch_size) * patch_size
            for y, x in iter_patch_coords(padded_h, padded_w, patch_size):
                self.index.append(PatchIndex(image_path, y, x, padded_h, padded_w))

        self._image_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.index[idx]

        image = self._image_cache.get(item.image_path)
        if image is None:
            image = load_rgb_image(item.image_path)
            image = pad_to_patch_multiple(image, self.patch_size)
            self._image_cache[item.image_path] = image
            if len(self._image_cache) > self.cache_size:
                self._image_cache.popitem(last=False)
        else:
            self._image_cache.move_to_end(item.image_path)

        patch = image[:, item.y:item.y + self.patch_size, item.x:item.x + self.patch_size]
        return patch


def list_test_public_images(dataset_base_dir: Path, object_name: str) -> list[tuple[str, Path]]:
    test_dir = dataset_base_dir / object_name / "test_public"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}. "
            "Expected structure: <dataset_base_dir>/<object_name>/test_public/<defect_name>/*.png"
        )
    result: list[tuple[str, Path]] = []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir() or defect_dir.name == "ground_truth":
            continue
        for image_path in sorted(defect_dir.glob("*.png")):
            result.append((defect_dir.name, image_path))
    return result
