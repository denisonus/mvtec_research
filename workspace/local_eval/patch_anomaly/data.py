"""
data.py – Dataset utilities for the MVTec AD2 patch-based anomaly detection pipeline.

Responsibilities:
  1. Load RGB images from disk and normalise to [0, 1] float tensors.
  2. Pad images so their dimensions are exact multiples of the patch size.
  3. Cut images into non-overlapping patches (for training) and stitch
     per-patch anomaly maps back into full-resolution images (for inference).
  4. Provide a PyTorch Dataset that lazily loads images and serves patches
     with an LRU cache to keep memory bounded.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# ── module-level logger ────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── helper data class ──────────────────────────────────────────────────
@dataclass(frozen=True)
class PatchIndex:
    """Points to one patch inside an image: its file path + top-left corner
    in the *padded* image coordinate system."""

    image_path: Path
    y: int
    x: int
    padded_h: int
    padded_w: int


# ── image loading & preprocessing ──────────────────────────────────────


def load_rgb_image(image_path: Path) -> torch.Tensor:
    """Load a PNG image, convert to RGB, and return a [C, H, W] float32
    tensor with pixel values in [0, 1]."""
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0  # uint8 → [0,1]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()  # HWC → CHW
    logger.debug("Loaded image %s  shape=%s", image_path.name, tuple(tensor.shape))
    return tensor


def pad_to_patch_multiple(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Zero-pad the image so that both height and width are exact multiples
    of *patch_size*.  Returns the original tensor unchanged if no padding
    is needed."""
    _, height, width = image.shape
    padded_h = ceil(height / patch_size) * patch_size
    padded_w = ceil(width / patch_size) * patch_size
    pad_h = padded_h - height
    pad_w = padded_w - width
    if pad_h == 0 and pad_w == 0:
        return image
    logger.debug(
        "Padding image from (%d, %d) → (%d, %d)  [pad_h=%d, pad_w=%d]",
        height,
        width,
        padded_h,
        padded_w,
        pad_h,
        pad_w,
    )
    return F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)


def iter_patch_coords(
    height: int, width: int, patch_size: int
) -> Iterable[tuple[int, int]]:
    """Yield (y, x) top-left corners for a non-overlapping grid of patches."""
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            yield y, x


def extract_patches(
    image: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, list[tuple[int, int]], tuple[int, int]]:
    """Cut a [C, H, W] image into non-overlapping square patches.

    Returns:
        patches     – [N, C, patch_size, patch_size] tensor
        coords      – list of (y, x) top-left corners in padded space
        original_hw – (H, W) *before* padding, needed to crop back later
    """
    _, original_h, original_w = image.shape
    padded = pad_to_patch_multiple(image, patch_size)
    _, padded_h, padded_w = padded.shape
    coords = list(iter_patch_coords(padded_h, padded_w, patch_size))
    patches = [padded[:, y : y + patch_size, x : x + patch_size] for y, x in coords]
    logger.debug(
        "Extracted %d patches (%dx%d) from image %dx%d (padded %dx%d)",
        len(patches),
        patch_size,
        patch_size,
        original_h,
        original_w,
        padded_h,
        padded_w,
    )
    return torch.stack(patches), coords, (original_h, original_w)


# ── anomaly-map stitching (inference time) ─────────────────────────────


def stitch_patch_maps(
    patch_maps: torch.Tensor,
    coords: list[tuple[int, int]],
    original_hw: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """Stitch per-patch anomaly maps back into a full-resolution map.

    Overlapping regions (if any) are averaged.  The result is cropped to the
    original (un-padded) image dimensions.

    Returns a 2-D numpy array of shape (original_h, original_w).
    """
    original_h, original_w = original_hw
    padded_h = ceil(original_h / patch_size) * patch_size
    padded_w = ceil(original_w / patch_size) * patch_size

    # Accumulator buffers for averaging overlapping patches
    full = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    counts = torch.zeros((padded_h, padded_w), dtype=torch.float32)

    for patch_map, (y, x) in zip(patch_maps, coords):
        full[y : y + patch_size, x : x + patch_size] += patch_map
        counts[y : y + patch_size, x : x + patch_size] += 1.0

    full /= torch.clamp_min(counts, 1.0)  # safe division (avoid /0)
    full = full[:original_h, :original_w]  # crop back to original size
    logger.debug(
        "Stitched %d patch maps → full anomaly map %dx%d",
        len(coords),
        original_h,
        original_w,
    )
    return full.numpy()


# ── training dataset ───────────────────────────────────────────────────


class TrainGoodPatchDataset(Dataset):
    """PyTorch dataset that serves RANDOM crops from *good* training images."""

    def __init__(
            self,
            dataset_base_dir: Path,
            object_name: str,
            patch_size: int,
            image_limit: int | None = None,
            cache_size: int = 8,
            patches_per_image: int = 12,
    ):
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        self.patch_size = patch_size
        self.cache_size = max(int(cache_size), 1)
        self.patches_per_image = patches_per_image

        image_dir = dataset_base_dir / object_name / "train" / "good"
        self.image_paths = sorted(image_dir.glob("*.png"))
        if image_limit is not None:
            self.image_paths = self.image_paths[:image_limit]

        self._image_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

        logger.info(
            "TrainGoodPatchDataset ready: %d images → %d random patches per epoch",
            len(self.image_paths),
            len(self),
        )

    def __len__(self) -> int:
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_idx = idx // self.patches_per_image
        image_path = self.image_paths[image_idx]

        image = self._image_cache.get(image_path)
        if image is None:
            image = load_rgb_image(image_path)

            _, h, w = image.shape
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            if pad_h > 0 or pad_w > 0:
                image = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)

            self._image_cache[image_path] = image
            if len(self._image_cache) > self.cache_size:
                self._image_cache.popitem(last=False)
        else:
            self._image_cache.move_to_end(image_path)

        _, h, w = image.shape

        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)

        return image[:, y: y + self.patch_size, x: x + self.patch_size]


# ── image listing helpers ──────────────────────────────────────────────


def list_test_public_images(
    dataset_base_dir: Path, object_name: str
) -> list[tuple[str, Path]]:
    """Return a sorted list of (defect_name, image_path) for every image
    under ``<base>/<object>/test_public/<defect>/``."""
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
    logger.debug("Found %d test_public images across defect categories", len(result))
    return result


def list_good_images(
    dataset_base_dir: Path,
    object_name: str,
    split: str,
    image_limit: int | None = None,
) -> list[Path]:
    """Return sorted paths to *good* images in the given dataset split
    (e.g. 'train', 'validation').  Optionally cap at *image_limit*."""
    split_good_dir = dataset_base_dir / object_name / split / "good"
    if not split_good_dir.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {split_good_dir}. "
            "Expected structure: <dataset_base_dir>/<object_name>/<split>/good/*.png"
        )
    image_paths = sorted(split_good_dir.glob("*.png"))
    if image_limit is not None:
        image_paths = image_paths[:image_limit]
    logger.debug("Listed %d good images from split '%s'", len(image_paths), split)
    return image_paths
