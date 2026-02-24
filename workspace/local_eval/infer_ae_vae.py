"""
infer_ae_vae.py – Run inference with a trained AE/VAE checkpoint and export
per-image anomaly maps as TIFF files (required by the MVTec AD2 evaluator).

Workflow:
  1. Load the saved checkpoint (model weights + metadata).
  2. Iterate over all test_public images.
  3. For each image: extract patches → reconstruct → compute pixel-wise
     MSE → stitch patch-level error maps into a full-resolution anomaly map.
  4. Write the anomaly map as a float32 TIFF to the expected directory
     structure: <anomaly_maps_dir>/<object>/<split>/<defect>/<image_id>.tiff
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import tifffile
from tqdm import tqdm

# ── module-level logger ────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── CLI argument parsing ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AE/VAE inference and export TIFF anomaly maps."
    )
    parser.add_argument("--dataset_base_dir", type=Path, default=Path("data/can"))
    parser.add_argument("--object_name", type=str, default="can")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("workspace/local_eval/benchmark_runs/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--anomaly_maps_dir",
        type=Path,
        default=Path("workspace/local_eval/benchmark_runs/anomaly_maps"),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── configure logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("=== infer_ae_vae.py started ===")

    # ── delayed imports (heavy libraries) ──
    import torch
    from patch_anomaly.data import (
        extract_patches,
        list_test_public_images,
        load_rgb_image,
        stitch_patch_maps,
    )
    from patch_anomaly.models import build_model

    # ── input validation ──
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # ── pick the best available device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ── load the saved checkpoint ──
    logger.info("Loading checkpoint from %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_type = checkpoint["model_type"]
    patch_size = int(checkpoint["patch_size"])
    latent_channels = int(checkpoint["latent_channels"])
    if patch_size % 16 != 0:
        raise ValueError(
            f"Invalid checkpoint patch_size={patch_size}. "
            "Expected a value divisible by 16 for this model architecture."
        )
    logger.info(
        "Checkpoint info: model_type=%s  patch_size=%d  latent_channels=%d",
        model_type,
        patch_size,
        latent_channels,
    )

    # ── build model and load weights ──
    model = build_model(model_type=model_type, latent_channels=latent_channels).to(
        device
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # switch to evaluation mode (disables dropout etc.)

    # ── discover test images ──
    image_items = list_test_public_images(args.dataset_base_dir, args.object_name)
    if len(image_items) == 0:
        raise RuntimeError("No test_public images found.")
    logger.info("Found %d test_public images to process", len(image_items))

    # ══════════════════════════════════════════════════════════════
    # INFERENCE LOOP — process one image at a time
    # ══════════════════════════════════════════════════════════════
    t0 = time.time()
    with torch.no_grad():
        for defect_name, image_path in tqdm(image_items, desc="Infer test_public"):
            # 1) Load image and cut into patches
            image = load_rgb_image(image_path)
            patches, coords, original_hw = extract_patches(image, patch_size)

            # 2) Reconstruct each patch batch and compute pixel-wise MSE
            patch_maps = []
            for start in range(0, patches.size(0), args.batch_size):
                batch = patches[start : start + args.batch_size].to(device)

                if model_type == "ae":
                    recon = model(batch)
                else:
                    recon, _, _ = model(batch)  # discard mu/logvar at inference

                # Per-pixel MSE across the 3 colour channels → [B, H, W]
                err = torch.mean((recon - batch) ** 2, dim=1)
                patch_maps.append(err.detach().cpu())

            # 3) Stitch patch-level error maps into a full-resolution map
            patch_maps_tensor = torch.cat(patch_maps, dim=0)
            full_map = stitch_patch_maps(
                patch_maps=patch_maps_tensor,
                coords=coords,
                original_hw=original_hw,
                patch_size=patch_size,
            )

            # 4) Save the anomaly map as a compressed TIFF
            image_id = image_path.stem
            out_dir = (
                args.anomaly_maps_dir / args.object_name / "test_public" / defect_name
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{image_id}.tiff"

            tifffile.imwrite(out_path, full_map.astype("float32"), compression="zlib")
            logger.debug("Saved anomaly map: %s", out_path)

    elapsed = time.time() - t0
    logger.info(
        "Inference complete: %d images in %.1fs (%.2f img/s). Maps saved to: %s",
        len(image_items),
        elapsed,
        len(image_items) / max(elapsed, 1e-6),
        args.anomaly_maps_dir,
    )


if __name__ == "__main__":
    main()
