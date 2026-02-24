"""
train_ae_vae.py – Train an AutoEncoder (AE) or Variational AutoEncoder (VAE)
on *good* patches from MVTec AD2 images.

Workflow:
  1. Parse CLI arguments (model type, hyperparameters, paths).
  2. Set random seeds for reproducibility.
  3. Build the patch dataset from the training split.
  4. Run the training loop with a validation check every epoch.
  5. Save the best and last checkpoints, plus training history as JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── module-level logger ────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── CLI argument parsing ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AE/VAE baseline on MVTec AD2 patches."
    )
    parser.add_argument("--dataset_base_dir", type=Path, default=Path("data/can"))
    parser.add_argument("--object_name", type=str, default="can")
    parser.add_argument("--model_type", choices=["ae", "vae"], default="ae")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_channels", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--image_limit", type=int, default=None)
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--validation_image_limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("workspace/local_eval/benchmark_runs/checkpoints"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── configure logging (timestamps + level for every message) ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("=== train_ae_vae.py started ===")
    logger.info(
        "Config: model=%s  epochs=%d  lr=%s  batch=%d  patch=%d  seed=%d",
        args.model_type,
        args.epochs,
        args.lr,
        args.batch_size,
        args.patch_size,
        args.seed,
    )

    # ── delayed imports (heavy libraries) ──
    import torch
    from torch.utils.data import DataLoader
    from patch_anomaly.data import (
        TrainGoodPatchDataset,
        extract_patches,
        list_good_images,
        load_rgb_image,
    )
    from patch_anomaly.models import ae_loss, build_model, vae_loss

    # ── input validation ──
    if args.patch_size <= 0:
        raise ValueError("--patch_size must be a positive integer.")
    if args.patch_size % 16 != 0:
        raise ValueError(
            f"--patch_size ({args.patch_size}) must be divisible by 16 "
            "because the model downsamples 4x by stride-2 convolutions."
        )

    # ── seed everything for reproducibility ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seeds set to %d", args.seed)

    # ── pick the best available device (CUDA > MPS > CPU) ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── build the training dataset (lazy patch extraction) ──
    logger.info("Loading training dataset from %s ...", args.dataset_base_dir)
    dataset = TrainGoodPatchDataset(
        dataset_base_dir=args.dataset_base_dir,
        object_name=args.object_name,
        patch_size=args.patch_size,
        image_limit=args.image_limit,
        cache_size=args.cache_size,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training patches found. Check dataset path/object name.")
    logger.info("Training dataset: %d patches", len(dataset))

    # ── load validation images (used for per-epoch eval) ──
    validation_image_paths = list_good_images(
        dataset_base_dir=args.dataset_base_dir,
        object_name=args.object_name,
        split=args.validation_split,
        image_limit=args.validation_image_limit,
    )
    if len(validation_image_paths) == 0:
        raise RuntimeError(
            f"No validation images found in split '{args.validation_split}/good'."
        )
    logger.info(
        "Validation: split='%s', %d images",
        args.validation_split,
        len(validation_image_paths),
    )

    # ── data loader with shuffling ──
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ── build model and optimizer ──
    model = build_model(args.model_type, latent_channels=args.latent_channels).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    epochs_no_improve = 0
    history: list[dict[str, float]] = []

    # ══════════════════════════════════════════════════════════════
    # TRAINING LOOP — one epoch = one full pass over all patches
    # ══════════════════════════════════════════════════════════════
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0

        # ── iterate over mini-batches ──
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

            # Forward pass — AE returns just the reconstruction,
            # VAE returns (recon, mu, logvar) for the KL term
            if args.model_type == "ae":
                recon = model(batch)
                loss = ae_loss(recon, batch)
                recon_loss = loss
                kld = torch.zeros((), device=device)
            else:
                recon, mu, logvar = model(batch)
                loss, recon_loss, kld = vae_loss(
                    recon, batch, mu, logvar, beta=args.beta
                )

            # Backward pass + weight update
            loss.backward()
            optimizer.step()

            # Accumulate per-sample losses for epoch-level averages
            total_loss += loss.item() * batch.size(0)
            total_recon += recon_loss.item() * batch.size(0)
            total_kld += kld.item() * batch.size(0)
            progress.set_postfix(loss=loss.item())

        # ── epoch-level training averages ──
        mean_loss = total_loss / len(dataset)
        mean_recon = total_recon / len(dataset)
        mean_kld = total_kld / len(dataset)

        # ═══════════════════════════════════════════════════════
        # VALIDATION — no gradient computation, evaluate on
        #   patches extracted from the validation split
        # ═══════════════════════════════════════════════════════
        model.eval()
        total_val_loss = 0.0
        total_val_recon = 0.0
        total_val_kld = 0.0
        total_val_patches = 0
        with torch.no_grad():
            for image_path in validation_image_paths:
                image = load_rgb_image(image_path)
                patches, _, _ = extract_patches(image, args.patch_size)
                for start in range(0, patches.size(0), args.batch_size):
                    batch = patches[start : start + args.batch_size].to(device)
                    if args.model_type == "ae":
                        recon = model(batch)
                        loss = ae_loss(recon, batch)
                        recon_loss = loss
                        kld = torch.zeros((), device=device)
                    else:
                        recon, mu, logvar = model(batch)
                        loss, recon_loss, kld = vae_loss(
                            recon, batch, mu, logvar, beta=args.beta
                        )
                    bs = batch.size(0)
                    total_val_loss += loss.item() * bs
                    total_val_recon += recon_loss.item() * bs
                    total_val_kld += kld.item() * bs
                    total_val_patches += bs

        if total_val_patches == 0:
            raise RuntimeError(
                "Validation produced zero patches. Check validation split data."
            )
        val_loss = total_val_loss / total_val_patches
        val_recon = total_val_recon / total_val_patches
        val_kld = total_val_kld / total_val_patches

        # ── log epoch results ──
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "recon": mean_recon,
            "kld": mean_kld,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_kld": val_kld,
        }
        history.append(row)
        logger.info(
            "Epoch %d/%d [%.1fs]  train_loss=%.6f recon=%.6f kld=%.6f  "
            "val_loss=%.6f val_recon=%.6f val_kld=%.6f",
            epoch,
            args.epochs,
            elapsed,
            mean_loss,
            mean_recon,
            mean_kld,
            val_loss,
            val_recon,
            val_kld,
        )

        # ── checkpoint saving ──
        # Always save "last.pt"; also save "best.pt" when val loss improves
        ckpt = {
            "state_dict": model.state_dict(),
            "model_type": args.model_type,
            "latent_channels": args.latent_channels,
            "patch_size": args.patch_size,
            "object_name": args.object_name,
            "seed": args.seed,
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(ckpt, args.output_dir / "best.pt")
            logger.info("  ↳ New best val_loss=%.6f — saved best.pt", val_loss)
        elif args.patience > 0:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    args.patience,
                )
                break

    # ── persist training history and run config ──
    with (args.output_dir / "train_history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    logger.info("Training complete. Checkpoints saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
