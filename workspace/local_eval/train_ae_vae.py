from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AE/VAE baseline on MVTec AD2 patches.")
    parser.add_argument("--dataset_base_dir", type=Path, default=Path("data/can"))
    parser.add_argument("--object_name", type=str, default="can")
    parser.add_argument("--model_type", choices=["ae", "vae"], default="ae")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
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
        default=Path("workspace/local_eval/experiments/exp01/checkpoints"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from patch_anomaly.data import (
        TrainGoodPatchDataset,
        extract_patches,
        list_good_images,
        load_rgb_image,
    )
    from patch_anomaly.models import ae_loss, build_model, vae_loss

    if args.patch_size <= 0:
        raise ValueError("--patch_size must be a positive integer.")
    if args.patch_size % 16 != 0:
        raise ValueError(
            f"--patch_size ({args.patch_size}) must be divisible by 16 "
            "because the model downsamples 4x by stride-2 convolutions."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TrainGoodPatchDataset(
        dataset_base_dir=args.dataset_base_dir,
        object_name=args.object_name,
        patch_size=args.patch_size,
        image_limit=args.image_limit,
        cache_size=args.cache_size,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training patches found. Check dataset path/object name.")

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
    print(
        f"Validation source: split='{args.validation_split}' images={len(validation_image_paths)}"
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model_type, latent_channels=args.latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            if args.model_type == "ae":
                recon = model(batch)
                loss = ae_loss(recon, batch)
                recon_loss = loss
                kld = torch.zeros((), device=device)
            else:
                recon, mu, logvar = model(batch)
                loss, recon_loss, kld = vae_loss(recon, batch, mu, logvar, beta=args.beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            total_recon += recon_loss.item() * batch.size(0)
            total_kld += kld.item() * batch.size(0)
            progress.set_postfix(loss=loss.item())

        mean_loss = total_loss / len(dataset)
        mean_recon = total_recon / len(dataset)
        mean_kld = total_kld / len(dataset)

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
                    batch = patches[start:start + args.batch_size].to(device)
                    if args.model_type == "ae":
                        recon = model(batch)
                        loss = ae_loss(recon, batch)
                        recon_loss = loss
                        kld = torch.zeros((), device=device)
                    else:
                        recon, mu, logvar = model(batch)
                        loss, recon_loss, kld = vae_loss(recon, batch, mu, logvar, beta=args.beta)
                    bs = batch.size(0)
                    total_val_loss += loss.item() * bs
                    total_val_recon += recon_loss.item() * bs
                    total_val_kld += kld.item() * bs
                    total_val_patches += bs
        if total_val_patches == 0:
            raise RuntimeError("Validation produced zero patches. Check validation split data.")
        val_loss = total_val_loss / total_val_patches
        val_recon = total_val_recon / total_val_patches
        val_kld = total_val_kld / total_val_patches

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
        print(
            f"epoch={epoch} loss={mean_loss:.6f} recon={mean_recon:.6f} kld={mean_kld:.6f} "
            f"val_loss={val_loss:.6f} val_recon={val_recon:.6f} val_kld={val_kld:.6f}"
        )

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
            torch.save(ckpt, args.output_dir / "best.pt")

    with (args.output_dir / "train_history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"Saved checkpoints to: {args.output_dir}")


if __name__ == "__main__":
    main()
