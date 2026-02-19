from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--num_workers", type=int, default=0)
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
    from patch_anomaly.data import TrainGoodPatchDataset
    from patch_anomaly.models import ae_loss, build_model, vae_loss

    torch.manual_seed(args.seed)

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
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if len(dataset) == 0:
        raise RuntimeError("No training patches found. Check dataset path/object name.")

    model = build_model(args.model_type, latent_channels=args.latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
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

        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "recon": mean_recon,
            "kld": mean_kld,
        }
        history.append(row)
        print(f"epoch={epoch} loss={mean_loss:.6f} recon={mean_recon:.6f} kld={mean_kld:.6f}")

        ckpt = {
            "state_dict": model.state_dict(),
            "model_type": args.model_type,
            "latent_channels": args.latent_channels,
            "patch_size": args.patch_size,
            "object_name": args.object_name,
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(ckpt, args.output_dir / "best.pt")

    with (args.output_dir / "train_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved checkpoints to: {args.output_dir}")


if __name__ == "__main__":
    main()