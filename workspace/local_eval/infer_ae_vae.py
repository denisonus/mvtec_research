from __future__ import annotations

import argparse
from pathlib import Path

import tifffile
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AE/VAE inference and export TIFF anomaly maps.")
    parser.add_argument("--dataset_base_dir", type=Path, default=Path("data/can"))
    parser.add_argument("--object_name", type=str, default="can")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("workspace/local_eval/experiments/exp01/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--anomaly_maps_dir",
        type=Path,
        default=Path("workspace/local_eval/experiments/exp01/anomaly_maps"),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    
    import torch
    from patch_anomaly.data import extract_patches, list_test_public_images, load_rgb_image, stitch_patch_maps
    from patch_anomaly.models import build_model

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_type = checkpoint["model_type"]
    patch_size = int(checkpoint["patch_size"])
    latent_channels = int(checkpoint["latent_channels"])

    model = build_model(model_type=model_type, latent_channels=latent_channels).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image_items = list_test_public_images(args.dataset_base_dir, args.object_name)
    if len(image_items) == 0:
        raise RuntimeError("No test_public images found.")

    with torch.no_grad():
        for defect_name, image_path in tqdm(image_items, desc="Infer test_public"):
            image = load_rgb_image(image_path)
            patches, coords, original_hw = extract_patches(image, patch_size)

            patch_maps = []
            for start in range(0, patches.size(0), args.batch_size):
                batch = patches[start:start + args.batch_size].to(device)

                if model_type == "ae":
                    recon = model(batch)
                else:
                    recon, _, _ = model(batch)

                err = torch.mean((recon - batch) ** 2, dim=1)
                patch_maps.append(err.detach().cpu())

            patch_maps_tensor = torch.cat(patch_maps, dim=0)
            full_map = stitch_patch_maps(
                patch_maps=patch_maps_tensor,
                coords=coords,
                original_hw=original_hw,
                patch_size=patch_size,
            )

            image_id = image_path.stem
            out_dir = args.anomaly_maps_dir / args.object_name / "test_public" / defect_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{image_id}.tiff"

            tifffile.imwrite(out_path, full_map.astype("float32"), compression="zlib")

    print(f"Wrote anomaly maps to: {args.anomaly_maps_dir}")


if __name__ == "__main__":
    main()