from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


@dataclass(frozen=True)
class MethodSpec:
    name: str
    train_cmd_template: list[str]
    infer_cmd_template: list[str]
    checkpoint_relpath: str = "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train/infer/eval for multiple anomaly-detection methods and compare metrics."
    )
    parser.add_argument("--dataset_base_dir", type=Path, default=Path("data/can"))
    parser.add_argument("--object_name", type=str, default="can")
    parser.add_argument("--methods", nargs="+", default=["ae", "vae"])
    parser.add_argument("--custom_methods_config", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--experiment_root", type=Path, default=Path("workspace/local_eval/benchmark_runs"))
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_channels", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--image_limit", type=int, default=None)
    parser.add_argument("--validation_image_limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_size", type=int, default=8)
    parser.add_argument("--pro_integration_limit", type=float, default=0.3)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_infer", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _template_builtin_specs(py: str) -> dict[str, MethodSpec]:
    def _ae_vae_spec(model_type: str) -> MethodSpec:
        return MethodSpec(
            name=model_type,
            train_cmd_template=[
                py, "workspace/local_eval/train_ae_vae.py",
                "--dataset_base_dir", "{dataset_base_dir}",
                "--object_name", "{object_name}",
                "--model_type", model_type,
                "--patch_size", "{patch_size}",
                "--batch_size", "{batch_size}",
                "--epochs", "{epochs}",
                "--lr", "{lr}",
                "--latent_channels", "{latent_channels}",
                "--beta", "{beta}",
                "--validation_split", "{validation_split}",
                "--output_dir", "{checkpoint_dir}",
                "--seed", "{seed}",
                "--num_workers", "{num_workers}",
                "--cache_size", "{cache_size}",
            ],
            infer_cmd_template=[
                py, "workspace/local_eval/infer_ae_vae.py",
                "--dataset_base_dir", "{dataset_base_dir}",
                "--object_name", "{object_name}",
                "--checkpoint", "{checkpoint}",
                "--anomaly_maps_dir", "{anomaly_maps_dir}",
                "--batch_size", "{batch_size}",
            ],
        )

    return {t: _ae_vae_spec(t) for t in ("ae", "vae")}


def _load_custom_specs(config_path: Path | None) -> dict[str, MethodSpec]:
    if config_path is None:
        return {}
    with config_path.open() as f:
        payload = json.load(f)
    methods = payload.get("methods", [])
    result: dict[str, MethodSpec] = {}
    for item in methods:
        name = item["name"]
        result[name] = MethodSpec(
            name=name,
            train_cmd_template=item["train_cmd_template"],
            infer_cmd_template=item["infer_cmd_template"],
            checkpoint_relpath=item.get("checkpoint_relpath", "best.pt"),
        )
    return result


def _render_command(template: list[str], values: dict[str, str]) -> list[str]:
    return [token.format_map(values) for token in template]


def _maybe_append_optional_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    updated = list(cmd)
    if args.image_limit is not None:
        updated.extend(["--image_limit", str(args.image_limit)])
    if args.validation_image_limit is not None:
        updated.extend(["--validation_image_limit", str(args.validation_image_limit)])
    return updated


def _run_command(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=cwd, check=True)


def _read_metrics(metrics_json: Path, object_name: str) -> dict[str, float]:
    with metrics_json.open() as f:
        payload = json.load(f)
    if object_name not in payload:
        raise KeyError(f"Metrics file {metrics_json} does not contain object '{object_name}'.")
    return {
        "au_pro": float(payload[object_name]["au_pro"]),
        "au_roc": float(payload[object_name]["classification_au_roc"]),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    agg_rows: list[dict[str, Any]] = []
    for method_name, method_rows in sorted(by_method.items()):
        au_pro_values = [float(r["au_pro"]) for r in method_rows]
        au_roc_values = [float(r["au_roc"]) for r in method_rows]
        agg_rows.append(
            {
                "method": method_name,
                "runs": len(method_rows),
                "au_pro_mean": mean(au_pro_values),
                "au_pro_std": pstdev(au_pro_values) if len(method_rows) > 1 else 0.0,
                "au_roc_mean": mean(au_roc_values),
                "au_roc_std": pstdev(au_roc_values) if len(method_rows) > 1 else 0.0,
            }
        )
    return agg_rows


def main() -> None:
    args = parse_args()
    root = repo_root()
    py = sys.executable

    # Resolve relative paths to absolute so they work in any subprocess cwd
    # (the evaluator runs from mvtec_ad_evaluation/, not the repo root).
    args.dataset_base_dir = args.dataset_base_dir.resolve() # type: ignore[attr-defined]
    args.experiment_root = args.experiment_root.resolve() # type: ignore[attr-defined]

    builtins = _template_builtin_specs(py=py)
    custom = _load_custom_specs(args.custom_methods_config)
    registry = {**builtins, **custom}

    missing = [m for m in args.methods if m not in registry]
    if missing:
        raise ValueError(
            f"Unknown method(s): {missing}. "
            f"Available: {sorted(registry.keys())}. "
            "Use --custom_methods_config to add methods."
        )

    run_rows: list[dict[str, Any]] = []
    args.experiment_root.mkdir(parents=True, exist_ok=True)

    for method_name in args.methods:
        method = registry[method_name]
        for seed in args.seeds:
            run_dir = args.experiment_root / args.object_name / method_name / f"seed_{seed}"
            checkpoint_dir = run_dir / "checkpoints"
            anomaly_maps_dir = run_dir / "anomaly_maps"
            metrics_dir = run_dir / "metrics"
            metrics_json = metrics_dir / "metrics.json"
            checkpoint = checkpoint_dir / method.checkpoint_relpath

            values = {
                "dataset_base_dir": str(args.dataset_base_dir),
                "object_name": args.object_name,
                "patch_size": str(args.patch_size),
                "batch_size": str(args.batch_size),
                "epochs": str(args.epochs),
                "lr": str(args.lr),
                "latent_channels": str(args.latent_channels),
                "beta": str(args.beta),
                "validation_split": args.validation_split,
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint": str(checkpoint),
                "anomaly_maps_dir": str(anomaly_maps_dir),
                "metrics_dir": str(metrics_dir),
                "seed": str(seed),
                "num_workers": str(args.num_workers),
                "cache_size": str(args.cache_size),
            }

            if not args.skip_train:
                train_cmd = _render_command(method.train_cmd_template, values)
                train_cmd = _maybe_append_optional_args(train_cmd, args)
                _run_command(train_cmd, cwd=root)

            if not args.skip_infer:
                infer_cmd = _render_command(method.infer_cmd_template, values)
                _run_command(infer_cmd, cwd=root)

            if not args.skip_eval:
                eval_cmd = [
                    py,
                    "evaluate_experiment.py",
                    "--dataset_base_dir",
                    str(args.dataset_base_dir),
                    "--anomaly_maps_dir",
                    str(anomaly_maps_dir),
                    "--output_dir",
                    str(metrics_dir),
                    "--evaluated_objects",
                    args.object_name,
                    "--pro_integration_limit",
                    str(args.pro_integration_limit),
                ]
                _run_command(eval_cmd, cwd=root / "mvtec_ad_evaluation")

            if not metrics_json.is_file():
                raise FileNotFoundError(f"Metrics not found after run: {metrics_json}")
            metrics = _read_metrics(metrics_json, object_name=args.object_name)
            run_rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    **metrics,
                }
            )

    summary_dir = args.experiment_root / args.object_name / "summary"
    per_run_csv = summary_dir / "per_run.csv"
    by_method_csv = summary_dir / "by_method.csv"
    by_method_json = summary_dir / "by_method.json"

    run_fields = ["method", "seed", "au_pro", "au_roc"]
    _write_csv(per_run_csv, run_fields, run_rows)

    by_method_rows = _aggregate(run_rows)
    method_fields = [
        "method",
        "runs",
        "au_pro_mean",
        "au_pro_std",
        "au_roc_mean",
        "au_roc_std",
    ]
    _write_csv(by_method_csv, method_fields, by_method_rows)
    with by_method_json.open("w") as f:
        json.dump(by_method_rows, f, indent=2)

    print("\nComparison finished.")
    print(f"- per-run results: {per_run_csv}")
    print(f"- aggregated results: {by_method_csv}")
    print(f"- aggregated json: {by_method_json}")


if __name__ == "__main__":
    main()
