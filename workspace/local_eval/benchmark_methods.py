"""
benchmark_methods.py – Orchestrate the full train → infer → evaluate pipeline
for one or more anomaly-detection methods across multiple random seeds, then
aggregate and compare the results.

This script is the top-level entry point for running reproducible benchmarks
on the MVTec AD2 dataset.  It:

  1. Builds a *method registry* (built-in AE/VAE + optional custom methods
     loaded from a JSON config file).
  2. For each (method × seed) combination:
       a. Trains the model              (unless --skip_train)
       b. Runs inference / anomaly maps (unless --skip_infer)
       c. Evaluates via the MVTec AD2 evaluation script (unless --skip_eval)
  3. Reads the per-run metric files, aggregates (mean ± std) across seeds,
     and writes per-run + by-method summary CSVs and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

# ── module-level logger ────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── method specification ──────────────────────────────────────────────
@dataclass(frozen=True)
class MethodSpec:
    """Describes one anomaly-detection method: its name and the CLI
    command templates for training and inference.  Placeholders like
    ``{dataset_base_dir}`` are filled in at run-time."""

    name: str
    train_cmd_template: list[str]
    infer_cmd_template: list[str]
    checkpoint_relpath: str = "best.pt"


# ── CLI argument parsing ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train/infer/eval for multiple anomaly-detection methods and compare metrics."
    )
    parser.add_argument("--dataset_base_dir", type=Path, default=Path("data/can"))
    parser.add_argument("--object_name", type=str, default="can")
    parser.add_argument("--methods", nargs="+", default=["ae", "vae"])
    parser.add_argument("--custom_methods_config", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--experiment_root",
        type=Path,
        default=Path("workspace/local_eval/benchmark_runs"),
    )
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
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
    """Return the project root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


# ── built-in method templates ─────────────────────────────────────────


def _template_builtin_specs(py: str) -> dict[str, MethodSpec]:
    """Create MethodSpec templates for the built-in AE and VAE methods.
    Placeholders (e.g. {dataset_base_dir}) are filled in by _render_command()."""

    def _ae_vae_spec(model_type: str) -> MethodSpec:
        return MethodSpec(
            name=model_type,
            train_cmd_template=[
                py,
                "workspace/local_eval/train_ae_vae.py",
                "--dataset_base_dir",
                "{dataset_base_dir}",
                "--object_name",
                "{object_name}",
                "--model_type",
                model_type,
                "--patch_size",
                "{patch_size}",
                "--batch_size",
                "{batch_size}",
                "--epochs",
                "{epochs}",
                "--patience",
                "{patience}",
                "--lr",
                "{lr}",
                "--latent_channels",
                "{latent_channels}",
                "--beta",
                "{beta}",
                "--validation_split",
                "{validation_split}",
                "--output_dir",
                "{checkpoint_dir}",
                "--seed",
                "{seed}",
                "--num_workers",
                "{num_workers}",
                "--cache_size",
                "{cache_size}",
            ],
            infer_cmd_template=[
                py,
                "workspace/local_eval/infer_ae_vae.py",
                "--dataset_base_dir",
                "{dataset_base_dir}",
                "--object_name",
                "{object_name}",
                "--checkpoint",
                "{checkpoint}",
                "--anomaly_maps_dir",
                "{anomaly_maps_dir}",
                "--batch_size",
                "{batch_size}",
            ],
        )

    return {t: _ae_vae_spec(t) for t in ("ae", "vae")}


# ── custom method loading ─────────────────────────────────────────────


def _load_custom_specs(config_path: Path | None) -> dict[str, MethodSpec]:
    """Load additional method definitions from a JSON configuration file.
    Returns an empty dict if no config is provided."""
    if config_path is None:
        return {}
    logger.info("Loading custom method specs from: %s", config_path)
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
    logger.info("Loaded %d custom method(s): %s", len(result), list(result.keys()))
    return result


# ── command rendering / execution helpers ─────────────────────────────


def _render_command(template: list[str], values: dict[str, str]) -> list[str]:
    """Replace {placeholder} tokens in a command template."""
    return [token.format_map(values) for token in template]


def _maybe_append_optional_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    """Append --image_limit / --validation_image_limit if they were set."""
    updated = list(cmd)
    if args.image_limit is not None:
        updated.extend(["--image_limit", str(args.image_limit)])
    if args.validation_image_limit is not None:
        updated.extend(["--validation_image_limit", str(args.validation_image_limit)])
    return updated


def _run_command(cmd: list[str], cwd: Path) -> None:
    """Run a subprocess, logging the command line first."""
    subprocess.run(cmd, cwd=cwd, check=True)


# ── metric reading / writing helpers ──────────────────────────────────


def _read_metrics(metrics_json: Path, object_name: str) -> dict[str, float]:
    """Read AU-PRO and AU-ROC from the evaluator's JSON output."""
    with metrics_json.open() as f:
        payload = json.load(f)
    if object_name not in payload:
        raise KeyError(
            f"Metrics file {metrics_json} does not contain object '{object_name}'."
        )
    metrics = {
        "au_pro": float(payload[object_name]["au_pro"]),
        "au_roc": float(payload[object_name]["classification_au_roc"]),
    }
    logger.info(
        "  Metrics: AU-PRO=%.4f  AU-ROC=%.4f", metrics["au_pro"], metrics["au_roc"]
    )
    return metrics


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.debug("Wrote CSV: %s (%d rows)", path, len(rows))


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-run rows → mean ± std per method."""
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


# ══════════════════════════════════════════════════════════════════════
# MAIN — wire everything together
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    # ── configure logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("=== benchmark_methods.py started ===")
    logger.info(
        "Methods: %s | Seeds: %s | Object: %s",
        args.methods,
        args.seeds,
        args.object_name,
    )

    root = repo_root()
    py = sys.executable

    # Resolve relative paths to absolute so they work in any subprocess cwd
    # (the evaluator runs from mvtec_ad_evaluation/, not the repo root).
    args.dataset_base_dir = args.dataset_base_dir.resolve()  # type: ignore[attr-defined]
    args.experiment_root = args.experiment_root.resolve()  # type: ignore[attr-defined]

    # ── build the method registry (built-in + optional custom methods) ──
    builtins = _template_builtin_specs(py=py)
    custom = _load_custom_specs(args.custom_methods_config)
    registry = {
        **builtins,
        **custom,
    }  # custom methods override built-ins with same name
    logger.info("Method registry: %s", sorted(registry.keys()))

    # Validate that every requested method is in the registry
    missing = [m for m in args.methods if m not in registry]
    if missing:
        raise ValueError(
            f"Unknown method(s): {missing}. "
            f"Available: {sorted(registry.keys())}. "
            "Use --custom_methods_config to add methods."
        )

    run_rows: list[dict[str, Any]] = []
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    total_runs = len(args.methods) * len(args.seeds)
    run_counter = 0

    # ══════════════════════════════════════════════════════════════
    # PIPELINE LOOP — for each (method × seed) combination:
    #   train → infer → evaluate → collect metrics
    # ══════════════════════════════════════════════════════════════
    overall_start = time.time()
    for method_name in args.methods:
        method = registry[method_name]
        for seed in args.seeds:
            run_counter += 1
            logger.info(
                "━━━ Run %d/%d: method=%s  seed=%d ━━━",
                run_counter,
                total_runs,
                method_name,
                seed,
            )

            # Directories for this specific run
            run_dir = (
                args.experiment_root / args.object_name / method_name / f"seed_{seed}"
            )
            checkpoint_dir = run_dir / "checkpoints"
            anomaly_maps_dir = run_dir / "anomaly_maps"
            metrics_dir = run_dir / "metrics"
            metrics_json = metrics_dir / "metrics.json"
            checkpoint = checkpoint_dir / method.checkpoint_relpath

            # Template values that get substituted into command templates
            values = {
                "dataset_base_dir": str(args.dataset_base_dir),
                "object_name": args.object_name,
                "patch_size": str(args.patch_size),
                "batch_size": str(args.batch_size),
                "epochs": str(args.epochs),
                "patience": str(args.patience),
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

            # ── STEP 1: Train ──
            if not args.skip_train:
                logger.info(
                    "[TRAIN] Starting training for %s (seed=%d)…", method_name, seed
                )
                train_cmd = _render_command(method.train_cmd_template, values)
                train_cmd = _maybe_append_optional_args(train_cmd, args)
                _run_command(train_cmd, cwd=root)
                logger.info("[TRAIN] Finished.")
            else:
                logger.info("[TRAIN] Skipped (--skip_train).")

            # ── STEP 2: Infer ──
            if not args.skip_infer:
                logger.info("[INFER] Generating anomaly maps…")
                infer_cmd = _render_command(method.infer_cmd_template, values)
                _run_command(infer_cmd, cwd=root)
                logger.info("[INFER] Finished.")
            else:
                logger.info("[INFER] Skipped (--skip_infer).")

            # ── STEP 3: Evaluate (MVTec AD2 evaluator) ──
            if not args.skip_eval:
                logger.info("[EVAL] Running MVTec AD2 evaluator…")
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
                logger.info("[EVAL] Finished.")
            else:
                logger.info("[EVAL] Skipped (--skip_eval).")

            # ── STEP 4: Collect metrics ──
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

    # ══════════════════════════════════════════════════════════════
    # SUMMARY — aggregate across seeds and write results
    # ══════════════════════════════════════════════════════════════
    summary_dir = args.experiment_root / args.object_name / "summary"
    per_run_csv = summary_dir / "per_run.csv"
    by_method_csv = summary_dir / "by_method.csv"
    by_method_json = summary_dir / "by_method.json"

    # Write per-run detail CSV
    run_fields = ["method", "seed", "au_pro", "au_roc"]
    _write_csv(per_run_csv, run_fields, run_rows)

    # Aggregate mean ± std and write summary
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

    overall_elapsed = time.time() - overall_start
    logger.info("─── Benchmark complete (%.1fs total) ───", overall_elapsed)
    logger.info("  Per-run results:   %s", per_run_csv)
    logger.info("  Aggregated CSV:    %s", by_method_csv)
    logger.info("  Aggregated JSON:   %s", by_method_json)

    # Print the summary table to stdout for quick inspection
    for row in by_method_rows:
        logger.info(
            "  %s: AU-PRO=%.4f±%.4f  AU-ROC=%.4f±%.4f  (%d runs)",
            row["method"],
            row["au_pro_mean"],
            row["au_pro_std"],
            row["au_roc_mean"],
            row["au_roc_std"],
            row["runs"],
        )


if __name__ == "__main__":
    main()
