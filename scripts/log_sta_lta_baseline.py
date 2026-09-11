#!/usr/bin/env python3
# file location: scripts/log_sta_lta_baseline.py

"""
Log the STA/LTA baseline (from the sweep) to MLflow.

Reads the most recent sweep JSON from evaluation_results/sta_lta_sweep/
and logs two runs to the seismic-fbp-baselines experiment:

    1. Winner (top-ranked by ±3 accuracy) — phase = baseline-v1.0
    2. MAE-optimal (lowest mean absolute error) — phase = baseline-v1.0-mae-optimal

Each run gets:
    - Tags: dataset, model_type=STALTA, phase, env, plus sta/lta/threshold
    - Params: sta_window, lta_window, threshold (for MLflow search)
    - Metrics: mae, median, std, within_3, predicted_picks, matched_traces
    - Artifact: the sweep JSON itself

Usage:
    python scripts/log_sta_lta_baseline.py
    python scripts/log_sta_lta_baseline.py --sweep path/to/sweep.json
    python scripts/log_sta_lta_baseline.py --dry-run
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tracking_conventions import EXPERIMENT_BASELINES

DEFAULT_SWEEP_DIR = Path("evaluation_results/sta_lta_sweep")


# ============================================================
# HELPERS
# ============================================================


def find_latest_sweep(sweep_dir: Path) -> Path:
    """Return the most recent sweep_*.json under sweep_dir."""
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
    candidates = sorted(
        sweep_dir.glob("sweep_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No sweep_*.json files found in {sweep_dir}")
    return candidates[0]


def extract_winner(payload: dict) -> dict:
    """Extract the winner config from a sweep payload."""
    winner = payload.get("winner")
    if not winner:
        raise ValueError("Sweep payload has no 'winner' key")
    return winner


def extract_mae_optimal(payload: dict) -> dict:
    """
    Find the config with the lowest MAE across all results.
    Only considers results with status == 'ok'.
    """
    results = payload.get("all_results", [])
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        raise ValueError("No 'ok' results in sweep payload")
    return min(ok_results, key=lambda r: r["mae"])


def build_run_name(label: str, dataset: str, phase: str) -> str:
    """
    Human-readable run name:
        <label>_<dataset>_<phase>_<HHMMSS>
    """
    hhmmss = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"{label}_{dataset}_{phase}_{hhmmss}"


def log_baseline_run(
    mlflow_manager,
    logger,
    result: dict,
    phase: str,
    sweep_path: Path,
    dry_run: bool = False,
) -> str | None:
    """
    Log a single baseline run. Returns the run_id, or None on dry-run.
    """
    dataset = "Halfmile"  # STA/LTA sweep was on Halfmile only
    label = "STALTA"
    run_name = build_run_name(label, dataset, phase)

    tags = {
        "dataset": dataset,
        "model_type": label,
        "phase": phase,
        "env": "research",
        "sta_window": str(result["sta_window"]),
        "lta_window": str(result["lta_window"]),
        "threshold": str(result["threshold"]),
    }

    params = {
        "sta_window": result["sta_window"],
        "lta_window": result["lta_window"],
        "threshold": result["threshold"],
    }

    metrics = {
        "mae_samples": float(result["mae"]),
        "median_samples": float(result["median"]),
        "std_samples": float(result["std"]),
        "within_3_accuracy": float(result["within_3"]),
        "predicted_picks": int(result["predicted_picks"]),
        "matched_traces": int(result["matched_traces"]),
        "total_traces": int(result["total_traces"]),
    }

    logger.info("")
    logger.info("-" * 70)
    logger.info(f"Logging run: {run_name}")
    logger.info(f"  phase:       {phase}")
    logger.info(f"  sta_window:  {result['sta_window']}")
    logger.info(f"  lta_window:  {result['lta_window']}")
    logger.info(f"  threshold:   {result['threshold']}")
    logger.info(f"  MAE:         {metrics['mae_samples']:.2f}")
    logger.info(f"  Median:      {metrics['median_samples']:.2f}")
    logger.info(f"  ±3 acc:      {metrics['within_3_accuracy']:.4f}")
    logger.info(
        f"  Predicted:   {metrics['predicted_picks']} / {metrics['total_traces']}"
    )
    logger.info("-" * 70)

    if dry_run:
        logger.info("  (dry run — not logging)")
        return None

    # Start the run with explicit name and tags
    run_id = mlflow_manager.start_run(
        config_dict=params,
        run_name=run_name,
        tags=tags,
    )

    # Log metrics at step 0
    mlflow_manager.log_metrics(metrics, step=0)

    # Log the sweep JSON as an artifact
    try:
        mlflow_manager.log_artifact(str(sweep_path), artifact_path="sweep")
        logger.info(f"  ✅ Artifact logged: {sweep_path.name}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  ⚠️  Could not log sweep artifact: {e}")

    mlflow_manager.end_run()
    logger.info(f"  ✅ Run logged: {run_id}")
    return run_id


# ============================================================
# MAIN
# ============================================================


@click.command()
@click.option(
    "--sweep",
    "sweep_path_str",
    default=None,
    help=f"Path to a sweep JSON. Default: latest under {DEFAULT_SWEEP_DIR}",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be logged without actually logging.",
)
def main(sweep_path_str: str | None, dry_run: bool) -> None:
    logger = setup_logger(task_name="log_sta_lta_baseline")

    print("=" * 80)
    print("LOG STA/LTA BASELINE TO MLFLOW")
    print("=" * 80)

    # Locate the sweep JSON
    if sweep_path_str:
        sweep_path = Path(sweep_path_str)
    else:
        sweep_path = find_latest_sweep(DEFAULT_SWEEP_DIR)

    print(f"  Sweep JSON: {sweep_path}")
    print(f"  Experiment: {EXPERIMENT_BASELINES}")

    with open(sweep_path, "r") as f:
        payload = json.load(f)

    print(f"  Sweep timestamp: {payload.get('timestamp', 'N/A')}")
    print(f"  Sweep configs:   {payload.get('n_configs', 'N/A')}")
    print(f"  Sweep shots:     {payload.get('n_shots', 'N/A')}")

    # Extract the two configs we care about
    winner = extract_winner(payload)
    mae_optimal = extract_mae_optimal(payload)

    # Check for duplication: is the winner also the MAE-optimal?
    same_config = (
        winner["sta_window"] == mae_optimal["sta_window"]
        and winner["lta_window"] == mae_optimal["lta_window"]
        and winner["threshold"] == mae_optimal["threshold"]
    )

    print()
    print("  Winner (by ±3 accuracy):")
    print(
        f"    sta={winner['sta_window']}, lta={winner['lta_window']}, "
        f"thr={winner['threshold']}"
    )
    print(f"    MAE={winner['mae']:.2f}, ±3={winner['within_3'] * 100:.1f}%")

    print()
    print("  MAE-optimal:")
    print(
        f"    sta={mae_optimal['sta_window']}, lta={mae_optimal['lta_window']}, "
        f"thr={mae_optimal['threshold']}"
    )
    print(f"    MAE={mae_optimal['mae']:.2f}, ±3={mae_optimal['within_3'] * 100:.1f}%")

    if same_config:
        print()
        print("  ℹ️  Winner and MAE-optimal are the same config — logging once.")

    # Get the MLflow manager for the baselines experiment
    mlflow_manager = get_mlflow_manager(
        experiment_name=EXPERIMENT_BASELINES,
        enable_system_metrics=False,  # no GPU monitoring for a picker
        enable_autolog=False,  # no PyTorch autolog for a picker
    )

    # Log the winner
    winner_run_id = log_baseline_run(
        mlflow_manager=mlflow_manager,
        logger=logger,
        result=winner,
        phase="baseline-v1.0",
        sweep_path=sweep_path,
        dry_run=dry_run,
    )

    # Log the MAE-optimal (if distinct)
    mae_run_id = None
    if not same_config:
        mae_run_id = log_baseline_run(
            mlflow_manager=mlflow_manager,
            logger=logger,
            result=mae_optimal,
            phase="baseline-v1.0-mae-optimal",
            sweep_path=sweep_path,
            dry_run=dry_run,
        )

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if dry_run:
        print("  DRY RUN — nothing was logged.")
    else:
        if winner_run_id:
            print(f"  Winner run:      {winner_run_id}")
        if mae_run_id:
            print(f"  MAE-optimal run: {mae_run_id}")
        print()
        print(f"  Experiment: {EXPERIMENT_BASELINES}")
        print("  View:       mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print()


if __name__ == "__main__":
    main()
