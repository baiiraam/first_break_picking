#!/usr/bin/env python3
# file location: scripts/sweep_sta_lta.py

"""
STA/LTA parameter sweep on real Halfmile data.

Runs a grid of (sta_window, lta_window, threshold) combinations
against the first N Halfmile shots and reports metrics for each.

Primary metric:   ±3 sample accuracy (higher is better)
Tie-breaker:      MAE (lower is better)

The winning config is the classical baseline for the project.

Usage:
    python scripts/sweep_sta_lta.py
    python scripts/sweep_sta_lta.py --n-shots 3          # faster
    python scripts/sweep_sta_lta.py --output path.json   # custom output

Output:
    - Printed table (full grid + top-5 + winner)
    - JSON file at evaluation_results/sta_lta_sweep/sweep_<timestamp>.json
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import click
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import STALTAPicker
from src.training.metrics import FirstBreakMetrics
from src.utils.hdf5_utils import load_shot_data, load_shot_indices

# ============================================================
# SWEEP GRID
# ============================================================

STA_WINDOWS = [10, 15, 20, 25, 30]
LTA_WINDOWS = [60, 80, 100, 120, 150]
THRESHOLDS = [2.0, 2.5, 3.0, 4.0, 5.0]

# ============================================================
# DATA SOURCE
# ============================================================

HDF5_PATH = "data/raw/Halfmile3D_add_geom_sorted.hdf5"
TARGET_TRACES = 1578
N_SAMPLES = 751
SAMPLING_INTERVAL_MS = 2.0

DEFAULT_N_SHOTS = 5

# ============================================================
# OUTPUT
# ============================================================

DEFAULT_OUTPUT_DIR = "evaluation_results/sta_lta_sweep"


# ============================================================
# HELPERS
# ============================================================


def load_shots(n_shots: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load the first N shots from Halfmile HDF5."""
    if not os.path.exists(HDF5_PATH):
        raise FileNotFoundError(f"HDF5 not found: {HDF5_PATH}")

    unique_shots, start_indices, end_indices = load_shot_indices(HDF5_PATH)
    n_available = len(unique_shots)
    n_to_load = min(n_shots, n_available)

    shots: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_to_load):
        start = int(start_indices[i])
        end = int(end_indices[i])
        shot_data, shot_picks_ms = load_shot_data(
            HDF5_PATH, start, end, TARGET_TRACES, N_SAMPLES
        )
        shot_picks_samples = np.round(shot_picks_ms / SAMPLING_INTERVAL_MS).astype(
            np.int64
        )
        shots.append((shot_data, shot_picks_samples))

    return shots


def evaluate_config(
    sta_window: int,
    lta_window: int,
    threshold: float,
    shots: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """
    Run a single STA/LTA config over a preloaded list of shots.

    Returns a metrics dict. On error, returns a dict with
    status="error" and a message.
    """
    try:
        picker = STALTAPicker(
            sta_window=sta_window,
            lta_window=lta_window,
            threshold=threshold,
        )
    except ValueError as e:
        return {
            "status": "error",
            "error": str(e),
            "sta_window": sta_window,
            "lta_window": lta_window,
            "threshold": threshold,
        }

    fb_metrics = FirstBreakMetrics(tolerance_samples=3)
    total_predicted = 0
    total_traces = 0

    for shot_data, shot_picks_samples in shots:
        pred_picks = picker.pick(shot_data)
        fb_metrics.update(pred_picks, shot_picks_samples)
        total_predicted += int(np.count_nonzero(pred_picks))
        total_traces += shot_data.shape[0]

    metrics = fb_metrics.compute()

    if metrics["total_traces"] == 0:
        return {
            "status": "no_picks",
            "sta_window": sta_window,
            "lta_window": lta_window,
            "threshold": threshold,
            "total_traces": total_traces,
            "predicted_picks": total_predicted,
            "matched_traces": 0,
        }

    return {
        "status": "ok",
        "sta_window": sta_window,
        "lta_window": lta_window,
        "threshold": threshold,
        "total_traces": total_traces,
        "predicted_picks": total_predicted,
        "matched_traces": metrics["total_traces"],
        "mae": metrics["mean_absolute_error"],
        "median": metrics["median_absolute_error"],
        "std": metrics["std_absolute_error"],
        "within_3": metrics["accuracy_within_tolerance"],
    }


def sort_key(r: dict[str, Any]) -> tuple[float, float, int]:
    """
    Sort by ±3 accuracy descending, then MAE ascending, then
    predicted-picks descending (as a mild tiebreak).
    Rows with status != "ok" go to the bottom.
    """
    if r["status"] != "ok":
        return (float("inf"), float("inf"), 0)
    return (-r["within_3"], r["mae"], -r["predicted_picks"])


def print_table(rows: list[dict[str, Any]], title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(
        f"  {'Rank':<5} {'STA':<5} {'LTA':<5} {'Thr':<5} "
        f"{'Predicted':<11} {'Matched':<9} "
        f"{'MAE':<9} {'Median':<9} {'±3 acc':<9}"
    )
    print("  " + "-" * 90)
    for i, r in enumerate(rows, start=1):
        if r["status"] != "ok":
            print(
                f"  {i:<5} {r['sta_window']:<5} {r['lta_window']:<5} "
                f"{r['threshold']:<5} — {r['status']}"
            )
            continue
        print(
            f"  {i:<5} {r['sta_window']:<5} {r['lta_window']:<5} "
            f"{r['threshold']:<5} "
            f"{r['predicted_picks']:<11} {r['matched_traces']:<9} "
            f"{r['mae']:<9.2f} {r['median']:<9.2f} "
            f"{100 * r['within_3']:<8.1f}%"
        )


# ============================================================
# MAIN
# ============================================================


@click.command()
@click.option(
    "--n-shots",
    type=int,
    default=DEFAULT_N_SHOTS,
    help=f"Number of Halfmile shots to evaluate (default: {DEFAULT_N_SHOTS})",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help=f"Output JSON path. Default: {DEFAULT_OUTPUT_DIR}/sweep_<timestamp>.json",
)
@click.option(
    "--top-n",
    type=int,
    default=20,
    help="How many top configs to show in the ranked table (default: 20)",
)
def main(n_shots: int, output: str | None, top_n: int) -> None:
    print("=" * 100)
    print("STA/LTA PARAMETER SWEEP — HALFMILE")
    print("=" * 100)

    # ---- Build grid ----
    grid = list(product(STA_WINDOWS, LTA_WINDOWS, THRESHOLDS))
    # Filter out invalid combos (sta >= lta) before running
    valid_grid = [(s, l, t) for (s, l, t) in grid if s < l]
    n_invalid = len(grid) - len(valid_grid)
    n_configs = len(valid_grid)

    print(f"  HDF5:             {HDF5_PATH}")
    print(f"  Shots:            first {n_shots}")
    print(f"  sta_window grid:  {STA_WINDOWS}")
    print(f"  lta_window grid:  {LTA_WINDOWS}")
    print(f"  threshold grid:   {THRESHOLDS}")
    print(f"  Raw grid:         {len(grid)} combos")
    print(f"  Invalid (sta≥lta):{n_invalid} combos skipped")
    print(f"  Valid configs:    {n_configs}")

    # ---- Load data once ----
    print(f"\nLoading {n_shots} shots...")
    t0 = time.time()
    shots = load_shots(n_shots)
    total_traces = sum(s[0].shape[0] for s in shots)
    print(
        f"  Loaded {len(shots)} shots, {total_traces} traces in {time.time() - t0:.1f}s"
    )

    # ---- Run sweep ----
    print(f"\nRunning {n_configs} configs...")
    t0 = time.time()
    results: list[dict[str, Any]] = []
    for i, (sta, lta, thr) in enumerate(valid_grid, start=1):
        if i % 25 == 0 or i == 1 or i == n_configs:
            print(f"  [{i:>3}/{n_configs}] sta={sta}, lta={lta}, thr={thr}")
        results.append(evaluate_config(sta, lta, thr, shots))
    elapsed = time.time() - t0
    print(f"  Sweep complete in {elapsed:.1f}s ({elapsed / n_configs:.2f}s per config)")

    # ---- Sort and report ----
    ok_results = [r for r in results if r["status"] == "ok"]
    other_results = [r for r in results if r["status"] != "ok"]

    if not ok_results:
        print("\n❌ No configs produced valid picks. Cannot rank.")
        sys.exit(1)

    ok_results.sort(key=sort_key)
    ranked = ok_results + other_results

    # Top-N table
    top_rows = ranked[: min(top_n, len(ok_results))]
    print_table(top_rows, f"TOP {len(top_rows)} CONFIGS (by ±3 accuracy)")

    # Top-5 side by side
    print()
    print("=" * 100)
    print("TOP 5 CONFIGS — SIDE BY SIDE")
    print("=" * 100)
    print(
        f"  {'Rank':<5} {'STA':<5} {'LTA':<5} {'Thr':<5} "
        f"{'Predicted':<11} {'MAE':<9} {'Median':<9} {'±3 acc':<9}"
    )
    print("  " + "-" * 70)
    for i, r in enumerate(ok_results[:5], start=1):
        print(
            f"  {i:<5} {r['sta_window']:<5} {r['lta_window']:<5} "
            f"{r['threshold']:<5} "
            f"{r['predicted_picks']:<11} {r['mae']:<9.2f} "
            f"{r['median']:<9.2f} {100 * r['within_3']:<8.1f}%"
        )

    # ---- Winner ----
    winner = ok_results[0]
    print()
    print("=" * 100)
    print("🏆 WINNING CONFIG — CLASSICAL BASELINE")
    print("=" * 100)
    print(f"  sta_window:       {winner['sta_window']}")
    print(f"  lta_window:       {winner['lta_window']}")
    print(f"  threshold:        {winner['threshold']}")
    print()
    print(
        f"  Predicted picks:  {winner['predicted_picks']} / {winner['total_traces']} traces "
        f"({100 * winner['predicted_picks'] / winner['total_traces']:.1f}%)"
    )
    print(f"  Matched traces:   {winner['matched_traces']}")
    print(f"  MAE:              {winner['mae']:.2f} samples")
    print(f"  Median:           {winner['median']:.2f} samples")
    print(f"  Std:              {winner['std']:.2f} samples")
    print(f"  ±3 accuracy:      {100 * winner['within_3']:.1f}%")
    print()
    print("  These numbers are the classical baseline for Halfmile.")

    # ---- Save JSON ----
    if output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"sweep_{ts}.json"
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hdf5_path": HDF5_PATH,
        "n_shots": len(shots),
        "total_traces": total_traces,
        "grid": {
            "sta_windows": STA_WINDOWS,
            "lta_windows": LTA_WINDOWS,
            "thresholds": THRESHOLDS,
        },
        "n_configs": n_configs,
        "elapsed_seconds": round(elapsed, 2),
        "primary_metric": "within_3",
        "tiebreaker_metric": "mae",
        "winner": winner,
        "all_results": ranked,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"💾 Full sweep saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
