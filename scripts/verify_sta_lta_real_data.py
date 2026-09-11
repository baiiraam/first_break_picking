#!/usr/bin/env python3
# file location: scripts/verify_sta_lta_real_data.py

"""
Run STA/LTA on real Halfmile shots and report accuracy.

This is the scientific check — the synthetic script only proves the
algorithm is implemented correctly. This one shows what the picker
actually achieves on the data we care about.

Read-only. Does not write to MLflow or save any files. The point is
to see a number, not to archive it (that's B.2's job).

Usage:
    python scripts/verify_sta_lta_real_data.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import STALTAPicker
from src.training.metrics import FirstBreakMetrics
from src.utils.hdf5_utils import load_shot_data, load_shot_indices


# ============================================================
# CONFIG
# ============================================================

HDF5_PATH = "data/raw/Halfmile3D_add_geom_sorted.hdf5"
TARGET_TRACES = 1578
N_SAMPLES = 751
SAMPLING_INTERVAL_MS = 2.0

N_SHOTS = 3          # how many shots to evaluate
PICKER_PARAMS = dict(sta_window=20, lta_window=100, threshold=3.0)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("=" * 80)
    print("VERIFY STA/LTA PICKER — REAL HALFMILE DATA (B.1)")
    print("=" * 80)
    print(f"  HDF5:        {HDF5_PATH}")
    print(f"  Shots:       first {N_SHOTS}")
    print(f"  Picker:      STALTAPicker({PICKER_PARAMS})")
    print(f"  Sampling:    {SAMPLING_INTERVAL_MS} ms/sample")

    if not os.path.exists(HDF5_PATH):
        print(f"\n❌ HDF5 not found: {HDF5_PATH}")
        return 1

    # ---- Load shot indices ----
    print("\nLoading shot indices...")
    unique_shots, start_indices, end_indices = load_shot_indices(HDF5_PATH)
    print(f"  Found {len(unique_shots)} shots")

    # ---- Instantiate picker ----
    picker = STALTAPicker(**PICKER_PARAMS)

    # ---- Metrics accumulator ----
    fb_metrics = FirstBreakMetrics(tolerance_samples=3)

    per_shot_results: list[dict] = []

    # ---- Iterate over the first N_SHOTS ----
    for shot_idx in range(min(N_SHOTS, len(unique_shots))):
        shot_id = int(unique_shots[shot_idx])
        start = int(start_indices[shot_idx])
        end = int(end_indices[shot_idx])

        print()
        print("-" * 80)
        print(f"Shot {shot_idx}: SHOTID={shot_id}  rows={start}..{end}")
        print("-" * 80)

        # Load raw shot data and ground-truth picks (in ms)
        shot_data, shot_picks_ms = load_shot_data(
            HDF5_PATH, start, end, TARGET_TRACES, N_SAMPLES
        )

        # Convert ground-truth picks from ms → samples
        shot_picks_true = np.round(shot_picks_ms / SAMPLING_INTERVAL_MS).astype(np.int64)

        # Run the picker
        pred_picks = picker.pick(shot_data)

        # Restrict comparison to traces with valid ground-truth picks
        valid_gt_mask = (shot_picks_true > 0) & (shot_picks_true < N_SAMPLES)
        n_valid_gt = int(valid_gt_mask.sum())

        # Update metrics (FirstBreakMetrics handles the valid_mask internally)
        fb_metrics.update(pred_picks, shot_picks_true)

        # Local per-shot stats
        valid_pred_mask = pred_picks > 0
        n_pred = int(valid_pred_mask.sum())

        # Signed errors on traces with both true and predicted picks
        both_mask = valid_gt_mask & valid_pred_mask
        if both_mask.sum() > 0:
            signed_errors = (
                pred_picks[both_mask] - shot_picks_true[both_mask]
            ).astype(float)
            mean_signed = float(signed_errors.mean())
            mae_shot = float(np.abs(signed_errors).mean())
            frac_correct = float(
                (np.abs(signed_errors) <= 3).mean()
            )
        else:
            mean_signed = float("nan")
            mae_shot = float("nan")
            frac_correct = float("nan")

        per_shot_results.append({
            "shot_idx": shot_idx,
            "shot_id": shot_id,
            "n_traces": int(shot_data.shape[0]),
            "n_valid_gt": n_valid_gt,
            "n_pred": n_pred,
            "mean_signed": mean_signed,
            "mae_shot": mae_shot,
            "frac_correct": frac_correct,
        })

        print(f"  Traces:              {shot_data.shape[0]}")
        print(f"  Valid ground truth:  {n_valid_gt}")
        print(f"  Predicted picks:     {n_pred} ({100 * n_pred / shot_data.shape[0]:.1f}%)")
        if both_mask.sum() > 0:
            print(f"  Mean signed error:   {mean_signed:+.2f} samples")
            print(f"  MAE (this shot):     {mae_shot:.2f} samples")
            print(f"  Within ±3 samples:   {100 * frac_correct:.1f}%")
        else:
            print("  No traces with both true and predicted picks.")

    # ---- Aggregate metrics ----
    print()
    print("=" * 80)
    print("AGGREGATE METRICS ACROSS ALL EVALUATED SHOTS")
    print("=" * 80)

    metrics = fb_metrics.compute()

    print(f"  Total traces evaluated (with valid GT + pred): "
          f"{metrics['total_traces']}")
    print(f"  Mean absolute error:   {metrics['mean_absolute_error']:.2f} samples")
    print(f"  Median absolute error: {metrics['median_absolute_error']:.2f} samples")
    print(f"  Std absolute error:    {metrics['std_absolute_error']:.2f} samples")
    print(f"  Max absolute error:    {metrics['max_absolute_error']:.2f} samples")
    print(f"  Within ±3 samples:     "
          f"{100 * metrics['accuracy_within_tolerance']:.1f}%")

    # ---- Interpretation ----
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    mae = metrics["mean_absolute_error"]
    if metrics["total_traces"] == 0:
        print("  ❌ No valid comparisons were made. Inspect the per-shot output.")
        return 1
    elif mae < 5:
        verdict = "excellent for a classical method"
    elif mae < 15:
        verdict = "reasonable for a classical method"
    elif mae < 40:
        verdict = "mediocre; parameter tuning may help (see B.2)"
    else:
        verdict = "poor; likely needs a different method or a bug is present"

    print(f"  Real-data MAE is {mae:.2f} samples — {verdict}.")
    print()
    print("  This is a single-parameter run with the default settings.")
    print("  B.2 will sweep parameters to find the best STA/LTA config.")
    print("  Whatever number B.2 produces becomes the classical baseline.")

    # ---- Pass/fail ----
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Sanity checks: picker runs, produces picks, produces finite metrics.
    sanity_ok = (
        metrics["total_traces"] > 0
        and np.isfinite(metrics["mean_absolute_error"])
        and metrics["mean_absolute_error"] < 500  # very loose upper bound
    )

    if sanity_ok:
        print("  ✅ STA/LTA ran on real data and produced sensible metrics.")
        print("  ✅ B.1 real-data verification complete.")
        return 0
    else:
        print("  ❌ STA/LTA produced unexpected results. Inspect above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())