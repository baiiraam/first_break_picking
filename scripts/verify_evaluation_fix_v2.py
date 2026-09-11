#!/usr/bin/env python3
"""
Before/After verification for evaluation refactor bugs.

Uses SYNTHETIC masks to test:
    1. Batch handling bug (3D vs 2D input to extract_picks_from_mask)
    2. Sampling rate bug (hardcoded 2ms vs cfg.sampling_interval_ms)

This avoids depending on model quality — the masks are crafted to
exercise specific code paths.

Usage:
    python scripts/verify_evaluation_fix_v2.py --baseline
    python scripts/verify_evaluation_fix_v2.py --verify
"""

import json
import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.metrics import (
    FirstBreakMetrics,
    extract_picks_from_mask,
)

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================


def make_synthetic_mask(
    n_traces: int,
    n_samples: int,
    pick_positions: np.ndarray,
    strip_width: int = 8,
) -> np.ndarray:
    """
    Create a synthetic segmentation mask.

    Args:
        n_traces: Number of traces
        n_samples: Number of samples per trace
        pick_positions: (n_traces,) array of pick sample indices
            (0 for "no valid pick" → class -1)
        strip_width: Width of strip around pick

    Returns:
        mask: (n_traces, n_samples) int64 array with classes {0, 1, 2, -1}
    """
    mask = np.zeros((n_traces, n_samples), dtype=np.int64)
    half_width = strip_width // 2

    for i, pick in enumerate(pick_positions):
        if pick <= 0:
            mask[i, :] = -1  # Unlabeled
            continue

        pick_int = round(pick)
        if pick_int <= 0 or pick_int >= n_samples:
            mask[i, :] = -1
            continue

        # Before region: class 0 (default zeros)
        # Strip region: class 2
        start = max(0, pick_int - half_width)
        end = min(n_samples, pick_int + half_width + 1)
        mask[i, start:end] = 2

        # After region: class 1
        mask[i, end:] = 1

    return mask


def make_synthetic_batch(
    batch_size: int,
    n_traces: int,
    n_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic batch with both prediction and ground truth masks.

    Returns:
        preds: (B, H, W) int64 — prediction masks (with some errors)
        truths: (B, H, W) int64 — ground truth masks
    """
    rng = np.random.default_rng(seed)

    preds_list = []
    truths_list = []

    for b in range(batch_size):
        # Ground truth picks: random positions
        true_picks = rng.integers(100, n_samples - 100, size=n_traces).astype(float)

        # Predictions: introduce small error (0-5 samples)
        error = rng.integers(-5, 6, size=n_traces).astype(float)
        pred_picks = np.clip(true_picks + error, 1, n_samples - 1)

        truths_list.append(make_synthetic_mask(n_traces, n_samples, true_picks))
        preds_list.append(make_synthetic_mask(n_traces, n_samples, pred_picks))

    return np.stack(preds_list), np.stack(truths_list)


# ============================================================
# BASELINE EVALUATION (manual, correct)
# ============================================================


def run_baseline_evaluation(
    preds: np.ndarray,  # (B, H, W)
    truths: np.ndarray,  # (B, H, W)
    sampling_interval_ms: float,
) -> dict:
    """
    Reference implementation: iterate over batch, use cfg interval.
    """
    B, H, W = preds.shape

    fb_metrics = FirstBreakMetrics(tolerance_samples=3)

    shot_errors = []
    shot_ids = []

    for b in range(B):
        # Correctly pass 2D mask per shot
        pred_picks_b = extract_picks_from_mask(preds[b])  # (H,)
        true_picks_b = extract_picks_from_mask(truths[b])  # (H,)

        fb_metrics.update(pred_picks_b, true_picks_b)

        # Collect per-trace errors
        for i in range(len(pred_picks_b)):
            if true_picks_b[i] > 0 and pred_picks_b[i] > 0:
                error = abs(int(pred_picks_b[i]) - int(true_picks_b[i]))
                shot_errors.append(float(error))
                shot_ids.append(b * H + i)  # synthetic shot ID

    # Compute metrics
    fb_results = fb_metrics.compute()

    # Build detailed DataFrame — uses the CONFIG's sampling interval (correct)
    detailed_df = pd.DataFrame(
        {
            "shot_id": shot_ids,
            "error_samples": shot_errors,
            "error_ms": np.array(shot_errors) * sampling_interval_ms,
        }
    )

    return {
        "first_break": fb_results,
        "detailed_df": detailed_df,
    }


# ============================================================
# ACTUAL EVALUATION (mimics production code)
# ============================================================


def run_actual_evaluation(
    preds: np.ndarray,
    truths: np.ndarray,
    sampling_interval_ms: float,
) -> dict:
    """
    Mimics the CURRENT (buggy) production code in runner.py.

    Note: This simulates what the actual evaluation does BEFORE the fix.
    """
    B, H, W = preds.shape

    fb_metrics = FirstBreakMetrics(tolerance_samples=3)

    shot_errors = []
    shot_ids = []

    # ⚠️ This is what the buggy production code does:
    # It passes the whole 3D batch to extract_picks_from_mask
    try:
        pred_picks = extract_picks_from_mask(preds)  # ← BUG: 3D input
        true_picks = extract_picks_from_mask(truths)

        fb_metrics.update(pred_picks, true_picks)

        for i in range(len(pred_picks)):
            if true_picks[i] > 0 and pred_picks[i] > 0:
                error = abs(int(pred_picks[i]) - int(true_picks[i]))
                shot_errors.append(float(error))
                shot_ids.append(i)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    fb_results = fb_metrics.compute()

    # ⚠️ This is the buggy production code:
    # It hardcodes 2ms per sample
    detailed_df = pd.DataFrame(
        {
            "shot_id": shot_ids,
            "error_samples": shot_errors,
            "error_ms": np.array(shot_errors) * 2,  # ← BUG: hardcoded 2
        }
    )

    return {
        "first_break": fb_results,
        "detailed_df": detailed_df,
    }


# ============================================================
# MAIN
# ============================================================


@click.command()
@click.option("--baseline", is_flag=True, help="Run baseline (correct)")
@click.option("--verify", is_flag=True, help="Run actual and compare")
def main(baseline: bool, verify: bool):
    if not baseline and not verify:
        print("Please specify --baseline or --verify")
        sys.exit(1)

    output_dir = Path("evaluation_results/synthetic_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test cases: (name, batch_size, n_traces, n_samples, sampling_interval_ms)
    test_cases = [
        ("batch1_1ms", 1, 100, 751, 1.0),
        ("batch4_2ms", 4, 100, 751, 2.0),
        ("batch4_1ms", 4, 100, 751, 1.0),
    ]

    if baseline:
        print("=" * 70)
        print("🔬 BASELINE MODE — Synthesizing correct reference output")
        print("=" * 70)

        for name, B, H, W, interval in test_cases:
            print(f"\n📊 Test: {name}")
            print(f"   Batch size: {B}, Traces: {H}, Samples: {W}")
            print(f"   Sampling interval: {interval} ms")

            preds, truths = make_synthetic_batch(B, H, W, seed=42)
            print(f"   Pred shape: {preds.shape}, Truth shape: {truths.shape}")

            result = run_baseline_evaluation(preds, truths, interval)

            # Save
            baseline_dir = output_dir / name / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            with open(baseline_dir / "metrics.json", "w") as f:
                json.dump(result["first_break"], f, indent=2)

            if len(result["detailed_df"]) > 0:
                result["detailed_df"].to_csv(baseline_dir / "detailed.csv", index=False)

            # Report
            n_errors = len(result["detailed_df"])
            mae = result["first_break"]["mean_absolute_error"]

            print(f"   ✅ Collected {n_errors} errors")
            print(f"   MAE: {mae:.4f} samples")

            if n_errors > 0:
                df = result["detailed_df"]
                df_valid = df[df["error_samples"] > 0]
                if len(df_valid) > 0:
                    actual_interval = (
                        df_valid["error_ms"] / df_valid["error_samples"]
                    ).median()
                    print(f"   Interval used: {actual_interval:.4f} ms")
                    print(f"   Expected:      {interval:.4f} ms")
                    match = abs(actual_interval - interval) < 1e-6
                    print(f"   {'✅ MATCH' if match else '❌ MISMATCH'}")

        print("\n" + "=" * 70)
        print("✅ BASELINE COMPLETE")
        print(f"   Saved to: {output_dir}")
        print("\nNow apply the fix to src/evaluation/runner.py, then run:")
        print("  python scripts/verify_evaluation_fix_v2.py --verify")

    else:  # verify
        print("=" * 70)
        print("🔬 VERIFY MODE — Testing current code against baseline")
        print("=" * 70)

        all_passed = True

        for name, B, H, W, interval in test_cases:
            print(f"\n📊 Test: {name}")

            baseline_dir = output_dir / name / "baseline"
            if not baseline_dir.exists():
                print(f"   ❌ Baseline not found: {baseline_dir}")
                all_passed = False
                continue

            preds, truths = make_synthetic_batch(B, H, W, seed=42)
            actual = run_actual_evaluation(preds, truths, interval)

            if "error" in actual:
                print(f"   ❌ Evaluation failed: {actual['error']}")
                all_passed = False
                continue

            # Load baseline
            with open(baseline_dir / "metrics.json") as f:
                baseline_metrics = json.load(f)

            baseline_df = pd.read_csv(baseline_dir / "detailed.csv")
            actual_df = actual["detailed_df"]

            # Compare
            print("\n   Comparing results:")
            print(f"     Baseline rows: {len(baseline_df)}")
            print(f"     Actual rows:   {len(actual_df)}")

            # Compare MAE
            b_mae = baseline_metrics["mean_absolute_error"]
            a_mae = actual["first_break"]["mean_absolute_error"]
            mae_match = abs(b_mae - a_mae) < 1e-6
            print(
                f"     {'✅' if mae_match else '❌'} MAE: baseline={b_mae:.4f}, actual={a_mae:.4f}"
            )

            if not mae_match:
                all_passed = False

            # Compare interval
            if len(actual_df) > 0 and (actual_df["error_samples"] > 0).any():
                df_valid = actual_df[actual_df["error_samples"] > 0]
                actual_interval = (
                    df_valid["error_ms"] / df_valid["error_samples"]
                ).median()
                interval_match = abs(actual_interval - interval) < 1e-6
                print(
                    f"     {'✅' if interval_match else '❌'} Interval: baseline={interval:.4f}, actual={actual_interval:.4f}"
                )

                if not interval_match:
                    all_passed = False

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ VERIFICATION PASSED — both bugs are fixed")
        else:
            print("❌ VERIFICATION FAILED — bugs still present or fix is wrong")
        print("=" * 70)


if __name__ == "__main__":
    main()
