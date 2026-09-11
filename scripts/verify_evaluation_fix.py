#!/usr/bin/env python3
"""
Before/After verification script for evaluation refactor bugs.

This script verifies two related bugs introduced during the evaluation
refactor:
    1. Batch handling bug: extract_picks_from_mask receives 3D tensors
    2. Sampling rate bug: hardcoded 2ms instead of config value

Usage:
    # Before the fix (baseline):
    python scripts/verify_evaluation_fix.py --baseline

    # After the fix (verify):
    python scripts/verify_evaluation_fix.py --verify
"""

import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SeismicConfig
from src.data.chunked_dataset import ChunkedDataManager
from src.models.loader import load_evaluation_model
from src.preprocessing.manifest import load_manifest
from src.training.metrics import (
    FirstBreakMetrics,
    SegmentationMetrics,
    extract_picks_from_mask,
)
from src.utils.logger import setup_logger

# ============================================================
# CONFIGURATION
# ============================================================

# Test cases: (config_file, model_checkpoint, split, expected_interval)
TEST_CASES = [
    {
        "name": "halfmile_2ms",
        "config": "configs/halfmile.yaml",
        "model": "models/registry/PicoUNet_Halfmile_best.pt",
        "split": "test",
        "expected_interval": 2.0,
    },
    {
        "name": "halfmile_test_1ms",
        "config": "configs/halfmile_test_1ms.yaml",
        "model": "models/registry/PicoUNet_Halfmile_best.pt",
        "split": "test",
        "expected_interval": 1.0,
    },
]


# ============================================================
# BASELINE EVALUATION (works around both bugs)
# ============================================================


def run_baseline_evaluation(
    cfg: SeismicConfig,
    model_path: str,
    split: str,
    logger,
) -> dict:
    """
    Run evaluation with a manual workaround for the batch bug.

    This is the reference implementation. After the fix, the actual
    EvaluationRunner should produce identical results.

    Workaround:
        Instead of passing the whole batch (B, H, W) to
        extract_picks_from_mask, we iterate over the batch and process
        each (H, W) mask separately.
    """
    device_obj = torch.device(cfg.device)

    # Load manifest
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)

    # Create data manager
    data_manager = ChunkedDataManager(
        chunk_dir=str(chunk_dir),
        manifest=manifest,
        cache_size=2,
        shuffle_chunks=False,
    )

    # Load model
    model_obj = load_evaluation_model(model_path, cfg, device_obj, logger)

    # Get dataset
    dataset_obj = data_manager.get_dataset(split)
    loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.device == "cuda",
    )

    # Initialize metrics
    seg_metrics = SegmentationMetrics(num_classes=3)
    fb_metrics = FirstBreakMetrics(tolerance_samples=3)

    shot_errors: list[float] = []
    shot_ids: list[int] = []

    model_obj.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device_obj)
            y = y.to(device_obj)

            outputs = model_obj(x)
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)

            # Update segmentation metrics
            seg_metrics.update(preds, y)

            # ✅ WORKAROUND: Iterate over the batch manually
            preds_np = preds.cpu().numpy()  # (B, H, W)
            y_np = y.cpu().numpy()  # (B, H, W)

            batch_size = preds_np.shape[0]

            for b in range(batch_size):
                pred_picks_b = extract_picks_from_mask(preds_np[b])  # (H,)
                true_picks_b = extract_picks_from_mask(y_np[b])  # (H,)
                fb_metrics.update(pred_picks_b, true_picks_b)

                # Collect detailed errors
                for i in range(len(pred_picks_b)):
                    if true_picks_b[i] > 0 and pred_picks_b[i] > 0:
                        error = abs(pred_picks_b[i] - true_picks_b[i])
                        shot_errors.append(float(error))

                        try:
                            global_idx = batch_idx * batch_size + b
                            shot_id = dataset_obj.get_shot_id(global_idx)
                            shot_ids.append(int(shot_id))
                        except (AttributeError, IndexError, KeyError):
                            shot_ids.append(batch_idx * batch_size + b)

    # Compute metrics
    seg_results = seg_metrics.compute()
    fb_results = fb_metrics.compute()

    # Build detailed DataFrame using the CONFIG's sampling interval
    detailed_df = pd.DataFrame(
        {
            "shot_id": shot_ids,
            "error_samples": shot_errors,
            "error_ms": np.array(shot_errors) * cfg.sampling_interval_ms,  # ✅ CORRECT
        }
    )

    return {
        "split": split,
        "segmentation": seg_results,
        "first_break": fb_results,
        "n_shots": len(dataset_obj),
        "detailed_df": detailed_df,
    }


# ============================================================
# ACTUAL EVALUATION (uses production code)
# ============================================================


def run_actual_evaluation(
    cfg: SeismicConfig,
    model_path: str,
    split: str,
    logger,
    detailed: bool = True,
) -> dict:
    """
    Run evaluation using the production EvaluationRunner.

    This will FAIL before the fix (both bugs are present).
    This should SUCCEED after the fix.
    """
    from src.evaluation import EvaluationRunner

    device_obj = torch.device(cfg.device)

    # Load manifest
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)

    # Create data manager
    data_manager = ChunkedDataManager(
        chunk_dir=str(chunk_dir),
        manifest=manifest,
        cache_size=2,
        shuffle_chunks=False,
    )

    # Load model
    model_obj = load_evaluation_model(model_path, cfg, device_obj, logger)

    # Get dataset
    dataset_obj = data_manager.get_dataset(split)
    loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.device == "cuda",
    )

    # Create runner
    runner = EvaluationRunner(
        model=model_obj,
        device_obj=device_obj,
        cfg=cfg,
        logger=logger,
        detailed=detailed,
    )

    # Run evaluation
    metrics, detailed_results = runner.evaluate_split(
        loader=loader,
        dataset_obj=dataset_obj,
        split_name=split,
    )

    # Extract per-split results
    split_data = metrics[split]
    seg_results = split_data["segmentation"]
    fb_results = split_data["first_break"]

    # Build DataFrame from detailed results
    if detailed_results and "dataframe" in detailed_results:
        detailed_df = detailed_results["dataframe"]
    else:
        detailed_df = pd.DataFrame()

    return {
        "split": split,
        "segmentation": seg_results,
        "first_break": fb_results,
        "n_shots": split_data["n_shots"],
        "detailed_df": detailed_df,
    }


# ============================================================
# COMPARISON
# ============================================================


def compare_results(baseline: dict, actual: dict, name: str) -> bool:
    """Compare baseline vs actual evaluation results."""
    print(f"\n📊 Comparing: {name}")
    print("-" * 60)

    all_match = True

    # Compare segmentation metrics
    seg_keys = ["accuracy", "mean_iou", "mean_f1"]
    print("\n  Segmentation:")
    for key in seg_keys:
        b_val = baseline["segmentation"][key]
        a_val = actual["segmentation"][key]
        match = abs(b_val - a_val) < 1e-6
        status = "✅" if match else "❌"
        print(f"    {status} {key}: baseline={b_val:.6f}, actual={a_val:.6f}")
        if not match:
            all_match = False

    # Compare first-break metrics
    fb_keys = ["mean_absolute_error", "accuracy_within_tolerance"]
    print("\n  First-Break:")
    for key in fb_keys:
        b_val = baseline["first_break"][key]
        a_val = actual["first_break"][key]
        match = abs(b_val - a_val) < 1e-6
        status = "✅" if match else "❌"
        print(f"    {status} {key}: baseline={b_val:.6f}, actual={a_val:.6f}")
        if not match:
            all_match = False

    # Compare detailed CSVs
    b_df = baseline["detailed_df"]
    a_df = actual["detailed_df"]

    if len(b_df) == 0 and len(a_df) == 0:
        print("\n  Detailed: (both empty)")
    elif len(b_df) != len(a_df):
        print(f"\n  ❌ Detailed: row count mismatch ({len(b_df)} vs {len(a_df)})")
        all_match = False
    else:
        # Compare column by column
        print(f"\n  Detailed ({len(b_df)} rows):")
        for col in b_df.columns:
            if col not in a_df.columns:
                print(f"    ❌ Column missing in actual: {col}")
                all_match = False
                continue

            if col == "shot_id":
                # Integer comparison
                match = (b_df[col].values == a_df[col].values).all()
            else:
                # Float comparison with tolerance
                match = np.allclose(b_df[col].values, a_df[col].values, rtol=1e-6)

            status = "✅" if match else "❌"
            print(f"    {status} {col}")
            if not match:
                all_match = False

    return all_match


# ============================================================
# MAIN
# ============================================================


@click.command()
@click.option(
    "--baseline",
    is_flag=True,
    help="Run baseline evaluation (workaround). Saves reference outputs.",
)
@click.option(
    "--verify",
    is_flag=True,
    help="Run actual evaluation and compare to baseline.",
)
def main(baseline: bool, verify: bool):
    """Before/After verification for evaluation fixes."""

    if not baseline and not verify:
        print("Please specify --baseline or --verify")
        print("\nUsage:")
        print("  # Before the fix:")
        print("  python scripts/verify_evaluation_fix.py --baseline")
        print()
        print("  # After the fix:")
        print("  python scripts/verify_evaluation_fix.py --verify")
        sys.exit(1)

    logger = setup_logger(task_name="verify_eval")
    output_dir = Path("evaluation_results/verification")

    if baseline:
        # ============================================
        # BASELINE MODE
        # ============================================
        print("=" * 70)
        print("🔬 BASELINE MODE (workaround)")
        print("=" * 70)
        print()
        print("This runs a REFERENCE evaluation that works around both bugs.")
        print("The output will be used to verify the actual fix later.")
        print()

        output_dir.mkdir(parents=True, exist_ok=True)

        for test_case in TEST_CASES:
            name = test_case["name"]
            print(f"\n{'=' * 70}")
            print(f"Running baseline: {name}")
            print(f"  Config: {test_case['config']}")
            print(f"  Model:  {test_case['model']}")
            print(f"  Split:  {test_case['split']}")
            print(f"  Expected interval: {test_case['expected_interval']} ms")
            print(f"{'=' * 70}\n")

            # Load config
            with open(test_case["config"]) as f:
                cfg = SeismicConfig(**yaml.safe_load(f))
            cfg.device = "mps"

            # Run baseline
            result = run_baseline_evaluation(
                cfg=cfg,
                model_path=test_case["model"],
                split=test_case["split"],
                logger=logger,
            )

            # Save results
            baseline_dir = output_dir / name / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            # Save segmentation metrics
            import json

            with open(baseline_dir / "metrics.json", "w") as f:
                json.dump(
                    {
                        "segmentation": result["segmentation"],
                        "first_break": result["first_break"],
                        "n_shots": result["n_shots"],
                    },
                    f,
                    indent=2,
                )

            # Save detailed CSV
            if len(result["detailed_df"]) > 0:
                result["detailed_df"].to_csv(baseline_dir / "detailed.csv", index=False)

            print(f"\n✅ Baseline saved to: {baseline_dir}")
            print(f"   Seg IoU: {result['segmentation']['mean_iou']:.4f}")
            print(f"   FB MAE:  {result['first_break']['mean_absolute_error']:.4f}")

            # Compute actual interval used
            df = result["detailed_df"]
            if len(df) > 0 and (df["error_samples"] > 0).any():
                df_valid = df[df["error_samples"] > 0]
                actual_interval = (
                    df_valid["error_ms"] / df_valid["error_samples"]
                ).median()
                print(f"   Interval used: {actual_interval:.4f} ms/sample")
                print(
                    f"   Expected:      {test_case['expected_interval']:.4f} ms/sample"
                )

        print()
        print("=" * 70)
        print("✅ BASELINE COMPLETE")
        print("=" * 70)
        print()
        print("Now apply the fix to src/evaluation/runner.py, then run:")
        print("  python scripts/verify_evaluation_fix.py --verify")

    else:
        # ============================================
        # VERIFY MODE
        # ============================================
        print("=" * 70)
        print("🔬 VERIFY MODE (production code)")
        print("=" * 70)
        print()
        print("This runs the ACTUAL evaluation and compares to baseline.")
        print()

        # Check baseline exists
        if not output_dir.exists():
            print(f"❌ Baseline directory not found: {output_dir}")
            print("   Run with --baseline first.")
            sys.exit(1)

        all_passed = True

        for test_case in TEST_CASES:
            name = test_case["name"]
            baseline_dir = output_dir / name / "baseline"

            if not baseline_dir.exists():
                print(f"❌ Baseline not found for {name}")
                all_passed = False
                continue

            print(f"\n{'=' * 70}")
            print(f"Verifying: {name}")
            print(f"{'=' * 70}\n")

            # Load config
            with open(test_case["config"]) as f:
                cfg = SeismicConfig(**yaml.safe_load(f))
            cfg.device = "mps"

            # Run actual evaluation
            try:
                actual = run_actual_evaluation(
                    cfg=cfg,
                    model_path=test_case["model"],
                    split=test_case["split"],
                    logger=logger,
                    detailed=True,
                )
            except Exception as e:
                print(f"❌ Evaluation failed: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                all_passed = False
                continue

            # Load baseline
            import json

            with open(baseline_dir / "metrics.json") as f:
                baseline_metrics = json.load(f)

            baseline_detailed = None
            if (baseline_dir / "detailed.csv").exists():
                baseline_detailed = pd.read_csv(baseline_dir / "detailed.csv")

            baseline = {
                "split": test_case["split"],
                "segmentation": baseline_metrics["segmentation"],
                "first_break": baseline_metrics["first_break"],
                "n_shots": baseline_metrics["n_shots"],
                "detailed_df": baseline_detailed
                if baseline_detailed is not None
                else pd.DataFrame(),
            }

            # Compare
            matches = compare_results(baseline, actual, name)

            if matches:
                print(f"\n✅ {name}: ALL CHECKS PASSED")
            else:
                print(f"\n❌ {name}: SOME CHECKS FAILED")
                all_passed = False

        print()
        print("=" * 70)
        if all_passed:
            print("✅ VERIFICATION PASSED — fix is correct")
        else:
            print("❌ VERIFICATION FAILED — fix has issues")
        print("=" * 70)


if __name__ == "__main__":
    main()
