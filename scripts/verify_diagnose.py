#!/usr/bin/env python3
# file location: scripts/verify_diagnose.py

"""
Verify F.2 diagnostics (coherence + gallery).

Usage:
    python scripts/verify_diagnose.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainability import CoherenceAnalyzer, ErrorGalleryGenerator
from src.utils.logger import setup_logger


# ============================================================
# HELPERS
# ============================================================

def make_per_trace(
    shot_ids: list[int],
    n_traces: int,
    gt_pattern: str = "smooth",
    ml_pattern: str = "smooth",
    sta_pattern: str = "noisy",
) -> pd.DataFrame:
    """Build a synthetic per-trace DataFrame."""
    rows = []
    for sid in shot_ids:
        for t in range(n_traces):
            gt = 100 + t  # smooth wavefront
            if ml_pattern == "smooth":
                ml = gt
            elif ml_pattern == "zigzag":
                ml = gt + (10 if t % 2 == 0 else -10)
            else:
                ml = gt
            if sta_pattern == "noisy":
                sta = gt + (5 if t % 3 == 0 else -3)
            else:
                sta = gt
            rows.append({
                "shot_id": sid,
                "trace_index": t,
                "gt_pick_sample": gt,
                "sta_lta_pick": sta,
                "sta_lta_error_samples": abs(sta - gt),
                "ml_pick": ml,
                "ml_error_samples": abs(ml - gt),
            })
    return pd.DataFrame(rows)


# ============================================================
# TESTS
# ============================================================

def test_coherence_smooth() -> bool:
    print()
    print("TEST 1 — Coherence on smooth picks is low")
    logger = setup_logger(task_name="verify_diagnose")
    ca = CoherenceAnalyzer(logger=logger)
    df = make_per_trace([1], n_traces=50, ml_pattern="smooth")
    metrics = ca.compute_metrics(df)

    ml_coh = float(metrics.iloc[0]["ml_coherence"])
    print(f"  Smooth ML coherence: {ml_coh}")
    ok = ml_coh < 1.0
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} (need < 1.0)")
    return ok


def test_coherence_zigzag() -> bool:
    print()
    print("TEST 2 — Coherence on zigzag picks is high")
    logger = setup_logger(task_name="verify_diagnose")
    ca = CoherenceAnalyzer(logger=logger)
    df = make_per_trace([1], n_traces=50, ml_pattern="zigzag")
    metrics = ca.compute_metrics(df)

    ml_coh = float(metrics.iloc[0]["ml_coherence"])
    print(f"  Zigzag ML coherence: {ml_coh}")
    ok = ml_coh > 10.0
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} (need > 10.0)")
    return ok


def test_coherence_ratio() -> bool:
    print()
    print("TEST 3 — ML/GT ratio is sensible")
    logger = setup_logger(task_name="verify_diagnose")
    ca = CoherenceAnalyzer(logger=logger)
    df_smooth = make_per_trace([1], n_traces=50, ml_pattern="smooth")
    df_zigzag = make_per_trace([1], n_traces=50, ml_pattern="zigzag")

    m_smooth = ca.compute_metrics(df_smooth)
    m_zigzag = ca.compute_metrics(df_zigzag)

    ratio_smooth = float(m_smooth.iloc[0]["ml_over_gt_ratio"])
    ratio_zigzag = float(m_zigzag.iloc[0]["ml_over_gt_ratio"])

    print(f"  Smooth ratio: {ratio_smooth}")
    print(f"  Zigzag ratio: {ratio_zigzag}")

    ok = ratio_zigzag > ratio_smooth
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} "
          f"(zigzag ratio should exceed smooth)")
    return ok


def test_coherence_edge_cases() -> bool:
    print()
    print("TEST 4 — Coherence handles edge cases")

    logger = setup_logger(task_name="verify_diagnose")
    ca = CoherenceAnalyzer(logger=logger)

    # All-zeros mask -> no valid picks
    df_empty = pd.DataFrame({
        "shot_id": [1, 1, 1],
        "trace_index": [0, 1, 2],
        "gt_pick_sample": [0, 0, 0],
        "sta_lta_pick": [0, 0, 0],
        "sta_lta_error_samples": [-1, -1, -1],
        "ml_pick": [0, 0, 0],
        "ml_error_samples": [-1, -1, -1],
    })

    metrics = ca.compute_metrics(df_empty)
    ml_coh = float(metrics.iloc[0]["ml_coherence"])
    ok = np.isinf(ml_coh)
    print(f"  All-zeros ML coherence: {ml_coh} (expected inf)")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_gallery_generation() -> bool:
    print()
    print("TEST 5 — Gallery generation produces a valid PNG")

    from pathlib import Path
    import tempfile

    logger = setup_logger(task_name="verify_diagnose")

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ErrorGalleryGenerator(output_dir=Path(tmpdir), logger=logger)
        df = make_per_trace([1, 2], n_traces=50)
        shots = {
            1: np.random.randn(50, 200).astype(np.float32),
            2: np.random.randn(50, 200).astype(np.float32),
        }

        path = gen.generate_gallery(
            per_trace=df,
            shots=shots,
            method="ml",
            n_worst=10,
            output_filename="test_gallery.png",
        )
        ok = path.exists() and path.stat().st_size > 1024
        print(f"  Output: {path.name}, size: {path.stat().st_size} bytes")
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
        return ok


def test_metrics_columns() -> bool:
    print()
    print("TEST 6 — Metrics DataFrame has expected columns")

    logger = setup_logger(task_name="verify_diagnose")
    ca = CoherenceAnalyzer(logger=logger)
    df = make_per_trace([1, 2, 3], n_traces=50)
    metrics = ca.compute_metrics(df)

    expected = {
        "shot_id", "ml_coherence", "gt_coherence", "sta_coherence",
        "ml_over_gt_ratio", "n_valid",
    }
    missing = expected - set(metrics.columns)
    ok = not missing
    print(f"  Columns: {list(metrics.columns)}")
    if missing:
        print(f"  Missing: {sorted(missing)}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("=" * 80)
    print("VERIFY DIAGNOSE (F.2)")
    print("=" * 80)

    tests = [
        ("Coherence smooth", test_coherence_smooth),
        ("Coherence zigzag", test_coherence_zigzag),
        ("ML/GT ratio", test_coherence_ratio),
        ("Coherence edge cases", test_coherence_edge_cases),
        ("Gallery generation", test_gallery_generation),
        ("Metrics columns", test_metrics_columns),
    ]

    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"\n  ❌ {name} raised: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    all_ok = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("🎉 ALL CHECKS PASSED — F.2 is correct.")
        return 0
    else:
        print("❌ SOME CHECKS FAILED — inspect the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())