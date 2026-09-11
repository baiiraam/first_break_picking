#!/usr/bin/env python3
# file location: scripts/verify_extract_picks.py

"""
Verify extract_picks_from_mask correctness (E.2).

Constructs synthetic masks with known strip positions and asserts
that the function returns the correct center.

Usage:
    python scripts/verify_extract_picks.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.metrics import extract_picks_from_mask

N_SAMPLES = 751


# ============================================================
# HELPERS
# ============================================================

def make_mask_with_strip(strip_start: int, strip_end: int, n_samples=N_SAMPLES):
    """
    Build a mask with:
        class 0 before strip
        class 2 from strip_start to strip_end (inclusive)
        class 1 after strip
    Returns (1, n_samples) int64.
    """
    mask = np.zeros((1, n_samples), dtype=np.int64)
    mask[0, strip_start:strip_end + 1] = 2
    mask[0, strip_end + 1:] = 1
    return mask


def make_mask_without_strip(first_after: int, n_samples=N_SAMPLES):
    """
    Build a mask with no class-2 strip, but class-1 starting at
    `first_after`. Returns (1, n_samples) int64.
    """
    mask = np.zeros((1, n_samples), dtype=np.int64)
    mask[0, first_after:] = 1
    return mask


# ============================================================
# TESTS
# ============================================================

def test_clean_strip():
    print()
    print("TEST 1 — Clean strip, known center")
    mask = make_mask_with_strip(strip_start=100, strip_end=108)
    pick = int(extract_picks_from_mask(mask)[0])
    expected = (100 + 108) // 2  # = 104
    print(f"  Strip: 100-108, expected pick: {expected}, got: {pick}")
    ok = pick == expected
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_strip_near_start():
    print()
    print("TEST 2 — Strip near sample 0")
    mask = make_mask_with_strip(strip_start=0, strip_end=8)
    pick = int(extract_picks_from_mask(mask)[0])
    expected = 4
    print(f"  Strip: 0-8, expected pick: {expected}, got: {pick}")
    ok = pick == expected
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_strip_near_end():
    print()
    print("TEST 3 — Strip near last sample")
    mask = make_mask_with_strip(strip_start=742, strip_end=750)
    pick = int(extract_picks_from_mask(mask)[0])
    expected = 746
    print(f"  Strip: 742-750, expected pick: {expected}, got: {pick}")
    ok = pick == expected
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_no_strip_with_class1():
    print()
    print("TEST 4 — No strip, class-1 starts at sample 100")
    mask = make_mask_without_strip(first_after=100)
    pick = int(extract_picks_from_mask(mask)[0])
    expected = max(100 - 4, 0)  # = 96
    print(f"  First class-1: 100, expected pick: {expected}, got: {pick}")
    ok = pick == expected
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_all_zeros():
    print()
    print("TEST 5 — All-zeros mask")
    mask = np.zeros((1, N_SAMPLES), dtype=np.int64)
    pick = int(extract_picks_from_mask(mask)[0])
    expected = 0
    print(f"  Expected: {expected}, got: {pick}")
    ok = pick == expected
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_multiple_traces():
    print()
    print("TEST 6 — 5 traces, different strip positions")
    # Strips: [100-108], [200-208], [300-308], [400-408], [500-508]
    mask = np.zeros((5, N_SAMPLES), dtype=np.int64)
    expected_centers = [104, 204, 304, 404, 504]
    for i, center in enumerate(expected_centers):
        start = center - 4
        end = center + 4
        mask[i, start:end + 1] = 2
        mask[i, end + 1:] = 1

    picks = extract_picks_from_mask(mask)
    ok = True
    for i, expected in enumerate(expected_centers):
        got = int(picks[i])
        match = got == expected
        if not match:
            ok = False
        print(f"  trace {i}: expected {expected}, got {got} "
              f"{'✅' if match else '❌'}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_asymmetric_strip():
    print()
    print("TEST 7 — Asymmetric strip 100-103 (4 samples)")
    mask = make_mask_with_strip(strip_start=100, strip_end=103)
    pick = int(extract_picks_from_mask(mask)[0])
    expected = (100 + 103) // 2  # = 101
    print(f"  Strip: 100-103, expected pick: {expected}, got: {pick}")
    ok = pick == expected
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_dtype_shape():
    print()
    print("TEST 8 — dtype and shape contract")
    mask = make_mask_with_strip(strip_start=100, strip_end=108)
    picks = extract_picks_from_mask(mask)
    ok = picks.dtype == np.int64 and picks.shape == (1,)
    print(f"  dtype: {picks.dtype}, shape: {picks.shape}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("=" * 80)
    print("VERIFY extract_picks_from_mask (E.2)")
    print("=" * 80)

    tests = [
        ("Clean strip", test_clean_strip),
        ("Strip near start", test_strip_near_start),
        ("Strip near end", test_strip_near_end),
        ("No strip, class-1 present", test_no_strip_with_class1),
        ("All-zeros mask", test_all_zeros),
        ("Multiple traces", test_multiple_traces),
        ("Asymmetric strip", test_asymmetric_strip),
        ("dtype and shape contract", test_dtype_shape),
    ]

    results = []
    for name, fn in tests:
        results.append((name, fn()))

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
        print("🎉 ALL CHECKS PASSED — E.2 is correct.")
        return 0
    else:
        print("❌ SOME CHECKS FAILED — inspect the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())