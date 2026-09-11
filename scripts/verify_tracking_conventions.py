#!/usr/bin/env python3
# file location: scripts/verify_tracking_conventions.py

"""
Contract check for the tracking conventions module (C.1).

Verifies that:
    1. All experiment-name constants have the expected strings
    2. All helper functions return dict[str, str] with no None values
    3. Every canonical tag key is present in the appropriate helpers
    4. The sweep helper includes its extra tags

Fast, deterministic, no MLflow server required.

Usage:
    python scripts/verify_tracking_conventions.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.tracking_conventions import (
    ALL_EXPERIMENTS,
    CANONICAL_TAG_KEYS,
    EXPERIMENT_BASELINES,
    EXPERIMENT_EVALUATION,
    EXPERIMENT_SWEEPS,
    EXPERIMENT_TRAINING,
    TAG_DATASET,
    TAG_ENV,
    TAG_MODEL_TYPE,
    TAG_PHASE,
    baseline_tags,
    evaluation_tags,
    sweep_tags,
    training_tags,
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_EXPERIMENTS = {
    "EXPERIMENT_TRAINING": "seismic-fbp-training",
    "EXPERIMENT_EVALUATION": "seismic-fbp-evaluation",
    "EXPERIMENT_BASELINES": "seismic-fbp-baselines",
    "EXPERIMENT_SWEEPS": "seismic-fbp-sweeps",
}

EXPECTED_TAG_KEYS = {
    "TAG_DATASET": "dataset",
    "TAG_MODEL_TYPE": "model_type",
    "TAG_PHASE": "phase",
    "TAG_ENV": "env",
}


# ============================================================
# CHECKS
# ============================================================

def check_experiment_constants() -> bool:
    """Assert each experiment-name constant matches the expected string."""
    print("\n" + "=" * 80)
    print("1. Experiment name constants")
    print("=" * 80)

    actual = {
        "EXPERIMENT_TRAINING": EXPERIMENT_TRAINING,
        "EXPERIMENT_EVALUATION": EXPERIMENT_EVALUATION,
        "EXPERIMENT_BASELINES": EXPERIMENT_BASELINES,
        "EXPERIMENT_SWEEPS": EXPERIMENT_SWEEPS,
    }

    passed = True
    for name, expected in EXPECTED_EXPERIMENTS.items():
        got = actual[name]
        match = got == expected
        status = "✅" if match else "❌"
        print(f"  {status} {name:<24} = {got!r}  (expected {expected!r})")
        if not match:
            passed = False

    # ALL_EXPERIMENTS should be the tuple of all four
    expected_all = tuple(EXPECTED_EXPERIMENTS.values())
    if tuple(ALL_EXPERIMENTS) == expected_all:
        print(f"  ✅ ALL_EXPERIMENTS       = {ALL_EXPERIMENTS}")
    else:
        print(f"  ❌ ALL_EXPERIMENTS       = {ALL_EXPERIMENTS}")
        print(f"     expected              = {expected_all}")
        passed = False

    return passed


def check_tag_key_constants() -> bool:
    """Assert each tag-key constant matches the expected string."""
    print("\n" + "=" * 80)
    print("2. Tag key constants")
    print("=" * 80)

    actual = {
        "TAG_DATASET": TAG_DATASET,
        "TAG_MODEL_TYPE": TAG_MODEL_TYPE,
        "TAG_PHASE": TAG_PHASE,
        "TAG_ENV": TAG_ENV,
    }

    passed = True
    for name, expected in EXPECTED_TAG_KEYS.items():
        got = actual[name]
        match = got == expected
        status = "✅" if match else "❌"
        print(f"  {status} {name:<16} = {got!r}  (expected {expected!r})")
        if not match:
            passed = False

    # CANONICAL_TAG_KEYS should equal the four tag keys in order
    expected_canonical = tuple(EXPECTED_TAG_KEYS.values())
    if tuple(CANONICAL_TAG_KEYS) == expected_canonical:
        print(f"  ✅ CANONICAL_TAG_KEYS = {CANONICAL_TAG_KEYS}")
    else:
        print(f"  ❌ CANONICAL_TAG_KEYS = {CANONICAL_TAG_KEYS}")
        print(f"     expected           = {expected_canonical}")
        passed = False

    return passed


def check_helper_shape(name: str, tags: dict) -> bool:
    """Assert a helper returns dict[str, str] with no None values."""
    ok = True
    if not isinstance(tags, dict):
        print(f"    ❌ {name}: not a dict (got {type(tags).__name__})")
        return False
    for k, v in tags.items():
        if not isinstance(k, str):
            print(f"    ❌ {name}: key {k!r} is not str")
            ok = False
        if not isinstance(v, str):
            print(f"    ❌ {name}: value {v!r} for key {k!r} is not str")
            ok = False
    return ok


def check_helper_canonical_keys(name: str, tags: dict) -> bool:
    """Assert a helper includes all four canonical keys."""
    missing = [k for k in CANONICAL_TAG_KEYS if k not in tags]
    if missing:
        print(f"    ❌ {name}: missing canonical keys {missing}")
        return False
    return True


def check_helpers() -> bool:
    """Call each helper and check shape + canonical keys."""
    print("\n" + "=" * 80)
    print("3. Helper functions")
    print("=" * 80)

    passed = True

    # training_tags
    t = training_tags(dataset="Halfmile", model_type="MPSLightUNet")
    print(f"\n  training_tags('Halfmile', 'MPSLightUNet'):")
    for k, v in t.items():
        print(f"    {k}: {v}")
    ok = check_helper_shape("training_tags", t) and check_helper_canonical_keys("training_tags", t)
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    passed = passed and ok

    # evaluation_tags
    e = evaluation_tags(dataset="Halfmile", model_type="MPSLightUNet")
    print(f"\n  evaluation_tags('Halfmile', 'MPSLightUNet'):")
    for k, v in e.items():
        print(f"    {k}: {v}")
    ok = check_helper_shape("evaluation_tags", e) and check_helper_canonical_keys("evaluation_tags", e)
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    passed = passed and ok

    # baseline_tags
    b = baseline_tags(dataset="Halfmile", method="STALTA")
    print(f"\n  baseline_tags('Halfmile', 'STALTA'):")
    for k, v in b.items():
        print(f"    {k}: {v}")
    ok = check_helper_shape("baseline_tags", b) and check_helper_canonical_keys("baseline_tags", b)
    # Also assert phase defaults to baseline-v1.0
    if b.get(TAG_PHASE) != "baseline-v1.0":
        print(f"    ❌ baseline_tags: phase should default to 'baseline-v1.0', got {b.get(TAG_PHASE)!r}")
        ok = False
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    passed = passed and ok

    # sweep_tags
    s = sweep_tags(dataset="Halfmile", model="pico", loss="combo", sweep_id="abc123")
    print(f"\n  sweep_tags('Halfmile', 'pico', 'combo', 'abc123'):")
    for k, v in s.items():
        print(f"    {k}: {v}")
    ok = check_helper_shape("sweep_tags", s) and check_helper_canonical_keys("sweep_tags", s)
    # Extra sweep-specific keys
    for extra in ("loss", "sweep_id"):
        if extra not in s:
            print(f"    ❌ sweep_tags: missing extra tag {extra!r}")
            ok = False
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    passed = passed and ok

    return passed


def check_none_handling() -> bool:
    """Verify None values are dropped rather than kept."""
    print("\n" + "=" * 80)
    print("4. None handling")
    print("=" * 80)

    t = training_tags(dataset="Halfmile", model_type="MPSLightUNet", phase=None)  # type: ignore[arg-type]
    if TAG_PHASE in t:
        print(f"  ❌ phase=None was kept as {t[TAG_PHASE]!r}; expected it to be dropped")
        return False
    print(f"  ✅ phase=None dropped from output (keys: {sorted(t.keys())})")
    return True


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("=" * 80)
    print("VERIFY TRACKING CONVENTIONS (C.1)")
    print("=" * 80)

    results = [
        ("Experiment constants", check_experiment_constants()),
        ("Tag key constants", check_tag_key_constants()),
        ("Helper functions", check_helpers()),
        ("None handling", check_none_handling()),
    ]

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    all_passed = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL CHECKS PASSED — C.1 is correct.")
        return 0
    else:
        print("❌ SOME CHECKS FAILED — inspect the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())