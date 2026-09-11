#!/usr/bin/env python3
# file location: scripts/verify_sampler_cache_sizing.py

"""
Verify the sampler-aware cache sizing change (D.1).

Uses synthetic dataset metadata + memory budgets so the test is
fast and does not depend on real HDF5 files or chunk manifests.

Usage:
    # 1. Before the change:
    python scripts/verify_sampler_cache_sizing.py --save-baseline

    # 2. Apply the D.1 edit to src/batch/smart_config.py

    # 3. After the change:
    python scripts/verify_sampler_cache_sizing.py --verify
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import click

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch.smart_config import calculate_optimal_config


# ============================================================
# TEST MATRIX
# ============================================================

MODELS = ["pico", "tiny", "mpslight"]

# Each dataset entry: (num_chunks, total_shots)
# Halfmile has many small chunks; Lalor has fewer, larger chunks.
DATASETS = {
    "Halfmile": {"num_chunks": 8, "total_shots": 124},
    "Lalor":    {"num_chunks": 5, "total_shots": 95},
}

SAMPLER_MODES = [True, False]

# Fixed memory / device for all rows (MPS-like Apple Silicon budget)
AVAILABLE_MEMORY_GB = 16.0
DEVICE_TYPE = "mps"

BASELINE_PATH = Path("evaluation_results/verify_sampler_cache/baseline.json")
RESULT_PATH = Path("evaluation_results/verify_sampler_cache/after.json")


# ============================================================
# MONKEY-PATCHED HELPERS
# ============================================================
# calculate_optimal_config() calls get_dataset_info() and
# get_actual_data_shape() which read from disk. We patch them so the
# test is self-contained and deterministic.

def _make_fake_dataset_info(num_chunks: int, total_shots: int):
    def _fake_get_dataset_info(dataset_name: str) -> dict[str, Any]:
        return {
            "total_shots": total_shots,
            "total_traces": 0,
            "samples_per_trace": 0,
            "file_size_mb": 0,
            "chunk_size": 69,
            "num_chunks": num_chunks,
        }
    return _fake_get_dataset_info


def _fake_get_actual_data_shape(dataset_name: str) -> tuple[int, int]:
    return (0, 0)


# ============================================================
# RUN MATRIX
# ============================================================

def run_matrix() -> list[dict[str, Any]]:
    """
    Run calculate_optimal_config() over the full matrix.

    Returns a list of row dicts, one per (model, dataset, sampler) combo.
    Captures max_cache_by_memory from the explanation dict when present.
    """
    import src.batch.smart_config as sc

    original_get_dataset_info = sc.get_dataset_info
    original_get_actual_data_shape = sc.get_actual_data_shape

    rows: list[dict[str, Any]] = []

    try:
        for dataset_name, meta in DATASETS.items():
            sc.get_dataset_info = _make_fake_dataset_info(
                num_chunks=meta["num_chunks"],
                total_shots=meta["total_shots"],
            )
            sc.get_actual_data_shape = _fake_get_actual_data_shape

            for sampler_on in SAMPLER_MODES:
                global_config = {"chunk_aware_sampling": sampler_on}

                for model_name in MODELS:
                    config = calculate_optimal_config(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        available_memory_gb=AVAILABLE_MEMORY_GB,
                        device_type=DEVICE_TYPE,
                        global_config=global_config,
                        auto_config=None,
                    )

                    if not config:
                        rows.append({
                            "model": model_name,
                            "dataset": dataset_name,
                            "sampler": sampler_on,
                            "error": "calculate_optimal_config returned empty",
                        })
                        continue

                    final = config["final_config"]
                    cache_calc = config["calculations"]["optimal_cache"]

                    rows.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "sampler": sampler_on,
                        "optimal_cache": final["cache_size"],
                        "optimal_batch": final["batch_size"],
                        "memory_limit_gb": final["memory_limit_gb"],
                        "cache_policy": cache_calc.get("policy", "N/A"),
                        "max_cache_by_memory": cache_calc.get("max_cache_by_memory"),
                    })
    finally:
        sc.get_dataset_info = original_get_dataset_info
        sc.get_actual_data_shape = original_get_actual_data_shape

    return rows


# ============================================================
# PRINTING
# ============================================================

def print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)
    print(
        f"{'Model':<10} {'Dataset':<10} {'Sampler':<8} "
        f"{'Cache':<7} {'MaxByMem':<10} {'Batch':<7} {'MemLimit':<10} {'Policy':<40}"
    )
    print("-" * 110)
    for r in rows:
        if "error" in r:
            print(
                f"{r['model']:<10} {r['dataset']:<10} "
                f"{str(r['sampler']):<8} ERROR: {r['error']}"
            )
            continue
        max_mem = r.get("max_cache_by_memory")
        max_mem_str = str(max_mem) if max_mem is not None else "N/A"
        print(
            f"{r['model']:<10} {r['dataset']:<10} {str(r['sampler']):<8} "
            f"{r['optimal_cache']:<7} {max_mem_str:<10} "
            f"{r['optimal_batch']:<7} "
            f"{r['memory_limit_gb']:<10} {r['cache_policy']:<40}"
        )
    print("-" * 110)


# ============================================================
# SAVE / LOAD
# ============================================================

def write_baseline(rows: list[dict[str, Any]]) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n💾 Baseline saved to: {BASELINE_PATH}")


def load_baseline() -> list[dict[str, Any]]:
    if not BASELINE_PATH.exists():
        print(f"❌ Baseline not found: {BASELINE_PATH}")
        print("   Run with --save-baseline first.")
        sys.exit(1)
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


def write_after(rows: list[dict[str, Any]]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n💾 After-change results saved to: {RESULT_PATH}")


# ============================================================
# COMPARISON / SUCCESS CRITERIA
# ============================================================

def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    return {
        (r["model"], r["dataset"], r["sampler"]): r
        for r in rows
        if "error" not in r
    }


def compare_and_report(
    baseline: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> bool:
    """
    Compare baseline vs after and print a per-row verdict.

    Success criteria (D.1, Option A, strengthened):
        Rule 1: sampler=ON rows satisfy 1 <= cache <= min(3, num_chunks)
        Rule 2: sampler=OFF rows → every value identical to baseline
        Rule 3: ALL rows → optimal_batch identical to baseline
        Rule 4a: sampler=ON rows → optimal_cache <= max_cache_by_memory
                 (memory is still the hard ceiling)
        Rule 4b: sampler=ON rows → optimal_cache <= baseline_heuristic_cache + 1
                 (no surprise growth beyond 1 chunk)
        Rule 5: exit 0 iff 1-4b hold

    The baseline JSON's optimal_cache field IS the heuristic's prior
    output, so it serves directly as baseline_heuristic_cache for Rule 4b.

    Returns True if all criteria pass.
    """
    b = _index_rows(baseline)
    a = _index_rows(after)

    all_keys = sorted(set(b.keys()) | set(a.keys()))

    print()
    print("=" * 110)
    print("COMPARISON — baseline vs after")
    print("=" * 110)
    print(
        f"{'Model':<10} {'Dataset':<10} {'Sampler':<8} "
        f"{'Cache b→a':<12} {'MaxByMem':<10} {'MemLim b→a':<16} {'Verdict':<30}"
    )
    print("-" * 110)

    rule1_fail: list[str] = []
    rule2_fail: list[str] = []
    rule3_fail: list[str] = []
    rule4a_fail: list[str] = []
    rule4b_fail: list[str] = []

    for key in all_keys:
        model, dataset, sampler = key
        rb = b.get(key)
        ra = a.get(key)

        if rb is None or ra is None:
            print(f"{model:<10} {dataset:<10} {str(sampler):<8} MISSING ROW")
            continue

        cache_b = rb["optimal_cache"]
        cache_a = ra["optimal_cache"]
        batch_b = rb["optimal_batch"]
        batch_a = ra["optimal_batch"]
        mem_b = rb["memory_limit_gb"]
        mem_a = ra["memory_limit_gb"]
        max_by_mem = ra.get("max_cache_by_memory")

        verdict_parts: list[str] = []

        # Rule 1: sampler=ON → 1 <= cache <= min(3, num_chunks)
        if sampler is True:
            num_chunks = DATASETS[dataset]["num_chunks"]
            upper = min(3, num_chunks)
            if 1 <= cache_a <= upper:
                verdict_parts.append("R1 ✅")
            else:
                verdict_parts.append(f"R1 ❌ (need 1..{upper})")
                rule1_fail.append(f"{model}/{dataset}")

        # Rule 2: sampler=OFF → identical to baseline
        if sampler is False:
            if cache_b == cache_a and batch_b == batch_a and mem_b == mem_a:
                verdict_parts.append("R2 ✅")
            else:
                verdict_parts.append("R2 ❌")
                rule2_fail.append(f"{model}/{dataset}")

        # Rule 3: batch identical in both modes
        if batch_b == batch_a:
            verdict_parts.append("R3 ✅")
        else:
            verdict_parts.append("R3 ❌")
            rule3_fail.append(f"{model}/{dataset}")

        # Rules 4a + 4b: sampler=ON only
        if sampler is True:
            # 4a: cache <= max_cache_by_memory
            if max_by_mem is None:
                verdict_parts.append("R4a ⚠️ (no max_by_mem)")
            elif cache_a <= max_by_mem:
                verdict_parts.append("R4a ✅")
            else:
                verdict_parts.append(f"R4a ❌ ({cache_a}>{max_by_mem})")
                rule4a_fail.append(f"{model}/{dataset}")

            # 4b: cache <= baseline + 1
            limit = cache_b + 1
            if cache_a <= limit:
                verdict_parts.append("R4b ✅")
            else:
                verdict_parts.append(f"R4b ❌ ({cache_a}>{limit})")
                rule4b_fail.append(f"{model}/{dataset}")

        verdict = " ".join(verdict_parts)
        max_by_mem_str = str(max_by_mem) if max_by_mem is not None else "N/A"

        print(
            f"{model:<10} {dataset:<10} {str(sampler):<8} "
            f"{cache_b}→{cache_a:<9} {max_by_mem_str:<10} "
            f"{mem_b}→{mem_a:<13} {verdict:<30}"
        )

    print("-" * 110)
    print()

    passed = (
        not rule1_fail
        and not rule2_fail
        and not rule3_fail
        and not rule4a_fail
        and not rule4b_fail
    )

    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"Rule 1  (sampler-ON cache in [1, min(3, num_chunks)]): "
          f"{'✅ PASS' if not rule1_fail else '❌ FAIL: ' + ', '.join(rule1_fail)}")
    print(f"Rule 2  (sampler-OFF unchanged):                       "
          f"{'✅ PASS' if not rule2_fail else '❌ FAIL: ' + ', '.join(rule2_fail)}")
    print(f"Rule 3  (batch unchanged everywhere):                  "
          f"{'✅ PASS' if not rule3_fail else '❌ FAIL: ' + ', '.join(rule3_fail)}")
    print(f"Rule 4a (sampler-ON cache <= memory ceiling):          "
          f"{'✅ PASS' if not rule4a_fail else '❌ FAIL: ' + ', '.join(rule4a_fail)}")
    print(f"Rule 4b (sampler-ON cache <= baseline + 1):            "
          f"{'✅ PASS' if not rule4b_fail else '❌ FAIL: ' + ', '.join(rule4b_fail)}")
    print()

    if passed:
        print("🎉 ALL CHECKS PASSED — D.1 is correct.")
    else:
        print("❌ SOME CHECKS FAILED — inspect the diffs above.")

    return passed


# ============================================================
# CLI
# ============================================================

@click.command()
@click.option(
    "--save-baseline",
    "save_baseline_flag",
    is_flag=True,
    help="Capture current behavior and save as baseline.",
)
@click.option(
    "--verify",
    "verify_flag",
    is_flag=True,
    help="Run after the change and compare against baseline.",
)
def main(save_baseline_flag: bool, verify_flag: bool):
    """Verify the sampler-aware cache sizing change (D.1)."""

    if not save_baseline_flag and not verify_flag:
        print("Please specify --save-baseline or --verify")
        sys.exit(1)

    if save_baseline_flag:
        print()
        print("=" * 110)
        print("BASELINE MODE — capturing current cache sizing behavior")
        print("=" * 110)
        rows = run_matrix()
        print_rows("Baseline results", rows)
        write_baseline(rows)

        print()
        print("Next steps:")
        print("  1. Apply the D.1 edit to src/batch/smart_config.py")
        print("  2. Run:  python scripts/verify_sampler_cache_sizing.py --verify")
        return

    # verify mode
    print()
    print("=" * 110)
    print("VERIFY MODE — running after the change")
    print("=" * 110)

    baseline = load_baseline()
    after = run_matrix()
    print_rows("After-change results", after)
    write_after(after)

    passed = compare_and_report(baseline, after)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()