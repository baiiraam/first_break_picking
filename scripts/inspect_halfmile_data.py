#!/usr/bin/env python3
# file location: scripts/inspect_halfmile_data.py

"""
Inspect Halfmile HDF5 data to understand:
    - Shape, dtype, value range
    - Ground-truth pick distribution
    - Amplitude characteristics before/after the pick
    - A small ASCII visualization of a few traces near their picks

Read-only. No changes to any data or state.

Usage:
    python scripts/inspect_halfmile_data.py
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


HDF5_PATH = "data/raw/Halfmile3D_add_geom_sorted.hdf5"
N_SHOTS_TO_SAMPLE = 5          # how many shots to look at in detail
N_TRACES_PER_SHOT = 5          # traces per shot for the ASCII view
SAMPLING_INTERVAL_MS = 2.0     # from configs/halfmile.yaml


# ============================================================
# HELPERS
# ============================================================

def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def ascii_trace(trace: np.ndarray, width: int = 70, height: int = 6) -> list[str]:
    """
    Render a 1D trace as a small ASCII plot (rows top-to-bottom).
    Down-samples `width` columns; scales to `height` rows.
    """
    if len(trace) > width:
        idx = np.linspace(0, len(trace) - 1, width).astype(int)
        trace = trace[idx]

    vmin, vmax = float(trace.min()), float(trace.max())
    if vmax == vmin:
        vmax = vmin + 1e-9

    rows = [[" "] * len(trace) for _ in range(height)]
    for i, v in enumerate(trace):
        level = int((v - vmin) / (vmax - vmin) * (height - 1))
        row = height - 1 - level
        rows[row][i] = "█"

    lines = ["".join(r) for r in rows]
    return lines


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    if not os.path.exists(HDF5_PATH):
        print(f"❌ HDF5 not found: {HDF5_PATH}")
        return 1

    banner("HALFMILE DATA INSPECTION")

    with h5py.File(HDF5_PATH, "r") as f:
        group = f["TRACE_DATA"]["DEFAULT"]
        data_ds = group["data_array"]
        shotids_ds = group["SHOTID"]
        picks_ds = group["SPARE1"]

        total_rows = data_ds.shape[0]
        n_samples = data_ds.shape[1]

        print(f"File:               {HDF5_PATH}")
        print(f"data_array shape:   {data_ds.shape}")
        print(f"data_array dtype:   {data_ds.dtype}")
        print(f"SPARE1 shape:       {picks_ds.shape}")
        print(f"SPARE1 dtype:       {picks_ds.dtype}")
        print(f"Sampling interval:  {SAMPLING_INTERVAL_MS} ms/sample (from config)")

        # ---- Global amplitude stats on a sample of rows ----
        banner("GLOBAL AMPLITUDE STATS (first 5000 rows)")

        sample = data_ds[: min(5000, total_rows), :]
        print(f"  Shape sampled:   {sample.shape}")
        print(f"  min:             {sample.min():.4f}")
        print(f"  max:             {sample.max():.4f}")
        print(f"  mean:            {sample.mean():.4f}")
        print(f"  std:             {sample.std():.4f}")
        print(f"  abs p50:         {np.percentile(np.abs(sample), 50):.4f}")
        print(f"  abs p90:         {np.percentile(np.abs(sample), 90):.4f}")
        print(f"  abs p99:         {np.percentile(np.abs(sample), 99):.4f}")
        print(f"  abs max:         {np.abs(sample).max():.4f}")

        # ---- Pick distribution ----
        banner("GROUND-TRUTH PICK DISTRIBUTION")

        picks = picks_ds[:, 0]
        valid = picks[(picks > 0) & (picks < n_samples)]
        n_total = len(picks)
        n_valid = len(valid)

        print(f"  Total traces:    {n_total}")
        print(f"  Valid picks:     {n_valid} ({100 * n_valid / n_total:.1f}%)")
        print(f"  Invalid (≤0):    {n_total - n_valid}")

        if n_valid > 0:
            print(f"  min (ms):        {valid.min():.2f}")
            print(f"  max (ms):        {valid.max():.2f}")
            print(f"  mean (ms):       {valid.mean():.2f}")
            print(f"  median (ms):     {np.median(valid):.2f}")
            print(f"  std (ms):        {valid.std():.2f}")
            print()
            print("  → in SAMPLES (after dividing by sampling interval):")
            valid_samples = valid / SAMPLING_INTERVAL_MS
            print(f"  min:             {valid_samples.min():.1f}")
            print(f"  max:             {valid_samples.max():.1f}")
            print(f"  mean:            {valid_samples.mean():.1f}")
            print(f"  median:          {np.median(valid_samples):.1f}")
            print(f"  std:             {valid_samples.std():.1f}")
            print()
            print("  Histogram of pick samples (10 bins):")
            hist, edges = np.histogram(valid_samples, bins=10)
            for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
                bar = "█" * int(count / max(hist.max(), 1) * 40)
                print(f"    [{lo:6.1f}–{hi:6.1f}) {count:5d} {bar}")

        # ---- Find unique shots ----
        banner("SHOT STRUCTURE")

        shotids = shotids_ds[:].flatten()
        unique_shots, first_idx, counts = np.unique(
            shotids, return_index=True, return_counts=True
        )
        print(f"  Number of unique shots: {len(unique_shots)}")
        print(f"  Trace counts per shot:")
        print(f"    min:  {counts.min()}")
        print(f"    max:  {counts.max()}")
        print(f"    mean: {counts.mean():.1f}")
        print(f"    std:  {counts.std():.1f}")

        # ---- Detailed look at a few shots ----
        banner(f"DETAILED LOOK AT FIRST {N_SHOTS_TO_SAMPLE} SHOTS")

        for s_idx in range(min(N_SHOTS_TO_SAMPLE, len(unique_shots))):
            shot_id = int(unique_shots[s_idx])
            start = int(first_idx[s_idx])
            end = start + int(counts[s_idx])

            print()
            print("-" * 78)
            print(f"Shot {s_idx}:  SHOTID={shot_id}  "
                  f"rows={start}..{end}  n_traces={counts[s_idx]}")
            print("-" * 78)

            shot_data = data_ds[start:end, :]
            shot_picks = picks_ds[start:end, 0]

            # Range and stats
            print(f"  Data range:  [{shot_data.min():.4f}, {shot_data.max():.4f}]")
            print(f"  Data std:    {shot_data.std():.4f}")

            # Pick stats for this shot
            valid_picks_ms = shot_picks[(shot_picks > 0) & (shot_picks < n_samples)]
            if len(valid_picks_ms) == 0:
                print("  No valid picks in this shot.")
                continue

            valid_picks_samples = valid_picks_ms / SAMPLING_INTERVAL_MS
            print(f"  Valid picks: {len(valid_picks_ms)}/{len(shot_picks)} "
                  f"({100 * len(valid_picks_ms) / len(shot_picks):.1f}%)")
            print(f"  Pick sample range: "
                  f"[{valid_picks_samples.min():.1f}, {valid_picks_samples.max():.1f}]  "
                  f"median={np.median(valid_picks_samples):.1f}")

            # Show a few traces near their picks
            print()
            print(f"  First {N_TRACES_PER_SHOT} traces (ASCII, "
                  f"downsampled to ~70 columns):")
            for t in range(min(N_TRACES_PER_SHOT, shot_data.shape[0])):
                tr = shot_data[t]
                pick_ms = shot_picks[t]
                if pick_ms > 0 and pick_ms < n_samples:
                    pick_str = f"pick={pick_ms:.1f}ms ({pick_ms / SAMPLING_INTERVAL_MS:.1f} samp)"
                else:
                    pick_str = "pick=INVALID"

                # Local stats
                tr_std = tr.std()

                print(f"    trace {t:3d}  {pick_str}  "
                      f"std={tr_std:.4f}  "
                      f"[{tr.min():.3f}, {tr.max():.3f}]")
                for line in ascii_trace(tr, width=70, height=4):
                    print(f"      |{line}|")

                # Amplitude comparison: pre-pick window vs post-pick window
                if pick_ms > 0 and pick_ms < n_samples:
                    pick_samp = int(round(pick_ms / SAMPLING_INTERVAL_MS))
                    pre_lo = max(0, pick_samp - 50)
                    pre_hi = max(0, pick_samp - 5)
                    post_lo = min(n_samples, pick_samp + 5)
                    post_hi = min(n_samples, pick_samp + 50)

                    if pre_hi > pre_lo and post_hi > post_lo:
                        pre_rms = float(np.sqrt(np.mean(tr[pre_lo:pre_hi] ** 2)))
                        post_rms = float(np.sqrt(np.mean(tr[post_lo:post_hi] ** 2)))
                        ratio = post_rms / (pre_rms + 1e-12)
                        print(f"      RMS pre-pick [{pre_lo}..{pre_hi}]: {pre_rms:.4f}")
                        print(f"      RMS post-pick[{post_lo}..{post_hi}]: {post_rms:.4f}")
                        print(f"      Ratio post/pre: {ratio:.2f}×")

    banner("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())