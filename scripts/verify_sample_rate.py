#!/usr/bin/env python3
"""
Verify sample rate from HDF5 files.
"""

import h5py


def check_sample_rate(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        group = f["TRACE_DATA"]["DEFAULT"]

        # Check SAMP_RATE
        if "SAMP_RATE" in group:
            sample_rate = group["SAMP_RATE"][()]

            # ============================================================
            # FIX: Handle 2D array
            # ============================================================
            if sample_rate.ndim == 2:
                # Get first non-zero value
                sample_rate_val = sample_rate[0, 0] if sample_rate.shape[0] > 0 else 0
            elif sample_rate.ndim == 1:
                sample_rate_val = sample_rate[0] if len(sample_rate) > 0 else 0
            else:
                sample_rate_val = sample_rate

            print(f"  SAMP_RATE: {sample_rate_val} microseconds")
            print(f"  → {sample_rate_val / 1000:.2f} ms per sample")
            print(f"  → {1000 / sample_rate_val:.2f} samples per ms")

            # Check SPARE1 values
            spare = group["SPARE1"][()].flatten()
            valid = spare[spare > 0]
            if len(valid) > 0:
                max_ms = valid.max()
                max_samples = max_ms / (sample_rate_val / 1000)
                print(f"  SPARE1 max: {max_ms:.2f} ms")
                print(f"  Max sample index: {max_samples:.2f}")
                print(f"  Within n_samples? {max_samples < 751}")
        else:
            print("  SAMP_RATE not found")

        # Also check data_array shape
        if "data_array" in group:
            data = group["data_array"]
            print(f"  Data shape: {data.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 VERIFYING SAMPLE RATE")
    print("=" * 60)

    datasets = [
        ("Halfmile", "data/raw/Halfmile3D_add_geom_sorted.hdf5"),
        ("Brunswick", "data/raw/Brunswick_orig_1500ms_V2.hdf5"),
        ("Lalor", "data/raw/Lalor_raw_z_1500ms_norp_geom_v3.hdf5"),
        ("Sudbury", "data/raw/preprocessed_Sudbury3D.hdf"),
    ]

    for name, path in datasets:
        print(f"\n📁 {name}:")
        try:
            check_sample_rate(path)
        except FileNotFoundError:
            print(f"  ❌ File not found: {path}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
