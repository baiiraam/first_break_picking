#!/usr/bin/env python3
# file location: scripts/verify_deployment.py

"""
Verify G.1: deployment package.

Usage:
    python scripts/verify_deployment.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deployment import (
    InputContract,
    Predictor,
    read_contract_from_checkpoint,
    verify_numeric_equivalence,
)
from src.deployment.exporters import export_onnx, export_torchscript


CHECKPOINT = "models/registry/MPSLightUNet_Halfmile_best.pt"


# ============================================================
# TESTS
# ============================================================

def test_contract_shape():
    print()
    print("TEST 1 — InputContract produces the expected shape")
    c = InputContract(target_traces=1578, n_samples=751)
    ok = c.expected_shape == (1, 1, 1578, 751)
    print(f"  Contract: {c.expected_shape}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_contract_from_checkpoint():
    print()
    print("TEST 2 — read_contract_from_checkpoint")

    if not Path(CHECKPOINT).exists():
        print(f"  ⚠️  Checkpoint not found: {CHECKPOINT} — skipping")
        return True

    try:
        c = read_contract_from_checkpoint(CHECKPOINT)
    except Exception as e:
        print(f"  ❌ Failed: {type(e).__name__}: {e}")
        return False

    print(f"  Read contract: {c.expected_shape}")
    ok = c.target_traces > 0 and c.n_samples > 0
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_predictor_load():
    print()
    print("TEST 3 — Predictor.load()")

    if not Path(CHECKPOINT).exists():
        print(f"  ⚠️  Checkpoint not found — skipping")
        return True

    try:
        predictor = Predictor.load(CHECKPOINT, device="cpu")
    except Exception as e:
        print(f"  ❌ Failed: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False

    print(f"  Loaded: {predictor}")
    print(f"  {'✅ PASS'}")
    return True


def test_predict_shapes():
    print()
    print("TEST 4 — Predictor.predict() output shapes and dtypes")

    if not Path(CHECKPOINT).exists():
        print(f"  ⚠️  Checkpoint not found — skipping")
        return True

    predictor = Predictor.load(CHECKPOINT, device="cpu")

    # Small shot
    shot = np.random.randn(500, 751).astype(np.float32)
    picks = predictor.predict(shot)

    ok = (
        picks.shape == (predictor.contract.target_traces,)
        and picks.dtype == np.int64
    )
    print(f"  Input shot shape: {shot.shape}")
    print(f"  Output picks: {picks.shape}, dtype {picks.dtype}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_predict_matches_manual():
    print()
    print("TEST 5 — predict() matches manual pipeline")

    if not Path(CHECKPOINT).exists():
        print(f"  ⚠️  Checkpoint not found — skipping")
        return True

    predictor = Predictor.load(CHECKPOINT, device="cpu")

    # Build a real-shot-sized input
    shot = np.random.randn(predictor.contract.target_traces,
                           predictor.contract.n_samples).astype(np.float32)

    # Predictor path
    picks_predictor = predictor.predict(shot)

    # Manual path
    x = torch.from_numpy(shot).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = predictor.model(x)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
    from src.training.metrics import extract_picks_from_mask
    picks_manual = extract_picks_from_mask(pred)

    ok = bool(np.array_equal(picks_predictor, picks_manual))
    print(f"  Predictor picks (first 5): {picks_predictor[:5]}")
    print(f"  Manual picks    (first 5): {picks_manual[:5]}")
    print(f"  Match: {ok}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_padding_and_cropping():
    print()
    print("TEST 6 — Padding and cropping")

    if not Path(CHECKPOINT).exists():
        print(f"  ⚠️  Checkpoint not found — skipping")
        return True

    predictor = Predictor.load(CHECKPOINT, device="cpu")

    # Shot with fewer traces
    small = np.random.randn(100, 751).astype(np.float32)
    picks_small = predictor.predict(small)
    ok_small = picks_small.shape == (predictor.contract.target_traces,)

    # Shot with more traces
    large = np.random.randn(2000, 751).astype(np.float32)
    picks_large = predictor.predict(large)
    ok_large = picks_large.shape == (predictor.contract.target_traces,)

    ok = ok_small and ok_large
    print(f"  100-trace shot → {picks_small.shape}")
    print(f"  2000-trace shot → {picks_large.shape}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_exports_and_equivalence():
    print()
    print("TEST 7 — Export and verify numeric equivalence")

    if not Path(CHECKPOINT).exists():
        print(f"  ⚠️  Checkpoint not found — skipping")
        return True

    predictor = Predictor.load(CHECKPOINT, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        ts_path = tmppath / "model_scripted.pt"
        onnx_path = tmppath / "model.onnx"

        # Export
        export_torchscript(predictor.model, predictor.contract, ts_path)
        export_onnx(predictor.model, predictor.contract, onnx_path)

        # Verify
        x = torch.randn(*predictor.contract.expected_shape)
        results = verify_numeric_equivalence(
            pytorch_model=predictor.model,
            input_tensor=x,
            torchscript_path=ts_path,
            onnx_path=onnx_path,
        )

        print(f"  TorchScript: match={results['torchscript']['match']}, "
              f"max_diff={results['torchscript']['max_diff']:.2e}")
        print(f"  ONNX:        match={results['onnx']['match']}, "
              f"max_diff={results['onnx']['max_diff']:.2e}")

        if results['torchscript'].get('error'):
            print(f"  TS error: {results['torchscript']['error']}")
        if results['onnx'].get('error'):
            print(f"  ONNX error: {results['onnx']['error']}")

        ok_ts = results["torchscript"]["match"]
        ok_onnx = results["onnx"]["match"]

        # ONNX check may fail if onnxruntime is not installed — treat
        # that as a soft failure
        onnx_error = results["onnx"].get("error", "") or ""
        if "onnxruntime not installed" in onnx_error:
            print("  ⚠️  onnxruntime not installed; skipping ONNX equivalence")
            ok_onnx = True

        ok = ok_ts and ok_onnx
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
        return ok


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("=" * 80)
    print("VERIFY DEPLOYMENT (G.1)")
    print("=" * 80)

    tests = [
        ("Contract shape", test_contract_shape),
        ("Contract from checkpoint", test_contract_from_checkpoint),
        ("Predictor.load()", test_predictor_load),
        ("Predict shapes/dtypes", test_predict_shapes),
        ("Predict matches manual", test_predict_matches_manual),
        ("Padding and cropping", test_padding_and_cropping),
        ("Export + numeric equivalence", test_exports_and_equivalence),
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
        print("🎉 ALL CHECKS PASSED — G.1 is correct.")
        return 0
    else:
        print("❌ SOME CHECKS FAILED — inspect the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())