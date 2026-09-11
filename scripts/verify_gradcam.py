#!/usr/bin/env python3
# file location: scripts/verify_gradcam.py

"""
Verify Grad-CAM correctness (F.1).

Usage:
    python scripts/verify_gradcam.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainability import Explainer, GradCAM, find_last_conv_layer

# ============================================================
# TEST MODELS
# ============================================================

class TinySeg(nn.Module):
    """
    Tiny segmentation model: 2 conv layers, no downsampling.
    Output: 3 classes at input resolution.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 3, 3, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        return x


class ConstantModel(nn.Module):
    """Always outputs zero logits regardless of input."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        # Zero the weights
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)


# ============================================================
# TESTS
# ============================================================

def test_contract():
    print()
    print("TEST 1 — Contract: subclass, find_last_conv_layer, output format")
    ok = True

    if not issubclass(GradCAM, Explainer):
        print("  ❌ GradCAM is not a subclass of Explainer")
        ok = False
    else:
        print("  ✅ GradCAM is a subclass of Explainer")

    model = TinySeg()
    last = find_last_conv_layer(model)
    if last is model.conv2:
        print("  ✅ find_last_conv_layer returns the last conv (conv2)")
    else:
        print(f"  ❌ find_last_conv_layer returned {last}, expected conv2")
        ok = False

    cam = GradCAM()
    x = torch.randn(1, 1, 32, 48)
    heatmap = cam.explain(model, x, target_class=2)

    if heatmap.shape != (32, 48):
        print(f"  ❌ heatmap shape {heatmap.shape}, expected (32, 48)")
        ok = False
    else:
        print(f"  ✅ heatmap shape: {heatmap.shape}")

    if heatmap.dtype != np.float32:
        print(f"  ❌ heatmap dtype {heatmap.dtype}, expected float32")
        ok = False
    else:
        print(f"  ✅ heatmap dtype: {heatmap.dtype}")

    if heatmap.min() < 0 or heatmap.max() > 1:
        print(f"  ❌ heatmap values outside [0, 1]: min={heatmap.min()}, "
              f"max={heatmap.max()}")
        ok = False
    else:
        print(f"  ✅ heatmap range: [{heatmap.min():.4f}, "
              f"{heatmap.max():.4f}]")

    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_known_peak():
    """
    Train a tiny model to output class 1 where a bright spot is.
    Grad-CAM should highlight the bright spot.
    """
    print()
    print("TEST 2 — Known-peak: heatmap highlights the region that "
          "drives the prediction")

    torch.manual_seed(0)
    model = TinySeg()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Train the model: bright spot at (10, 15) in a 32x48 image
    # Target: class 2 logits high near the spot
    for step in range(200):
        x = torch.zeros(1, 1, 32, 48)
        # Small noise
        x += 0.05 * torch.randn_like(x)
        # Bright spot
        x[0, 0, 8:13, 13:18] = 1.0

        # Target mask: class 2 at the spot location
        target = torch.zeros(1, 32, 48, dtype=torch.long)
        target[0, 8:13, 13:18] = 2

        logits = model(x)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Now run Grad-CAM on the same kind of input
    x = torch.zeros(1, 1, 32, 48)
    x[0, 0, 8:13, 13:18] = 1.0

    cam = GradCAM()
    heatmap = cam.explain(model, x, target_class=2)

    # Peak location
    peak_row, peak_col = np.unravel_index(
        np.argmax(heatmap), heatmap.shape
    )
    print(f"  Peak location: ({peak_row}, {peak_col})")
    print("  Expected around: (10, 15)")

    # Within 6 pixels in each direction
    ok = abs(peak_row - 10) <= 6 and abs(peak_col - 15) <= 6
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} "
          f"(peak should be near (10, 15))")
    return ok


def test_zero_gradient():
    print()
    print("TEST 3 — Constant model produces all-zero heatmap")

    model = ConstantModel()
    x = torch.randn(1, 1, 32, 48)
    cam = GradCAM()
    heatmap = cam.explain(model, x, target_class=0)

    ok = np.all(heatmap == 0)
    print(f"  Heatmap max: {heatmap.max()}")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_hirescam():
    print()
    print("TEST 4 — HiResCAM produces a valid heatmap")

    torch.manual_seed(0)
    model = TinySeg()
    x = torch.randn(1, 1, 32, 48)

    cam = GradCAM(method="hirescam")
    heatmap = cam.explain(model, x, target_class=2)

    ok = (
        heatmap.shape == (32, 48)
        and heatmap.dtype == np.float32
        and heatmap.min() >= 0
        and heatmap.max() <= 1
    )
    print(f"  Shape: {heatmap.shape}, dtype: {heatmap.dtype}, "
          f"range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_real_model():
    print()
    print("TEST 5 — Real model smoke test")

    from src.config import SeismicConfig
    from src.models.loader import load_evaluation_model
    from src.utils.logger import setup_logger

    checkpoint = "models/registry/MPSLightUNet_Halfmile_best.pt"
    if not os.path.exists(checkpoint):
        print(f"  ⚠️  Checkpoint not found: {checkpoint} — skipping")
        return True

    logger = setup_logger(task_name="verify_gradcam")
    cfg = SeismicConfig(dataset_name="Halfmile")
    device = torch.device("cpu")  # CPU is fine for verify

    model = load_evaluation_model(checkpoint, cfg, device, logger)

    # Fake a shot-sized input
    x = torch.randn(1, 1, 1578, 751).to(device)

    cam = GradCAM()
    heatmap = cam.explain(model, x, target_class=2)

    ok = (
        heatmap.shape == (1578, 751)
        and heatmap.dtype == np.float32
        and heatmap.min() >= 0
        and heatmap.max() <= 1
    )
    print(f"  Heatmap shape: {heatmap.shape}")
    print(f"  Range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"  Non-trivial (max > 0): {heatmap.max() > 0}")
    ok = ok and heatmap.max() > 0
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("=" * 80)
    print("VERIFY GRAD-CAM (F.1)")
    print("=" * 80)

    tests = [
        ("Contract", test_contract),
        ("Known-peak", test_known_peak),
        ("Zero gradient", test_zero_gradient),
        ("HiResCAM variant", test_hirescam),
        ("Real model smoke test", test_real_model),
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
        print("🎉 ALL CHECKS PASSED — F.1 is correct.")
        return 0
    else:
        print("❌ SOME CHECKS FAILED — inspect the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())