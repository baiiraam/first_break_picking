#!/usr/bin/env python3
# file location: scripts/benchmark_inference.py

"""
Inference latency benchmark.

Measures:
    - Model load time (one-off)
    - Per-shot inference time (single shot, batch=1)
    - Per-shot inference time across multiple shots (amortized)
    - Throughput (shots per second)

Compares PyTorch, TorchScript, and ONNX if exports exist.

Usage:
    python scripts/benchmark_inference.py \
        --checkpoint models/registry/MPSLightUNet_Halfmile_best.pt \
        --n-iterations 20
"""

import os
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deployment import Predictor
from src.utils.logger import setup_logger


# ============================================================
# BENCHMARK
# ============================================================

def time_function(fn, n_warmup: int = 2, n_iter: int = 20) -> dict:
    """
    Time a function over n_iter calls, with n_warmup discarded warm-ups.

    Returns:
        {"min_ms": float, "median_ms": float, "mean_ms": float}
    """
    # Warm-up
    for _ in range(n_warmup):
        fn()

    # Measure
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(times)
    return {
        "min_ms": float(arr.min()),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "p90_ms": float(np.percentile(arr, 90)),
        "n_iter": n_iter,
    }


def benchmark_predictor(
    checkpoint: str,
    device: str,
    n_iter: int,
    logger,
) -> dict:
    """Benchmark PyTorch inference via the Predictor."""
    logger.info("")
    logger.info(f"Loading model on {device}...")
    t0 = time.perf_counter()
    predictor = Predictor.load(checkpoint, device=device, logger=logger)
    load_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"  Load time: {load_ms:.1f} ms")

    # Create a shot with the correct shape
    shot = np.random.randn(
        predictor.contract.target_traces,
        predictor.contract.n_samples,
    ).astype(np.float32)

    # Warm-up + time
    logger.info(f"  Timing {n_iter} iterations (batch=1)...")
    stats = time_function(
        lambda: predictor.predict(shot),
        n_warmup=2,
        n_iter=n_iter,
    )

    return {
        "load_ms": load_ms,
        "inference": stats,
        "device": device,
        "model": checkpoint,
    }


def benchmark_torchscript(
    torchscript_path: str,
    contract_shape: tuple,
    device: str,
    n_iter: int,
    logger,
) -> dict | None:
    """Benchmark TorchScript inference."""
    if not Path(torchscript_path).exists():
        logger.info(f"  TorchScript file not found: {torchscript_path} — skipping")
        return None

    logger.info("")
    logger.info(f"Loading TorchScript on {device}...")
    t0 = time.perf_counter()
    model = torch.jit.load(torchscript_path, map_location=device)
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"  Load time: {load_ms:.1f} ms")

    x = torch.randn(*contract_shape).to(device)

    def run():
        with torch.no_grad():
            model(x)

    stats = time_function(run, n_warmup=2, n_iter=n_iter)
    return {"load_ms": load_ms, "inference": stats}


def benchmark_onnx(
    onnx_path: str,
    contract_shape: tuple,
    n_iter: int,
    logger,
) -> dict | None:
    """Benchmark ONNX inference."""
    if not Path(onnx_path).exists():
        logger.info(f"  ONNX file not found: {onnx_path} — skipping")
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("  onnxruntime not installed — skipping ONNX benchmark")
        return None

    logger.info("")
    logger.info("Loading ONNX...")
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    load_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"  Load time: {load_ms:.1f} ms")

    x = np.random.randn(*contract_shape).astype(np.float32)
    input_name = session.get_inputs()[0].name

    def run():
        session.run(None, {input_name: x})

    stats = time_function(run, n_warmup=2, n_iter=n_iter)
    return {"load_ms": load_ms, "inference": stats}


# ============================================================
# MAIN
# ============================================================

@click.command()
@click.option("--checkpoint", "-c", required=True,
              help="Path to PyTorch checkpoint")
@click.option("--device", "-d", default="cpu",
              help="Device for PyTorch benchmark (cpu/cuda/mps)")
@click.option("--n-iterations", "-n", type=int, default=20,
              help="Number of timing iterations per backend")
@click.option("--torchscript", type=str, default=None,
              help="Optional TorchScript path to benchmark")
@click.option("--onnx", type=str, default=None,
              help="Optional ONNX path to benchmark")
def main(
    checkpoint: str,
    device: str,
    n_iterations: int,
    torchscript: str | None,
    onnx: str | None,
) -> None:
    """Benchmark inference latency for PyTorch, TorchScript, and ONNX."""
    logger = setup_logger(task_name="benchmark_inference")

    logger.info("=" * 70)
    logger.info("INFERENCE LATENCY BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info(f"  Device:     {device}")
    logger.info(f"  Iterations: {n_iterations}")

    # --- PyTorch ---
    pytorch_result = benchmark_predictor(
        checkpoint=checkpoint,
        device=device,
        n_iter=n_iterations,
        logger=logger,
    )
    contract_shape = (1, 1,
                      int(round(pytorch_result["inference"]["n_iter"] * 0)),
                      0)  # placeholder, we'll fix below
    # Actually re-read the contract
    from src.deployment import read_contract_from_checkpoint
    contract = read_contract_from_checkpoint(checkpoint)

    # --- TorchScript ---
    ts_result = None
    if torchscript is not None:
        ts_result = benchmark_torchscript(
            torchscript_path=torchscript,
            contract_shape=contract.expected_shape,
            device=device,
            n_iter=n_iterations,
            logger=logger,
        )

    # --- ONNX ---
    onnx_result = None
    if onnx is not None:
        onnx_result = benchmark_onnx(
            onnx_path=onnx,
            contract_shape=contract.expected_shape,
            n_iter=n_iterations,
            logger=logger,
        )

    # --- Summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  {'Backend':<14} {'Load (ms)':<12} "
                f"{'Median (ms)':<14} {'Min (ms)':<12} {'p90 (ms)':<12}")
    logger.info("-" * 70)

    p = pytorch_result["inference"]
    logger.info(
        f"  {'PyTorch':<14} {pytorch_result['load_ms']:<12.1f} "
        f"{p['median_ms']:<14.2f} {p['min_ms']:<12.2f} {p['p90_ms']:<12.2f}"
    )

    if ts_result is not None:
        t = ts_result["inference"]
        logger.info(
            f"  {'TorchScript':<14} {ts_result['load_ms']:<12.1f} "
            f"{t['median_ms']:<14.2f} {t['min_ms']:<12.2f} {t['p90_ms']:<12.2f}"
        )

    if onnx_result is not None:
        o = onnx_result["inference"]
        logger.info(
            f"  {'ONNX':<14} {onnx_result['load_ms']:<12.1f} "
            f"{o['median_ms']:<14.2f} {o['min_ms']:<12.2f} {o['p90_ms']:<12.2f}"
        )

    logger.info("")
    logger.info(
        f"  Throughput (PyTorch): "
        f"{1000 / p['median_ms']:.2f} shots/second"
    )


if __name__ == "__main__":
    main()