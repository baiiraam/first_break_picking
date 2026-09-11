# file location: src/deployment/validators.py

"""
Numeric equivalence checks for exported models.

Asserts that PyTorch, TorchScript, and ONNX produce the same output
(within tolerance) on the same input. Without this, an exported
model could silently differ in production.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch


def verify_numeric_equivalence(
    pytorch_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    torchscript_path: str | Path | None = None,
    onnx_path: str | Path | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, dict[str, Any]]:
    """
    Compare PyTorch output to TorchScript and ONNX on the same input.

    Args:
        pytorch_model: the source model (eval mode)
        input_tensor: (1, C, H, W) input to compare on
        torchscript_path: optional path to a .pt TorchScript file
        onnx_path: optional path to an .onnx file
        rtol: relative tolerance for allclose
        atol: absolute tolerance for allclose

    Returns:
        Dict with one entry per exported model:
            {
              "torchscript": {
                "match": bool,
                "max_diff": float,
                "shape": tuple,
                "error": str | None,
              },
              "onnx": {...},
            }
        Missing paths are simply omitted.
    """
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device

    with torch.no_grad():
        pytorch_out = pytorch_model(input_tensor.to(device))
    pytorch_np = pytorch_out.cpu().numpy()

    results: dict[str, dict[str, Any]] = {}

    # ---- TorchScript ----
    if torchscript_path is not None:
        path = Path(torchscript_path)
        if not path.exists():
            results["torchscript"] = {
                "match": False,
                "max_diff": float("inf"),
                "shape": None,
                "error": f"File not found: {path}",
            }
        else:
            try:
                scripted = torch.jit.load(str(path), map_location=device)
                scripted.eval()
                with torch.no_grad():
                    ts_out = scripted(input_tensor.to(device))
                ts_np = ts_out.cpu().numpy()

                if ts_np.shape != pytorch_np.shape:
                    results["torchscript"] = {
                        "match": False,
                        "max_diff": float("inf"),
                        "shape": tuple(ts_np.shape),
                        "error": f"Shape mismatch: pytorch={pytorch_np.shape}, "
                                 f"torchscript={ts_np.shape}",
                    }
                else:
                    max_diff = float(np.max(np.abs(ts_np - pytorch_np)))
                    match = bool(
                        np.allclose(ts_np, pytorch_np, rtol=rtol, atol=atol)
                    )
                    results["torchscript"] = {
                        "match": match,
                        "max_diff": max_diff,
                        "shape": tuple(ts_np.shape),
                        "error": None,
                    }
            except Exception as e:  # noqa: BLE001
                results["torchscript"] = {
                    "match": False,
                    "max_diff": float("inf"),
                    "shape": None,
                    "error": f"{type(e).__name__}: {e}",
                }

    # ---- ONNX ----
    if onnx_path is not None:
        path = Path(onnx_path)
        if not path.exists():
            results["onnx"] = {
                "match": False,
                "max_diff": float("inf"),
                "shape": None,
                "error": f"File not found: {path}",
            }
        else:
            try:
                import onnxruntime as ort

                session = ort.InferenceSession(
                    str(path),
                    providers=["CPUExecutionProvider"],
                )
                ort_inputs = {
                    session.get_inputs()[0].name: input_tensor.cpu().numpy()
                }
                ort_out = session.run(None, ort_inputs)[0]

                if ort_out.shape != pytorch_np.shape:
                    results["onnx"] = {
                        "match": False,
                        "max_diff": float("inf"),
                        "shape": tuple(ort_out.shape),
                        "error": f"Shape mismatch: pytorch={pytorch_np.shape}, "
                                 f"onnx={ort_out.shape}",
                    }
                else:
                    max_diff = float(np.max(np.abs(ort_out - pytorch_np)))
                    match = bool(
                        np.allclose(ort_out, pytorch_np, rtol=rtol, atol=atol)
                    )
                    results["onnx"] = {
                        "match": match,
                        "max_diff": max_diff,
                        "shape": tuple(ort_out.shape),
                        "error": None,
                    }
            except ImportError as e:
                results["onnx"] = {
                    "match": False,
                    "max_diff": float("inf"),
                    "shape": None,
                    "error": f"onnxruntime not installed: {e}",
                }
            except Exception as e:  # noqa: BLE001
                results["onnx"] = {
                    "match": False,
                    "max_diff": float("inf"),
                    "shape": None,
                    "error": f"{type(e).__name__}: {e}",
                }

    return results