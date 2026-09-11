# file location: src/deployment/exporters/onnx.py

"""
ONNX export.
"""

from pathlib import Path

import torch

from src.deployment.contract import InputContract
from src.types import LoggerType


def export_onnx(
    model: torch.nn.Module,
    contract: InputContract,
    output_path: str | Path,
    example_input: torch.Tensor | None = None,
    opset_version: int = 18,
    dynamic_axes: dict | None = None,
    logger: LoggerType | None = None,
) -> Path:
    """
    Export a model to ONNX.

    Args:
        model: PyTorch model, in eval mode
        contract: the model's InputContract
        output_path: where to save the .onnx file
        example_input: optional example input
        opset_version: ONNX opset version (default 18)
        dynamic_axes: optional dynamic axis specification
        logger: optional logger for progress messages

    Returns:
        Path to the saved ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if example_input is None:
        example_input = torch.randn(*contract.expected_shape)
        if logger:
            logger.info(
                f"  Created example input: {tuple(example_input.shape)}"
            )

    device = next(model.parameters()).device
    model = model.to(device)
    example_input = example_input.to(device)

    # Default dynamic axes: batch dimension only
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        (example_input,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )

    if logger:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✅ ONNX saved: {output_path} ({size_mb:.2f} MB)")

    return output_path