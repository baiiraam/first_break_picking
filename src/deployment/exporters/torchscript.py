# file location: src/deployment/exporters/torchscript.py

"""
TorchScript export.
"""

from pathlib import Path

import torch

from src.deployment.contract import InputContract
from src.types import LoggerType


def export_torchscript(
    model: torch.nn.Module,
    contract: InputContract,
    output_path: str | Path,
    example_input: torch.Tensor | None = None,
    logger: LoggerType | None = None,
) -> Path:
    """
    Export a model to TorchScript.

    Args:
        model: PyTorch model, in eval mode
        contract: the model's InputContract (for the example input)
        output_path: where to save the .pt file
        example_input: optional example input; if None, one is
                       created from the contract
        logger: optional logger for progress messages

    Returns:
        Path to the saved TorchScript file.
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

    # Move to model's device (usually CPU for export)
    device = next(model.parameters()).device
    model = model.to(device)
    example_input = example_input.to(device)

    scripted = torch.jit.trace(model, example_input)
    torch.jit.save(scripted, output_path)

    if logger:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✅ TorchScript saved: {output_path} ({size_mb:.2f} MB)")

    return output_path