# file location: src/training/utils/device.py

"""
Device setup and hardware warmup utilities.
"""

import torch
from torch import nn

from src.config import SeismicConfig
from src.types import LoggerType


def setup_device(requested_device: str, logger: LoggerType) -> torch.device:
    """
    Setup device with fallback handling.

    Args:
        requested_device: 'cpu', 'cuda', or 'mps'
        logger: logger instance

    Returns:
        torch.device: Configured device
    """
    if requested_device == "mps":
        if torch.backends.mps.is_available():
            try:
                test = torch.ones(1, device="mps")
                del test
                logger.info("MPS device initialized successfully")
                return torch.device("mps")
            except (
                RuntimeError,
                AttributeError,
                ValueError,
            ) as e:  # ✅ Specific exceptions
                logger.warning(f"MPS initialization failed: {e}, falling back to CPU")
                return torch.device("cpu")
        else:
            logger.warning("MPS not available, falling back to CPU")
            return torch.device("cpu")

    elif requested_device == "cuda":
        if torch.cuda.is_available():
            logger.info(f"CUDA device initialized: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")

    else:
        return torch.device("cpu")


def prepare_model(
    model: nn.Module,
    config: SeismicConfig,
    device: torch.device,
    logger: LoggerType,
) -> nn.Module:
    """
    Prepare model for training (DataParallel, device placement).

    Args:
        model: PyTorch model
        config: Configuration object
        device: Target device
        logger: logger instance

    Returns:
        Prepared model on correct device
    """
    if config.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
        logger.info(f"Multi-GPU enabled: {torch.cuda.device_count()} GPUs")
    else:
        logger.info(f"Single device: {device}")

    return model.to(device)


def warmup_mps_device(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    logger: LoggerType,
) -> None:
    """
    Warm up MPS shaders to avoid JIT compilation delay during training.

    Args:
        model: PyTorch model
        criterion: Loss function
        device: Target device (must be MPS)
        logger: logger instance
    """
    if device.type != "mps":
        return

    logger.info("🔥 Warming up MPS shaders (first pass can take 2-10 minutes)...")

    # Create dummy data with production shape
    dummy_x = torch.randn(1, 1, 1578, 751, device=device)
    dummy_y = torch.randint(0, 3, (1, 1578, 751), device=device)

    # Forward pass
    dummy_out = model(dummy_x)

    # Loss
    dummy_loss = criterion(dummy_out, dummy_y)

    # Backward pass (this triggers shader compilation)
    dummy_loss.backward()

    # Synchronize to ensure compilation completes
    torch.mps.synchronize()

    # Clear gradients and memory
    model.zero_grad()
    torch.mps.empty_cache()

    logger.info("✅ MPS warmup complete!")


def get_memory_usage(device: torch.device) -> dict[str, float]:
    """
    Get current memory usage for the device.

    Args:
        device: Target device

    Returns:
        Dictionary with memory usage in GB
    """
    memory = {}
    if device.type == "mps":
        memory["allocated"] = torch.mps.current_allocated_memory() / 1e9
        memory["max_allocated"] = torch.mps.driver_allocated_memory() / 1e9
    elif device.type == "cuda":
        memory["allocated"] = torch.cuda.memory_allocated() / 1e9
        memory["reserved"] = torch.cuda.memory_reserved() / 1e9
        memory["max_allocated"] = torch.cuda.max_memory_allocated() / 1e9
    return memory
