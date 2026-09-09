# src/utils/device_utils.py
"""
Device and memory detection utilities.
"""

import platform
import sys
from typing import Any

import psutil
import torch
from loguru import logger


def get_cpu_memory_info() -> dict[str, float]:
    """Get CPU/RAM memory information."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent": mem.percent,
        "free_gb": mem.free / (1024**3),
    }


def get_cuda_memory_info() -> dict[str, Any] | None:
    """Get CUDA GPU memory information."""
    if not torch.cuda.is_available():
        return None

    try:
        device_count = torch.cuda.device_count()
        devices = []

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            free = total_memory - allocated

            devices.append(
                {
                    "device_id": i,
                    "name": props.name,
                    "total_gb": total_memory,
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "free_gb": free,
                    "percent_used": (allocated / total_memory) * 100
                    if total_memory > 0
                    else 0,
                }
            )

        return {"device_count": device_count, "devices": devices}

    except (RuntimeError, AttributeError, ValueError) as e:
        logger.warning(f"Could not get CUDA memory info: {e}")
        return None


def get_mps_memory_info() -> dict[str, float] | None:
    """Get MPS (Apple Silicon) memory information."""
    if not torch.backends.mps.is_available():
        return None

    try:
        mem = psutil.virtual_memory()
        system_ram_gb = mem.total / (1024**3)
        mps_limit = min(system_ram_gb * 0.75, 16.0)

        allocated = 0.0
        if hasattr(torch.mps, "current_allocated_memory"):
            allocated = torch.mps.current_allocated_memory() / (1024**3)

        driver_allocated = 0.0
        if hasattr(torch.mps, "driver_allocated_memory"):
            driver_allocated = torch.mps.driver_allocated_memory() / (1024**3)

        return {
            "system_ram_gb": system_ram_gb,
            "available_gb": mem.available / (1024**3),
            "mps_limit_gb": mps_limit,
            "allocated_gb": allocated,
            "driver_allocated_gb": driver_allocated,
            "percent_used": (allocated / mps_limit) * 100 if mps_limit > 0 else 0,
            "recommended_limit_gb": min(mps_limit * 0.8, 14.0),
            "max_safe_gb": min(mps_limit * 0.9, 15.0),
            "aggressive_limit_gb": min(mps_limit * 0.95, 16.0),
        }

    except (RuntimeError, AttributeError, ValueError) as e:
        logger.warning(f"Could not get MPS memory info: {e}")
        return None


def get_device_info() -> dict[str, Any]:
    """Get complete device and memory information."""
    return {
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {"version": sys.version},
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "cpu": get_cpu_memory_info(),
        "cuda": get_cuda_memory_info() if torch.cuda.is_available() else None,
        "mps": get_mps_memory_info() if torch.backends.mps.is_available() else None,
    }


# src/utils/device_utils.py
# Fix the type checking

# src/utils/device_utils.py - lines 190-210


def get_recommended_memory_limits(
    info: dict[str, Any],
) -> dict[str, dict[str, float] | None]:
    """Get recommended memory limits for each device type."""
    recommendations: dict[str, dict[str, float] | None] = {}

    if info.get("cpu"):
        cpu = info["cpu"]
        if cpu is not None:  # ✅ Add None check
            available_gb: float = float(cpu.get("available_gb", 0.0))
            recommendations["cpu"] = {
                "recommended_gb": available_gb * 0.7,
                "max_gb": available_gb * 0.85,
            }
        else:
            recommendations["cpu"] = None
    else:
        recommendations["cpu"] = None

    # ✅ Fix: Check if cuda exists AND has devices AND is not None
    cuda_info = info.get("cuda")
    if cuda_info is not None and cuda_info.get("devices"):
        device = cuda_info["devices"][0]
        total_gb: float = float(device.get("total_gb", 0.0))
        recommendations["cuda"] = {
            "recommended_gb": total_gb * 0.8,
            "max_gb": total_gb * 0.9,
        }
    else:
        recommendations["cuda"] = None

    # ✅ Fix: Check if mps exists AND is not None
    mps_info = info.get("mps")
    if mps_info is not None:
        recommendations["mps"] = {
            "recommended_gb": float(mps_info.get("recommended_limit_gb", 8.0)),
            "max_gb": float(mps_info.get("max_safe_gb", 10.0)),
        }
    else:
        recommendations["mps"] = None

    return recommendations


# src/utils/device_utils.py


def detect_device() -> dict[str, Any]:
    """
    Detect the best available device and its memory.

    Returns:
        dict with keys: device_type, available_gb, device_name, device_memory_gb
    """
    info = get_device_info()
    recommendations = get_recommended_memory_limits(info) or {}

    device_info = {
        "device_type": "cpu",
        "available_gb": 8.0,
        "device_name": "CPU",
        "device_memory_gb": info["cpu"]["total_gb"],
    }

    # ✅ Fix: Check if cuda is available AND recommendations["cuda"] is not None
    if info["pytorch"]["cuda_available"]:
        cuda_rec = recommendations.get("cuda")
        if cuda_rec is not None:
            device_info.update(
                {
                    "device_type": "cuda",
                    "available_gb": cuda_rec["recommended_gb"],
                    "device_name": info["cuda"]["devices"][0]["name"],
                    "device_memory_gb": info["cuda"]["devices"][0]["total_gb"],
                }
            )
    # ✅ Fix: Check if mps is available AND recommendations["mps"] is not None
    elif info["pytorch"]["mps_available"]:
        mps_rec = recommendations.get("mps")
        if mps_rec is not None:
            device_info.update(
                {
                    "device_type": "mps",
                    "available_gb": mps_rec["recommended_gb"],
                    "device_name": "Apple Silicon (MPS)",
                    "device_memory_gb": info["mps"]["system_ram_gb"],
                }
            )

    return device_info
