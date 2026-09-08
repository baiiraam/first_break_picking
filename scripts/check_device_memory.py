#!/usr/bin/env python3
"""
Device Memory Detection Script - Complete Training Recommendations
Detects memory and recommends optimal batch_size, cache_size, and memory limits.
"""

import os
import platform
import sys
from typing import Any

import psutil
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger
from src.utils.model_registry import get_model_registry

logger = setup_logger(task_name="check_device_memory")


# ============================================================
# DEVICE MEMORY DETECTION
# ============================================================


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

    except (RuntimeError, AttributeError, ValueError) as e:  # ✅ Fixed
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

    except (RuntimeError, AttributeError, ValueError) as e:  # ✅ Fixed
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


def get_recommended_memory_limits(
    info: dict[str, Any],
) -> dict[str, dict[str, float] | None]:
    recommendations: dict[str, dict[str, float] | None] = {}

    if info.get("cpu"):
        cpu = info["cpu"]
        available_gb: float = float(cpu.get("available_gb", 0.0))
        recommendations["cpu"] = {
            "recommended_gb": available_gb * 0.7,
            "max_gb": available_gb * 0.85,
        }
    else:
        recommendations["cpu"] = None

    if info.get("cuda") and info["cuda"].get("devices"):
        device = info["cuda"]["devices"][0]
        total_gb: float = float(device.get("total_gb", 0.0))
        recommendations["cuda"] = {
            "recommended_gb": total_gb * 0.8,
            "max_gb": total_gb * 0.9,
        }
    else:
        recommendations["cuda"] = None

    if info.get("mps"):
        mps = info["mps"]
        recommendations["mps"] = {
            "recommended_gb": float(mps.get("recommended_limit_gb", 8.0)),
            "max_gb": float(mps.get("max_safe_gb", 10.0)),
        }
    else:
        recommendations["mps"] = None

    return recommendations


# ============================================================
# MODEL CONFIGURATION (Uses registry)
# ============================================================


def calculate_optimal_config(
    model_name: str,
    available_memory_gb: float,
    device_type: str,
) -> dict:
    """Calculate optimal batch_size, cache_size, and memory_limit."""

    registry = get_model_registry()
    profile = registry.get(model_name)

    if not profile:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {registry.get_model_names()}"
        )

    available_mb = available_memory_gb * 1024
    base_memory_mb = profile.base_memory_mb
    remaining_mb = available_mb - base_memory_mb
    safe_remaining_mb = remaining_mb * 0.8

    # 🆕 Removed unused variables
    # dataset_factor, total_shots, dataset_size_label were never used

    memory_per_batch_mb = profile.memory_per_batch_mb
    memory_per_cache_mb = profile.memory_per_cache_mb

    recommended_batch = profile.recommended_batch_size
    recommended_cache = profile.recommended_cache_size

    if memory_per_batch_mb > 0:
        max_possible_batch = int(safe_remaining_mb / memory_per_batch_mb)
    else:
        max_possible_batch = recommended_batch

    optimal_batch = min(recommended_batch, max(1, max_possible_batch))

    batch_memory_mb = optimal_batch * memory_per_batch_mb
    remaining_after_batch_mb = safe_remaining_mb - batch_memory_mb

    if memory_per_cache_mb > 0:
        max_possible_cache = int(remaining_after_batch_mb / memory_per_cache_mb)
    else:
        max_possible_cache = recommended_cache

    optimal_cache = min(recommended_cache, max(1, max_possible_cache))

    cache_memory_mb = optimal_cache * memory_per_cache_mb

    total_memory_mb = base_memory_mb + batch_memory_mb + cache_memory_mb

    if device_type == "mps":
        overhead_factor = 1.5
    elif device_type == "cuda":
        overhead_factor = 1.3
    else:
        overhead_factor = 1.2

    total_memory_gb = (total_memory_mb / 1024) * overhead_factor
    total_memory_gb = round(total_memory_gb * 2) / 2
    memory_limit_gb = max(0.5, total_memory_gb)

    can_fit = total_memory_gb < available_memory_gb * 0.85

    if total_memory_gb < available_memory_gb * 0.6:
        confidence = "high"
    elif total_memory_gb < available_memory_gb * 0.75:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "model": model_name,
        "params": profile.params,
        "device_type": device_type,
        "available_memory_gb": available_memory_gb,
        "optimal_batch_size": optimal_batch,
        "optimal_cache_size": optimal_cache,
        "recommended_memory_limit_gb": memory_limit_gb,
        "total_memory_mb": int(total_memory_mb),
        "base_memory_mb": base_memory_mb,
        "batch_memory_mb": batch_memory_mb,
        "cache_memory_mb": cache_memory_mb,
        "can_fit": can_fit,
        "confidence": confidence,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    """Main function."""
    info = get_device_info()

    print("=" * 80)
    print("🔍 DEVICE & MEMORY INFORMATION")
    print("=" * 80)

    if info["pytorch"]["cuda_available"] and info["cuda"]:
        device_type = "cuda"
        available_gb = info["cuda"]["devices"][0]["total_gb"] * 0.85
        print(f"\n🎯 Using CUDA device: {info['cuda']['devices'][0]['name']}")
    elif info["pytorch"]["mps_available"] and info["mps"]:
        device_type = "mps"
        available_gb = info["mps"]["recommended_limit_gb"]
        print("\n🍏 Using MPS device (Apple Silicon)")
    else:
        device_type = "cpu"
        available_gb = info["cpu"]["available_gb"] * 0.7
        print("\n💻 Using CPU")

    print(f"📊 Available memory for training: {available_gb:.1f} GB")

    # 🆕 Simplified: just show recommendations
    registry = get_model_registry()
    model_order = [
        "pico",
        "nano",
        "tiny",
        "mpslight",
        "light",
        "mobile",
        "efficient",
        "unet",
    ]

    print("\n📊 MODEL MEMORY RECOMMENDATIONS")
    print("=" * 80)

    for model_name in model_order:
        if registry.get(model_name):
            try:
                config = calculate_optimal_config(model_name, available_gb, device_type)
                status = "✅" if config["can_fit"] else "⚠️"
                print(
                    f"{status} {model_name:10} -> batch: {config['optimal_batch_size']:2}, "
                    f"cache: {config['optimal_cache_size']:2}, "
                    f"memory: {config['recommended_memory_limit_gb']:.1f}GB"
                )
            except (ValueError, KeyError, TypeError) as e:  # ✅ Fixed
                logger.warning(f"Could not calculate for {model_name}: {e}")

    print("\n✅ Done!")


if __name__ == "__main__":
    result = main()
