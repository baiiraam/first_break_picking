#!/usr/bin/env python3
"""
Device Memory Detection Script - Complete Training Recommendations
Detects memory and recommends optimal batch_size, cache_size, and memory limits.
"""

import json
import os
import platform
import sys
from typing import Any

import psutil
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger
from src.utils.memory import MODEL_PROFILES

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


def get_cuda_memory_info() -> dict[str, float] | None:
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

        return {
            "device_count": device_count,
            "devices": devices,
        }

    except Exception as e:  # noqa: BLE001
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

        allocated = 0
        if hasattr(torch.mps, "current_allocated_memory"):
            allocated = torch.mps.current_allocated_memory() / (1024**3)

        driver_allocated = 0
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

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not get MPS memory info: {e}")
        return None


def get_device_info() -> dict:
    """Get complete device and memory information."""

    info = {
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "cpu": get_cpu_memory_info(),
        "cuda": get_cuda_memory_info() if torch.cuda.is_available() else None,
        "mps": get_mps_memory_info() if torch.backends.mps.is_available() else None,
    }

    return info


# ============================================================
# RECOMMENDATIONS (ADD THESE FUNCTIONS)
# ============================================================


def get_recommended_memory_limits(info: dict) -> dict:
    """Get recommended memory limits for different devices."""

    recommendations = {}

    # CPU recommendation
    if info.get("cpu"):
        cpu = info["cpu"]
        available_gb = cpu.get("available_gb", 0)
        recommendations["cpu"] = {
            "recommended_gb": available_gb * 0.7,
            "max_gb": available_gb * 0.85,
        }
    else:
        recommendations["cpu"] = None

    # CUDA recommendation
    if info.get("cuda") and info["cuda"].get("devices"):
        device = info["cuda"]["devices"][0]
        total_gb = device.get("total_gb", 0)
        recommendations["cuda"] = {
            "recommended_gb": total_gb * 0.8,
            "max_gb": total_gb * 0.9,
        }
    else:
        recommendations["cuda"] = None

    # MPS recommendation
    if info.get("mps"):
        mps = info["mps"]
        recommendations["mps"] = {
            "recommended_gb": mps.get("recommended_limit_gb", 8.0),
            "max_gb": mps.get("max_safe_gb", 10.0),
        }
    else:
        recommendations["mps"] = None

    return recommendations

# ============================================================
# CONFIG VARIANT GENERATION
# ============================================================


def calculate_optimal_config(
    model_name: str,
    available_memory_gb: float,
    device_type: str,
) -> dict:
    """Calculate optimal batch_size, cache_size, and memory_limit."""

    profile = MODEL_PROFILES.get(model_name)
    if not profile:
        raise ValueError(f"Unknown model: {model_name}")

    available_mb = available_memory_gb * 1024
    base_memory_mb = profile.base_memory_mb
    remaining_mb = available_mb - base_memory_mb
    safe_remaining_mb = remaining_mb * 0.8

    # ============================================================
    # DEFINE THESE VARIABLES FIRST
    # ============================================================
    # Dataset factor (for now, use default since we don't have dataset info here)
    dataset_factor = 1.0
    total_shots = 142  # Default for Halfmile
    dataset_size_label = "medium"

    # Memory per batch and cache
    memory_per_batch_mb = profile.memory_per_batch_mb
    memory_per_cache_mb = profile.memory_per_cache_mb

    # Recommended values from profile
    recommended_batch = profile.recommended_batch_size
    recommended_cache = profile.recommended_cache_size

    # Calculate max possible
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
        "calculations": {
            "base_memory": {
                "value_mb": base_memory_mb,
                "value_gb": base_memory_mb / 1024,
                "description": f"Base memory for {model_name} model ({profile.params:,} params)",
            },
            "safe_remaining": {
                "value_mb": safe_remaining_mb,
                "value_gb": safe_remaining_mb / 1024,
                "description": "Memory available after base and 20% overhead",
            },
            "dataset_factor": {
                "value": dataset_factor,
                "description": f"Dataset size: {dataset_size_label} ({total_shots} shots)",
            },
            "optimal_batch": {
                "value": optimal_batch,
                "description": f"max_by_memory={max_possible_batch}, recommended={recommended_batch}, factor={dataset_factor:.1f}",
            },
            "batch_memory": {
                "value_mb": batch_memory_mb,
                "value_gb": batch_memory_mb / 1024,
                "description": f"{optimal_batch} batches × {memory_per_batch_mb}MB/batch",
            },
            "optimal_cache": {
                "value": optimal_cache,
                "description": f"max_by_memory={max_possible_cache}, recommended={recommended_cache}, factor={1.0:.1f}",
            },
            "cache_memory": {
                "value_mb": cache_memory_mb,
                "value_gb": cache_memory_mb / 1024,
                "description": f"{optimal_cache} caches × {memory_per_cache_mb}MB/cache",
            },
            "total_memory": {
                "value_mb": total_memory_mb,
                "value_gb": total_memory_mb / 1024,
                "description": f"base={base_memory_mb}MB + batch={batch_memory_mb}MB + cache={cache_memory_mb}MB",
            },
            "overhead_factor": {
                "value": overhead_factor,
                "description": f"Device type: {device_type.upper()} overhead",
            },
            "final_memory_limit": {
                "value_gb": memory_limit_gb,
                "description": f"total_memory × {overhead_factor} = {memory_limit_gb:.1f}GB",
            },
        },
    }


def get_all_model_recommendations(
    available_memory_gb: float,
    device_type: str,
) -> dict[str, dict]:
    """Get recommendations for all models."""

    recommendations = {}

    for model_name in MODEL_PROFILES:
        try:
            rec = calculate_optimal_config(
                model_name,
                available_memory_gb,
                device_type,
            )
            recommendations[model_name] = rec
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not calculate for {model_name}: {e}")

    return recommendations


def generate_config_variants_from_recommendations(
    model_name: str,
    recommendations: dict,
) -> list[dict]:
    """Generate progressive config variants from recommendations."""

    if model_name not in recommendations:
        return []

    rec = recommendations[model_name]

    variants = []

    if rec["params"] > 1000000:
        class_weights = "0.1,0.1,0.8"
    else:
        class_weights = "0.2,0.2,0.6"

    # Level 1: Optimal
    variants.append(
        {
            "batch_size": rec["optimal_batch_size"],
            "model": model_name,
            "cache_size": rec["optimal_cache_size"],
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": rec["recommended_memory_limit_gb"],
        }
    )

    # Level 2: Reduced batch
    if rec["optimal_batch_size"] > 2:
        variants.append(
            {
                "batch_size": max(1, rec["optimal_batch_size"] - 2),
                "model": model_name,
                "cache_size": rec["optimal_cache_size"],
                "class_weights": class_weights,
                "strip_width": 8,
                "memory_limit_gb": max(0.5, rec["recommended_memory_limit_gb"] * 0.8),
            }
        )

    # Level 3: Reduced both
    if rec["optimal_batch_size"] > 2 and rec["optimal_cache_size"] > 1:
        variants.append(
            {
                "batch_size": max(1, rec["optimal_batch_size"] - 2),
                "model": model_name,
                "cache_size": max(1, rec["optimal_cache_size"] - 1),
                "class_weights": class_weights,
                "strip_width": 8,
                "memory_limit_gb": max(0.5, rec["recommended_memory_limit_gb"] * 0.6),
            }
        )

    # Level 4: Minimal
    variants.append(
        {
            "batch_size": 1,
            "model": model_name,
            "cache_size": 1,
            "class_weights": "0.2,0.2,0.6"
            if rec["params"] > 1000000
            else class_weights,
            "strip_width": 8,
            "memory_limit_gb": max(1.0, rec["recommended_memory_limit_gb"] * 0.4),
        }
    )

    return variants


def generate_all_variants(
    available_memory_gb: float,
    device_type: str,
) -> dict[str, list[dict]]:
    """Generate config variants for all models."""

    recommendations = get_all_model_recommendations(available_memory_gb, device_type)

    all_variants = {}
    for model_name in recommendations:
        variants = generate_config_variants_from_recommendations(
            model_name, recommendations
        )
        if variants:
            all_variants[model_name] = variants

    return all_variants


def generate_aggressive_variants_for_model(
    model_name: str,
    available_memory_gb: float,
    device_type: str,
    max_batch_size: int = 8,
    max_cache_size: int = 5,
) -> list[dict]:
    """
    Generate aggressive config variants that start high and reduce on failure.
    """

    profile = MODEL_PROFILES.get(model_name)
    if not profile:
        return []

    # Determine class weights based on model size
    if profile.params > 1000000:
        class_weights = "0.1,0.1,0.8"
    else:
        class_weights = "0.2,0.2,0.6"

    # Start with aggressive values
    variants = []

    # Level 1: Most aggressive (high batch, high cache, high memory)
    variants.append(
        {
            "batch_size": min(max_batch_size, profile.recommended_batch_size + 2),
            "model": model_name,
            "cache_size": min(max_cache_size, profile.recommended_cache_size + 2),
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": min(
                available_memory_gb * 0.9, 14.0
            ),  # Use 90% of available
        }
    )

    # Level 2: Reduce batch and cache slightly
    variants.append(
        {
            "batch_size": min(max_batch_size, profile.recommended_batch_size + 1),
            "model": model_name,
            "cache_size": min(max_cache_size, profile.recommended_cache_size + 1),
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": min(available_memory_gb * 0.75, 12.0),
        }
    )

    # Level 3: Recommended settings
    variants.append(
        {
            "batch_size": profile.recommended_batch_size,
            "model": model_name,
            "cache_size": profile.recommended_cache_size,
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": min(available_memory_gb * 0.6, 10.0),
        }
    )

    # Level 4: Conservative
    variants.append(
        {
            "batch_size": max(1, profile.recommended_batch_size - 2),
            "model": model_name,
            "cache_size": max(1, profile.recommended_cache_size - 2),
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": min(available_memory_gb * 0.4, 6.0),
        }
    )

    # Level 5: Minimal (last resort)
    variants.append(
        {
            "batch_size": 1,
            "model": model_name,
            "cache_size": 1,
            "class_weights": "0.2,0.2,0.6",
            "strip_width": 8,
            "memory_limit_gb": min(available_memory_gb * 0.25, 4.0),
        }
    )

    return variants


def generate_nonlinear_variants(
    model_name: str,
    available_memory_gb: float,
    device_type: str,
) -> list[dict]:
    """
    Generate variants with non-linear halving strategy.
    Explores different combinations of batch_size and cache_size.
    """

    profile = MODEL_PROFILES.get(model_name)
    if not profile:
        return []

    # Determine class weights based on model size
    if profile.params > 1000000:
        class_weights = "0.1,0.1,0.8"
    else:
        class_weights = "0.2,0.2,0.6"

    variants = []

    # Memory limits for each level (start high, reduce)
    memory_levels = [
        min(available_memory_gb * 0.95, 14.0),  # Level 1
        min(available_memory_gb * 0.85, 12.0),  # Level 2
        min(available_memory_gb * 0.75, 10.0),  # Level 3
        min(available_memory_gb * 0.65, 8.0),  # Level 4
        min(available_memory_gb * 0.50, 6.0),  # Level 5
        min(available_memory_gb * 0.35, 4.0),  # Level 6
        min(available_memory_gb * 0.25, 3.0),  # Level 7
    ]

    # Your proposed progression
    batch_sizes = [64, 32, 32, 32, 16, 16, 16, 8, 8, 4, 4, 2, 1]
    cache_sizes = [64, 64, 32, 16, 32, 16, 8, 16, 8, 8, 4, 4, 2]

    # For different model sizes, adjust the starting values
    if profile.params < 100000:  # Pico, Nano, Tiny
        batch_sizes = [64, 64, 32, 32, 32, 16, 16, 16, 8, 8, 4, 2, 1]
        cache_sizes = [64, 32, 64, 32, 16, 32, 16, 8, 16, 8, 8, 4, 2]
    elif profile.params < 2000000:  # MPSLight
        batch_sizes = [48, 48, 24, 24, 24, 12, 12, 12, 6, 6, 3, 2, 1]
        cache_sizes = [48, 24, 48, 24, 12, 24, 12, 6, 12, 6, 6, 3, 2]
    elif profile.params < 5000000:  # Light, Mobile, Efficient
        batch_sizes = [32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 2, 2, 1]
        cache_sizes = [32, 16, 32, 16, 8, 16, 8, 4, 8, 4, 4, 2, 2]
    else:  # UNet
        batch_sizes = [16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 1, 1, 1]
        cache_sizes = [16, 8, 16, 8, 4, 8, 4, 2, 4, 2, 2, 1, 1]

    # Generate variants
    for i in range(len(batch_sizes)):
        # Get the memory level for this step
        mem_idx = min(i, len(memory_levels) - 1)

        variants.append(
            {
                "batch_size": max(1, batch_sizes[i]),
                "model": model_name,
                "cache_size": max(1, cache_sizes[i]),
                "class_weights": class_weights,
                "strip_width": 8,
                "memory_limit_gb": round(memory_levels[mem_idx], 1),
            }
        )

    # Remove duplicates
    seen = set()
    unique_variants = []
    for v in variants:
        key = (v["batch_size"], v["cache_size"], v["memory_limit_gb"])
        if key not in seen:
            seen.add(key)
            unique_variants.append(v)

    return unique_variants


# Add to check_device_memory.py


def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    """
    Get dataset characteristics from manifest.
    """
    import json
    from pathlib import Path

    manifest_path = Path(f"data/chunks/{dataset_name}/manifest.json")
    if not manifest_path.exists():
        return {
            "total_shots": 0,
            "total_traces": 0,
            "samples_per_trace": 0,
            "file_size_mb": 0,
            "chunk_size": 69,
        }

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Get config values
    config = manifest.get("config", {})

    return {
        "total_shots": manifest.get("total_shots", 0),
        "total_traces": config.get("target_traces", 0),
        "samples_per_trace": config.get("n_samples", 0),
        "file_size_mb": sum(
            c.get("file_size_mb", 0) for c in manifest.get("chunks", [])
        ),
        "chunk_size": config.get("chunk_size", 69),
        "num_chunks": len(manifest.get("chunks", [])),
    }


def get_actual_data_shape(dataset_name: str) -> tuple[int, int]:
    """
    Get the actual shape of the data for a dataset.
    """
    from pathlib import Path

    import torch

    chunk_dir = Path(f"data/chunks/{dataset_name}")
    if not chunk_dir.exists():
        return (0, 0)

    # Find first chunk file
    chunk_files = list(chunk_dir.glob("chunk_*.pt"))
    if not chunk_files:
        return (0, 0)

    try:
        chunk = torch.load(chunk_files[0], map_location="cpu", weights_only=True)
        data = chunk.get("data")
        if data is not None:
            # data shape: (n_shots, target_traces, n_samples)
            return (data.shape[1], data.shape[2])  # (traces, samples)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not load chunk for {dataset_name}: {e}")

    return (0, 0)


# Add to check_device_memory.py


def calculate_smart_config(
    model_name: str,
    dataset_name: str,
    available_memory_gb: float,
    device_type: str,
) -> dict[str, Any]:
    """
    Calculate optimal batch_size, cache_size, and memory_limit
    based on dataset characteristics and system resources.
    """

    profile = MODEL_PROFILES.get(model_name)
    if not profile:
        return {}

    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    _, _ = get_actual_data_shape(dataset_name)

    # Convert available memory to MB
    available_mb = available_memory_gb * 1024

    # Base memory for model
    base_memory_mb = profile.base_memory_mb

    # Memory available for batch and cache
    remaining_mb = available_mb - base_memory_mb

    # Safety margin (20% for PyTorch overhead)
    safe_remaining_mb = remaining_mb * 0.8

    # Calculate optimal batch size
    # Each batch uses: memory_per_batch_mb per shot
    # We want to maximize batch size while fitting in memory
    memory_per_batch_mb = profile.memory_per_batch_mb

    # Consider dataset size - larger datasets benefit from larger batches
    dataset_factor = 1.0
    if dataset_info["total_shots"] > 200:
        dataset_factor = 1.2  # More shots → larger batches
    elif dataset_info["total_shots"] < 50:
        dataset_factor = 0.8  # Fewer shots → smaller batches

    # Calculate max possible batch size
    max_batch_by_memory = (
        int(safe_remaining_mb / memory_per_batch_mb) if memory_per_batch_mb > 0 else 8
    )
    recommended_batch = profile.recommended_batch_size

    # Take the minimum of what fits and what's recommended
    optimal_batch = min(
        max(1, max_batch_by_memory), int(recommended_batch * dataset_factor)
    )

    # Ensure batch size doesn't exceed total shots
    if dataset_info["total_shots"] > 0:
        optimal_batch = min(optimal_batch, dataset_info["total_shots"])

    # Calculate memory used by batch
    batch_memory_mb = optimal_batch * memory_per_batch_mb

    # Calculate remaining memory for cache
    remaining_after_batch_mb = safe_remaining_mb - batch_memory_mb

    # Calculate optimal cache size
    # Cache stores chunks of data - each chunk has multiple shots
    memory_per_cache_mb = profile.memory_per_cache_mb

    # Cache size should be at least 1, but can be larger if memory allows
    max_cache_by_memory = (
        int(remaining_after_batch_mb / memory_per_cache_mb)
        if memory_per_cache_mb > 0
        else 3
    )
    recommended_cache = profile.recommended_cache_size

    # Cache size should be proportional to number of chunks
    num_chunks = dataset_info.get("num_chunks", 4)
    cache_factor = min(1.0, max(0.3, num_chunks / 10))  # More chunks → larger cache

    optimal_cache = min(
        max(1, max_cache_by_memory), int(recommended_cache * cache_factor)
    )

    # Ensure cache doesn't exceed available chunks
    if num_chunks > 0:
        optimal_cache = min(optimal_cache, num_chunks)

    # Calculate total memory needed
    total_memory_mb = (
        base_memory_mb
        + optimal_batch * memory_per_batch_mb
        + optimal_cache * memory_per_cache_mb
    )

    # Device-specific overhead
    if device_type == "mps":
        overhead_factor = 1.5
    elif device_type == "cuda":
        overhead_factor = 1.3
    else:
        overhead_factor = 1.2

    total_memory_gb = (total_memory_mb / 1024) * overhead_factor
    total_memory_gb = round(total_memory_gb * 2) / 2  # Round to nearest 0.5

    return {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_info": dataset_info,
        "optimal_batch_size": optimal_batch,
        "optimal_cache_size": optimal_cache,
        "recommended_memory_limit_gb": max(0.5, total_memory_gb),
        "total_memory_mb": int(total_memory_mb),
        "base_memory_mb": base_memory_mb,
        "batch_memory_mb": batch_memory_mb,
        "cache_memory_mb": optimal_cache * memory_per_cache_mb,
        "can_fit": total_memory_gb < available_memory_gb * 0.9,
        "available_memory_gb": available_memory_gb,
    }


def generate_smart_variants(
    model_name: str,
    dataset_name: str,
    available_memory_gb: float,
    device_type: str,
) -> list[dict]:
    """
    Generate smart config variants based on actual data and system resources.
    Starts with optimal calculated values, then provides fallbacks.
    """

    # Calculate optimal config
    optimal = calculate_smart_config(
        model_name, dataset_name, available_memory_gb, device_type
    )

    if not optimal:
        return []

    variants = []

    # Determine class weights
    if optimal.get("params", 0) > 1000000:
        class_weights = "0.1,0.1,0.8"
    else:
        class_weights = "0.2,0.2,0.6"

    # Level 1: Optimal calculated config
    variants.append(
        {
            "batch_size": optimal["optimal_batch_size"],
            "model": model_name,
            "cache_size": optimal["optimal_cache_size"],
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": optimal["recommended_memory_limit_gb"],
        }
    )

    # Level 2: Reduce batch by 25%
    variants.append(
        {
            "batch_size": max(1, int(optimal["optimal_batch_size"] * 0.75)),
            "model": model_name,
            "cache_size": optimal["optimal_cache_size"],
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": max(0.5, optimal["recommended_memory_limit_gb"] * 0.85),
        }
    )

    # Level 3: Reduce cache by 25%
    variants.append(
        {
            "batch_size": optimal["optimal_batch_size"],
            "model": model_name,
            "cache_size": max(1, int(optimal["optimal_cache_size"] * 0.75)),
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": max(0.5, optimal["recommended_memory_limit_gb"] * 0.85),
        }
    )

    # Level 4: Reduce both by 50%
    variants.append(
        {
            "batch_size": max(1, int(optimal["optimal_batch_size"] * 0.5)),
            "model": model_name,
            "cache_size": max(1, int(optimal["optimal_cache_size"] * 0.5)),
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": max(0.5, optimal["recommended_memory_limit_gb"] * 0.7),
        }
    )

    # Level 5: Conservative (batch=1, cache=1)
    variants.append(
        {
            "batch_size": 1,
            "model": model_name,
            "cache_size": 1,
            "class_weights": class_weights,
            "strip_width": 8,
            "memory_limit_gb": max(1.0, optimal["recommended_memory_limit_gb"] * 0.5),
        }
    )

    # Remove duplicates
    seen = set()
    unique_variants = []
    for v in variants:
        key = (v["batch_size"], v["cache_size"], v["memory_limit_gb"])
        if key not in seen:
            seen.add(key)
            unique_variants.append(v)

    return unique_variants


def generate_smart_configurations(
    dataset_name: str,
    available_memory_gb: float,
    device_type: str,
) -> dict[str, list[dict]]:
    """
    Generate smart configurations for all models for a specific dataset.
    """

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
    all_variants = {}

    for model_name in model_order:
        variants = generate_smart_variants(
            model_name=model_name,
            dataset_name=dataset_name,
            available_memory_gb=available_memory_gb,
            device_type=device_type,
        )
        if variants:
            all_variants[model_name] = variants

    return all_variants


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

    # Generate variants
    all_variants = generate_all_variants(available_gb, device_type)

    # Save to file
    flat_variants = []
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
    for model_name in model_order:
        if model_name in all_variants:
            flat_variants.extend(all_variants[model_name])

    with open("auto_config.json", "w") as f:
        json.dump(
            {
                "device_type": device_type,
                "available_gb": available_gb,
                "variants": flat_variants,
            },
            f,
            indent=2,
        )

    print("\n✅ Auto-config saved to: auto_config.json")
    print(f"   Generated {len(flat_variants)} config variants")

    return {
        "device_type": device_type,
        "available_gb": available_gb,
        "all_variants": all_variants,
        "flat_variants": flat_variants,
    }


if __name__ == "__main__":
    result = main()
