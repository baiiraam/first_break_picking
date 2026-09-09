"""
Smart auto-configuration logic for batch training.
"""

from typing import Any

from src.utils.model_registry import get_model_registry

from .config import get_actual_data_shape, get_dataset_info


def calculate_optimal_config(
    model_name: str,
    dataset_name: str,
    available_memory_gb: float,
    device_type: str,
    logger=None,  # ← Added logger parameter
) -> dict[str, Any]:
    """
    Calculate optimal config with detailed reasoning.
    Uses model registry for profiles.
    """

    # Get profile from registry
    registry = get_model_registry()
    profile = registry.get(model_name)

    if not profile:
        if logger:
            logger.warning(f"Unknown model: {model_name}, skipping")
        return {}

    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    _actual_traces, _actual_samples = get_actual_data_shape(dataset_name)

    # Convert available memory to MB
    available_mb = available_memory_gb * 1024

    # 1. CALCULATE BASE MEMORY
    base_memory_mb = profile.base_memory_mb
    base_memory_gb = base_memory_mb / 1024

    # 2. CALCULATE AVAILABLE MEMORY FOR BATCH AND CACHE
    remaining_mb = available_mb - base_memory_mb

    # 3. APPLY SAFETY MARGIN (20% for PyTorch overhead)
    safe_remaining_mb = remaining_mb * 0.8
    safe_remaining_gb = safe_remaining_mb / 1024

    # 4. CALCULATE OPTIMAL BATCH SIZE
    memory_per_batch_mb = profile.memory_per_batch_mb

    total_shots = dataset_info.get("total_shots", 0)
    if total_shots > 200:
        dataset_factor = 1.2
        dataset_size_label = "large"
    elif total_shots > 50:
        dataset_factor = 1.0
        dataset_size_label = "medium"
    else:
        dataset_factor = 0.8
        dataset_size_label = "small"

    if memory_per_batch_mb > 0:
        max_batch_by_memory = int(safe_remaining_mb / memory_per_batch_mb)
    else:
        max_batch_by_memory = 8

    recommended_batch = profile.recommended_batch_size

    optimal_batch: int = min(
        max(1, max_batch_by_memory),
        int(recommended_batch * dataset_factor),
        total_shots if total_shots > 0 else 64,
    )

    batch_memory_mb = optimal_batch * memory_per_batch_mb
    batch_memory_gb = batch_memory_mb / 1024

    # 5. CALCULATE OPTIMAL CACHE SIZE
    remaining_after_batch_mb = safe_remaining_mb - batch_memory_mb

    memory_per_cache_mb = profile.memory_per_cache_mb

    num_chunks = dataset_info.get("num_chunks", 4)
    cache_factor = min(1.0, max(0.3, num_chunks / 10))

    if memory_per_cache_mb > 0:
        max_cache_by_memory = int(remaining_after_batch_mb / memory_per_cache_mb)
    else:
        max_cache_by_memory = 3

    recommended_cache = profile.recommended_cache_size

    optimal_cache: int = min(
        max(1, max_cache_by_memory),
        int(recommended_cache * cache_factor),
        num_chunks if num_chunks > 0 else 4,
    )

    cache_memory_mb = optimal_cache * memory_per_cache_mb
    cache_memory_gb = cache_memory_mb / 1024

    # 6. CALCULATE TOTAL MEMORY
    total_memory_mb = base_memory_mb + batch_memory_mb + cache_memory_mb
    total_memory_gb = total_memory_mb / 1024

    if device_type == "mps":
        overhead_factor = 1.5
    elif device_type == "cuda":
        overhead_factor = 1.3
    else:
        overhead_factor = 1.2

    memory_limit_gb: float = (total_memory_mb / 1024) * overhead_factor
    memory_limit_gb = round(memory_limit_gb * 2) / 2
    memory_limit_gb = max(0.5, memory_limit_gb)

    can_fit = memory_limit_gb < available_memory_gb * 0.9

    # 8. BUILD EXPLANATION
    explanation = {
        "model": model_name,
        "dataset": dataset_name,
        "device": device_type,
        "available_memory_gb": available_memory_gb,
        "calculations": {
            "base_memory": {
                "value_mb": base_memory_mb,
                "value_gb": base_memory_gb,
                "description": f"Base memory for {model_name} model ({profile.params:,} params)",
            },
            "safe_remaining": {
                "value_mb": safe_remaining_mb,
                "value_gb": safe_remaining_gb,
                "description": "Memory available after base and 20% overhead",
            },
            "dataset_factor": {
                "value": dataset_factor,
                "description": f"Dataset size: {dataset_size_label} ({total_shots} shots)",
            },
            "optimal_batch": {
                "value": optimal_batch,
                "description": f"max_by_memory={max_batch_by_memory}, recommended={recommended_batch}, factor={dataset_factor:.1f}",
            },
            "batch_memory": {
                "value_mb": batch_memory_mb,
                "value_gb": batch_memory_gb,
                "description": f"{optimal_batch} batches × {memory_per_batch_mb}MB/batch",
            },
            "cache_factor": {
                "value": cache_factor,
                "description": f"{num_chunks} chunks available",
            },
            "optimal_cache": {
                "value": optimal_cache,
                "description": f"max_by_memory={max_cache_by_memory}, recommended={recommended_cache}, factor={cache_factor:.1f}",
            },
            "cache_memory": {
                "value_mb": cache_memory_mb,
                "value_gb": cache_memory_gb,
                "description": f"{optimal_cache} caches × {memory_per_cache_mb}MB/cache",
            },
            "total_memory": {
                "value_mb": total_memory_mb,
                "value_gb": total_memory_gb,
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
        "can_fit": can_fit,
        "final_config": {
            "batch_size": optimal_batch,
            "cache_size": optimal_cache,
            "memory_limit_gb": memory_limit_gb,
            "class_weights": "0.05,0.05,0.9"
            if profile.params > 1000000
            else "0.2,0.2,0.6",
        },
    }

    return explanation
