# file location: src/batch/smart_config.py

"""
Smart auto-configuration logic for batch training.
"""

from typing import Any

from src.utils.model_registry import get_model_registry

from .config import get_actual_data_shape, get_dataset_info


def _resolve_loss_config(
    dataset_name: str,
    global_config: dict[str, Any] | None,
    auto_config: dict[str, Any] | None,
    model_params: int,
) -> dict[str, Any]:
    """
    Resolve loss configuration with priority:
        1. auto_config.loss_overrides[dataset]
        2. global_config
        3. Sensible defaults

    Args:
        dataset_name: Name of the dataset
        global_config: Global config dict (may be None)
        auto_config: Auto config dict (may be None)
        model_params: Number of parameters in the model (for default class weights)

    Returns:
        Dict with keys: class_weights, loss_function, dice_weight, focal_gamma
    """

    def _format_weights(weights: Any) -> str:
        """Convert class weights to comma-separated string."""
        if isinstance(weights, str):
            return weights
        return ",".join(str(w) for w in weights)

    def _lookup(key: str, default: Any) -> Any:
        """Look up a loss config key with the priority order."""
        # Priority 1: auto_config.loss_overrides[dataset]
        if auto_config:
            value = auto_config.get("loss_overrides", {}).get(dataset_name, {}).get(key)
            if value is not None:
                return value

        # Priority 2: global_config
        if global_config and key in global_config:
            value = global_config[key]
            if value is not None:
                return value

        # Priority 3: default
        return default

    # Model-size-based default class weights
    default_weights = [0.1, 0.1, 0.8] if model_params > 1000000 else [0.2, 0.2, 0.6]

    return {
        "class_weights": _format_weights(_lookup("class_weights", default_weights)),
        "loss_function": _lookup("loss_function", "combo"),
        "dice_weight": _lookup("dice_weight", 0.5),
        "focal_gamma": _lookup("focal_gamma", 2.0),
    }


def calculate_optimal_config(
    model_name: str,
    dataset_name: str,
    available_memory_gb: float,
    device_type: str,
    global_config: dict[str, Any] | None = None,
    auto_config: dict[str, Any] | None = None,
    logger=None,
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

    # 5. CALCULATE OPTIMAL CACHE SIZE  ← MODIFIED (D.1, Option A)
    #
    # Two policies depending on the sampling strategy. Both respect the
    # memory budget as a hard ceiling — the sampler mode only changes the
    # *upper cap* on cache size, not whether memory is considered.
    #
    #   - Sampler ON (chunk_aware_sampling=True):
    #       ChunkAwareSampler yields samples chunk-by-chunk. The cache only
    #       needs to hold the currently-active chunk plus a small margin.
    #       Cap at 3 — but never exceed what memory allows.
    #
    #   - Sampler OFF (chunk_aware_sampling=False):
    #       Full shuffle. Each batch pulls random samples across chunks, so
    #       cache sizing is memory-bound with a size-based factor. Unchanged
    #       from the original heuristic.
    #
    remaining_after_batch_mb = safe_remaining_mb - batch_memory_mb

    memory_per_cache_mb = profile.memory_per_cache_mb

    num_chunks = dataset_info.get("num_chunks", 4)

    # Read sampler mode from global_config (default True, matches SeismicConfig)
    chunk_aware = True
    if global_config is not None:
        chunk_aware = bool(global_config.get("chunk_aware_sampling", True))

    if memory_per_cache_mb > 0:
        max_cache_by_memory = int(remaining_after_batch_mb / memory_per_cache_mb)
    else:
        max_cache_by_memory = 3

    recommended_cache = profile.recommended_cache_size

    if chunk_aware:
        # Sequential chunk access — cap at 3, but memory is still a hard ceiling.
        optimal_cache: int = min(
            3,
            num_chunks if num_chunks > 0 else 3,
            max(1, max_cache_by_memory),
        )
        cache_policy = "sampler-aware (sequential, cap 3)"
        cache_factor = None
    else:
        # Random access — original memory-bound heuristic (unchanged).
        cache_factor = min(1.0, max(0.3, num_chunks / 10))

        optimal_cache = min(
            max(1, max_cache_by_memory),
            int(recommended_cache * cache_factor),
            num_chunks if num_chunks > 0 else 4,
        )
        cache_policy = "random-access (memory-bound heuristic)"

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
            "optimal_cache": {
                "value": optimal_cache,
                "policy": cache_policy,
                "max_cache_by_memory": max_cache_by_memory,   # ← NEW structured field
                "description": (
                    f"chunk_aware={chunk_aware}, "
                    f"recommended={recommended_cache}, "
                    f"num_chunks={num_chunks}, "
                    f"max_by_memory={max_cache_by_memory}, "
                    f"cache_factor={cache_factor if cache_factor is not None else 'N/A'}"
                ),
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
            **_resolve_loss_config(
                dataset_name=dataset_name,
                global_config=global_config,
                auto_config=auto_config,
                model_params=profile.params,
            ),
        },
    }

    return explanation