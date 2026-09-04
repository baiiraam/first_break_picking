import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# ✅ Larger buffer size for faster checksums
CHECKSUM_BUFFER_SIZE = 65536  # 64KB (up from 4KB)


def compute_checksum(filepath: Path, buffer_size: int = CHECKSUM_BUFFER_SIZE) -> str:
    """
    Compute SHA-256 checksum of a file with optimized buffer size.

    Args:
        filepath: Path to the file
        buffer_size: Read buffer size in bytes (default: 64KB)
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(buffer_size), b""):
            sha256.update(block)
    return sha256.hexdigest()[:16]


def compute_checksum_from_bytes(data: bytes) -> str:
    """Compute checksum from bytes data (no file I/O)."""
    return hashlib.sha256(data).hexdigest()[:16]


def compute_checksum_from_string(json_str: str) -> str:
    """Compute checksum from JSON string (no file I/O)."""
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:16]


def get_next_version(manifest_path: Path) -> str:
    """Get next version number from existing manifest."""
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                existing = json.load(f)
            version_parts = existing.get("version", "1.0.0").split(".")
            patch = int(version_parts[2]) + 1
            return f"{version_parts[0]}.{version_parts[1]}.{patch}"
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning(
                f"Could not parse version from {manifest_path}, starting at 1.0.0"
            )
            return "1.0.0"
    return "1.0.0"


def generate_manifest(
    dataset_name: str,
    chunks: dict[str, list[dict]],
    config: dict[str, Any],
    chunk_dir: Path,
    total_shots: int,
    total_traces: int,
    increment_version: bool = True,
) -> dict[str, Any]:
    """
    Generate a manifest JSON file for the chunked dataset with checksums.
    """
    manifest_path = chunk_dir / "manifest.json"

    manifest = {
        "dataset": dataset_name,
        "version": get_next_version(manifest_path) if increment_version else "1.0.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "total_shots": total_shots,
        "total_traces": total_traces,
        "chunks": [],
    }

    for split_name, chunk_list in chunks.items():
        for chunk in chunk_list:
            chunk_filename = f"chunk_{chunk['id']:03d}_{split_name}.pt"
            chunk_path = chunk_dir / chunk_filename
            file_size_mb = (
                chunk_path.stat().st_size / (1024 * 1024) if chunk_path.exists() else 0
            )

            manifest["chunks"].append(
                {
                    "id": chunk["id"],
                    "filename": chunk_filename,
                    "split": split_name,
                    "shot_ids": chunk["shot_ids"],
                    "n_shots": chunk["n_shots"],
                    "start_idx": chunk.get("start_idx", 0),
                    "end_idx": chunk.get("end_idx", 0),
                    "file_size_mb": round(file_size_mb, 2),
                    "checksum": compute_checksum(chunk_path)
                    if chunk_path.exists()
                    else None,
                }
            )

    # Compute manifest checksum (after all chunks are added)
    manifest["manifest_checksum"] = None  # Will be computed after saving

    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    """Load manifest from JSON file and verify checksum in-memory."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r") as f:
        manifest = json.load(f)

    # ✅ Verify checksum in-memory (no temporary file)
    if manifest.get("manifest_checksum"):
        stored_checksum = manifest["manifest_checksum"]

        # Remove checksum from dict for computation
        manifest_copy = {k: v for k, v in manifest.items() if k != "manifest_checksum"}

        # ✅ Compute checksum from JSON string (no disk I/O)
        json_str = json.dumps(manifest_copy, sort_keys=True, indent=2)
        computed_checksum = compute_checksum_from_string(json_str)

        if stored_checksum != computed_checksum:
            logger.warning(
                f"Manifest checksum mismatch: stored={stored_checksum}, computed={computed_checksum}"
            )

    logger.info(
        f"Manifest loaded from {path} (version: {manifest.get('version', 'unknown')})"
    )
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> bool:
    """Validate manifest structure and consistency."""
    # Check required keys
    required_keys = ["dataset", "version", "created", "config", "chunks"]
    for key in required_keys:
        if key not in manifest:
            logger.error(f"Missing required key in manifest: {key}")
            return False

    # Check version format
    version = manifest.get("version", "")
    if not version:
        logger.error("Missing version")
        return False

    # Check chunks
    if not manifest["chunks"]:
        logger.error("No chunks found in manifest")
        return False

    # Check chunk consistency
    total_shots = 0
    seen_splits = set()
    for chunk in manifest["chunks"]:
        total_shots += chunk["n_shots"]

        # Check required keys per chunk
        chunk_keys = ["id", "filename", "split", "shot_ids", "n_shots"]
        for key in chunk_keys:
            if key not in chunk:
                logger.error(
                    f"Missing key in chunk {chunk.get('id', 'unknown')}: {key}"
                )
                return False

        # Check split
        if chunk["split"] not in ["train", "val", "test"]:
            logger.error(f"Invalid split in chunk {chunk['id']}: {chunk['split']}")
            return False
        seen_splits.add(chunk["split"])

    # Check total shots matches
    if total_shots != manifest["total_shots"]:
        logger.error(
            f"Total shots mismatch: {total_shots} vs {manifest['total_shots']}"
        )
        return False

    # Check all splits exist
    expected_splits = {"train", "val", "test"}
    missing_splits = expected_splits - seen_splits
    if missing_splits:
        logger.warning(f"Missing splits: {missing_splits}")
        # Not a fatal error, just a warning

    return True


def get_chunk_paths(manifest: dict[str, Any], chunk_dir: Path) -> dict[str, Path]:
    """Get all chunk file paths from manifest."""
    paths = {}
    for chunk in manifest["chunks"]:
        split = chunk["split"]
        filename = chunk["filename"]
        if split not in paths:
            paths[split] = []
        paths[split].append(chunk_dir / filename)
    return paths


def get_manifest_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    """Get statistics about the manifest."""
    total_chunks = len(manifest["chunks"])
    total_shots = manifest["total_shots"]
    total_size_mb = sum(c.get("file_size_mb", 0) for c in manifest["chunks"])

    split_stats = {}
    for chunk in manifest["chunks"]:
        split = chunk["split"]
        if split not in split_stats:
            split_stats[split] = {"chunks": 0, "shots": 0, "size_mb": 0}
        split_stats[split]["chunks"] += 1
        split_stats[split]["shots"] += chunk["n_shots"]
        split_stats[split]["size_mb"] += chunk.get("file_size_mb", 0)

    return {
        "total_chunks": total_chunks,
        "total_shots": total_shots,
        "total_size_mb": round(total_size_mb, 2),
        "total_size_gb": round(total_size_mb / 1024, 2),
        "split_stats": split_stats,
    }


def validate_manifest_files(manifest: dict[str, Any], chunk_dir: Path) -> bool:
    """
    Validate that all chunk files in the manifest exist and are readable.
    Returns True if all files exist, False otherwise.
    """
    all_exist = True
    for chunk in manifest["chunks"]:
        chunk_path = chunk_dir / chunk["filename"]
        if not chunk_path.exists():
            logger.error(f"Missing chunk file: {chunk_path}")
            all_exist = False
        elif chunk.get("file_size_mb", 0) == 0:
            # File exists but size is 0 - recompute
            logger.warning(f"File size is 0 for {chunk_path}, recomputing...")
            chunk["file_size_mb"] = chunk_path.stat().st_size / (1024 * 1024)
    return all_exist


def save_manifest(
    manifest: dict[str, Any],
    path: Path,
    validate_files: bool = False,  # ✅ Default to False
    force_validate: bool = False,  # ✅ Optional explicit validation
):
    """
    Save manifest to JSON file with checksum and optional file validation.

    Args:
        manifest: Manifest dictionary
        path: Path to save manifest
        validate_files: If True, validate chunk files exist before saving
        force_validate: If True, raise error on missing files instead of warning
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate files exist if requested
    if validate_files:
        chunk_dir = path.parent
        if not validate_manifest_files(manifest, chunk_dir):
            if force_validate:
                raise RuntimeError(f"Missing chunk files for manifest at {path}")
            else:
                logger.warning("Some chunk files are missing or have 0 size")

    # Save manifest without checksum first
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Compute checksum in-memory
    manifest_copy = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
    json_str = json.dumps(manifest_copy, sort_keys=True, indent=2)
    checksum = compute_checksum_from_string(json_str)
    manifest["manifest_checksum"] = checksum

    # Re-save with checksum
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved to {path} (checksum: {checksum})")
