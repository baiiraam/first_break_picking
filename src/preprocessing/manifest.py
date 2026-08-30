"""
Manifest generation for chunked datasets with checksums and versioning.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger


def compute_checksum(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            sha256.update(block)
    return sha256.hexdigest()[:16]


def get_next_version(manifest_path: Path) -> str:
    """Get next version number from existing manifest."""
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                existing = json.load(f)
            version_parts = existing.get('version', '1.0.0').split('.')
            patch = int(version_parts[2]) + 1
            return f"{version_parts[0]}.{version_parts[1]}.{patch}"
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning(f"Could not parse version from {manifest_path}, starting at 1.0.0")
            return "1.0.0"
    return "1.0.0"


def generate_manifest(
    dataset_name: str,
    chunks: Dict[str, List[Dict]],
    config: Dict[str, Any],
    chunk_dir: Path,
    total_shots: int,
    total_traces: int,
    increment_version: bool = True
) -> Dict[str, Any]:
    """
    Generate a manifest JSON file for the chunked dataset with checksums.
    """
    manifest_path = chunk_dir / "manifest.json"
    
    manifest = {
        "dataset": dataset_name,
        "version": get_next_version(manifest_path) if increment_version else "1.0.0",
        "created": datetime.now().isoformat(),
        "config": config,
        "total_shots": total_shots,
        "total_traces": total_traces,
        "chunks": []
    }
    
    for split_name, chunk_list in chunks.items():
        for chunk in chunk_list:
            chunk_filename = f"chunk_{chunk['id']:03d}_{split_name}.pt"
            chunk_path = chunk_dir / chunk_filename
            file_size_mb = chunk_path.stat().st_size / (1024 * 1024) if chunk_path.exists() else 0
            
            manifest["chunks"].append({
                "id": chunk['id'],
                "filename": chunk_filename,
                "split": split_name,
                "shot_ids": chunk['shot_ids'],
                "n_shots": chunk['n_shots'],
                "start_idx": chunk.get('start_idx', 0),
                "end_idx": chunk.get('end_idx', 0),
                "file_size_mb": round(file_size_mb, 2),
                "checksum": compute_checksum(chunk_path) if chunk_path.exists() else None,
            })
    
    # Compute manifest checksum (after all chunks are added)
    manifest["manifest_checksum"] = None  # Will be computed after saving
    
    return manifest


def save_manifest(manifest: Dict[str, Any], path: Path):
    """Save manifest to JSON file with checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save manifest without checksum first
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Compute and update checksum
    checksum = compute_checksum(path)
    manifest["manifest_checksum"] = checksum
    
    # Re-save with checksum
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest saved to {path} (checksum: {checksum})")


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load manifest from JSON file and verify checksum."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    
    with open(path, 'r') as f:
        manifest = json.load(f)
    
    # Verify manifest checksum if present
    if "manifest_checksum" in manifest and manifest["manifest_checksum"]:
        stored_checksum = manifest["manifest_checksum"]
        # Remove checksum before computing
        manifest_copy = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(manifest_copy, f, indent=2)
        computed_checksum = compute_checksum(temp_path)
        temp_path.unlink()
        
        if stored_checksum != computed_checksum:
            logger.warning(f"Manifest checksum mismatch: stored={stored_checksum}, computed={computed_checksum}")
    
    logger.info(f"Manifest loaded from {path} (version: {manifest.get('version', 'unknown')})")
    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> bool:
    """Validate manifest structure and consistency."""
    # Check required keys
    required_keys = ['dataset', 'version', 'created', 'config', 'chunks']
    for key in required_keys:
        if key not in manifest:
            logger.error(f"Missing required key in manifest: {key}")
            return False
    
    # Check version format
    version = manifest.get('version', '')
    if not version:
        logger.error("Missing version")
        return False
    
    # Check chunks
    if not manifest['chunks']:
        logger.error("No chunks found in manifest")
        return False
    
    # Check chunk consistency
    total_shots = 0
    seen_splits = set()
    for chunk in manifest['chunks']:
        total_shots += chunk['n_shots']
        
        # Check required keys per chunk
        chunk_keys = ['id', 'filename', 'split', 'shot_ids', 'n_shots']
        for key in chunk_keys:
            if key not in chunk:
                logger.error(f"Missing key in chunk {chunk.get('id', 'unknown')}: {key}")
                return False
        
        # Check split
        if chunk['split'] not in ['train', 'val', 'test']:
            logger.error(f"Invalid split in chunk {chunk['id']}: {chunk['split']}")
            return False
        seen_splits.add(chunk['split'])
    
    # Check total shots matches
    if total_shots != manifest['total_shots']:
        logger.error(f"Total shots mismatch: {total_shots} vs {manifest['total_shots']}")
        return False
    
    # Check all splits exist
    expected_splits = {'train', 'val', 'test'}
    missing_splits = expected_splits - seen_splits
    if missing_splits:
        logger.warning(f"Missing splits: {missing_splits}")
        # Not a fatal error, just a warning
    
    return True


def get_chunk_paths(manifest: Dict[str, Any], chunk_dir: Path) -> Dict[str, Path]:
    """Get all chunk file paths from manifest."""
    paths = {}
    for chunk in manifest['chunks']:
        split = chunk['split']
        filename = chunk['filename']
        if split not in paths:
            paths[split] = []
        paths[split].append(chunk_dir / filename)
    return paths


def get_manifest_stats(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Get statistics about the manifest."""
    total_chunks = len(manifest['chunks'])
    total_shots = manifest['total_shots']
    total_size_mb = sum(c.get('file_size_mb', 0) for c in manifest['chunks'])
    
    split_stats = {}
    for chunk in manifest['chunks']:
        split = chunk['split']
        if split not in split_stats:
            split_stats[split] = {'chunks': 0, 'shots': 0, 'size_mb': 0}
        split_stats[split]['chunks'] += 1
        split_stats[split]['shots'] += chunk['n_shots']
        split_stats[split]['size_mb'] += chunk.get('file_size_mb', 0)
    
    return {
        'total_chunks': total_chunks,
        'total_shots': total_shots,
        'total_size_mb': round(total_size_mb, 2),
        'total_size_gb': round(total_size_mb / 1024, 2),
        'split_stats': split_stats,
    }