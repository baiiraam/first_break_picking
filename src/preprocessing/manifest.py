"""
Manifest generation for chunked datasets.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from loguru import logger


def generate_manifest(
    dataset_name: str,
    chunks: Dict[str, List[Dict]],
    config: Dict[str, Any],
    chunk_dir: Path,
    total_shots: int,
    total_traces: int
) -> Dict[str, Any]:
    """
    Generate a manifest JSON file for the chunked dataset.
    """
    manifest = {
        "dataset": dataset_name,
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "config": config,
        "total_shots": total_shots,
        "total_traces": total_traces,
        "chunks": []
    }
    
    for split_name, chunk_list in chunks.items():
        for chunk in chunk_list:
            manifest["chunks"].append({
                "id": chunk['id'],
                "filename": f"chunk_{chunk['id']:03d}_{split_name}.pt",
                "split": split_name,
                "shot_ids": chunk['shot_ids'],
                "n_shots": chunk['n_shots'],
                "start_idx": chunk.get('start_idx', 0),
                "end_idx": chunk.get('end_idx', 0),
            })
    
    return manifest


def save_manifest(manifest: Dict[str, Any], path: Path):
    """Save manifest to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {path}")


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load manifest from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def validate_manifest(manifest: Dict[str, Any]) -> bool:
    """Validate manifest structure."""
    required_keys = ['dataset', 'version', 'created', 'config', 'chunks']
    for key in required_keys:
        if key not in manifest:
            logger.error(f"Missing required key in manifest: {key}")
            return False
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