#!/usr/bin/env python3
"""
Inspect the created chunks to verify the ms-to-samples conversion.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")


def inspect_chunk(chunk_path: Path, num_samples: int = 5):
    """Inspect a single chunk file."""
    
    print(f"\n📁 Inspecting: {chunk_path.name}")
    print("-" * 60)
    
    # Load chunk
    chunk = torch.load(chunk_path, map_location='cpu', weights_only=True)
    
    data = chunk['data']  # (n_shots, traces, samples)
    mask = chunk['mask']  # (n_shots, traces, samples)
    shot_ids = chunk['shot_ids']
    
    n_shots = data.shape[0]
    n_traces = data.shape[1]
    n_samples = data.shape[2]
    
    print(f"  Shots: {n_shots}")
    print(f"  Traces: {n_traces}")
    print(f"  Samples: {n_samples}")
    
    # FIX: shot_ids is a list, not a tensor
    if hasattr(shot_ids, 'tolist'):
        shot_ids_list = shot_ids.tolist()
    else:
        shot_ids_list = shot_ids
    
    print(f"  Shot IDs (first 10): {shot_ids_list[:10]}{'...' if n_shots > 10 else ''}")
    print(f"  Data range: {data.min():.2f} to {data.max():.2f}")
    print(f"  Mask classes: {np.unique(mask.numpy())}")
    
    # Show first few shots
    print(f"\n  📊 First {min(num_samples, n_shots)} shots:")
    
    for shot_idx in range(min(num_samples, n_shots)):
        shot_id = shot_ids_list[shot_idx]
        shot_mask = mask[shot_idx]  # (traces, samples)
        
        # Find where class 2 (strip) exists for each trace
        strip_present = (shot_mask == 2).any(dim=1)
        num_with_strip = strip_present.sum().item()
        
        # Find the center of the strip for the first few traces
        picks = []
        for trace_idx in range(min(5, n_traces)):
            strip_indices = torch.where(shot_mask[trace_idx] == 2)[0]
            if len(strip_indices) > 0:
                center = int(strip_indices[len(strip_indices)//2].item())
                picks.append(center)
            else:
                picks.append(None)
        
        print(f"\n    Shot {shot_id}:")
        print(f"      Traces with strip: {num_with_strip}/{n_traces}")
        print(f"      Strip centers (first 5 traces): {picks}")
        
        # Check if picks are in valid range (0-750 for Halfmile)
        valid_picks = [p for p in picks if p is not None]
        if valid_picks:
            avg_pick = np.mean(valid_picks)
            print(f"      Avg pick position: {avg_pick:.1f} samples")
            if avg_pick < n_samples:
                print(f"      ✅ Picks are in sample range (0-{n_samples-1})")
            else:
                print(f"      ❌ Picks are in milliseconds (>{n_samples})")


def inspect_manifest(chunk_dir: Path):
    """Inspect the manifest file."""
    
    manifest_path = chunk_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"\n📋 Manifest: {manifest_path}")
    print("-" * 60)
    print(f"  Dataset: {manifest.get('dataset')}")
    print(f"  Version: {manifest.get('version')}")
    print(f"  Total shots: {manifest.get('total_shots')}")
    print(f"  Total chunks: {len(manifest.get('chunks', []))}")
    
    config = manifest.get('config', {})
    print("\n  Config:")
    print(f"    target_traces: {config.get('target_traces')}")
    print(f"    n_samples: {config.get('n_samples')}")
    print(f"    strip_width: {config.get('strip_width')}")
    print(f"    sample_rate_ms: {config.get('sample_rate_ms', 'NOT SET!')}")
    
    if config.get('sample_rate_ms') is None:
        print("    ⚠️  sample_rate_ms is NOT set in config!")


def main():
    """Main function."""
    
    # Check which dataset to inspect
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "Halfmile"
    
    chunk_dir = Path(f"data/chunks/{dataset}")
    
    if not chunk_dir.exists():
        print(f"❌ Chunk directory not found: {chunk_dir}")
        return
    
    print("=" * 60)
    print(f"🔍 INSPECTING CHUNKS: {dataset}")
    print("=" * 60)
    
    # Inspect manifest
    inspect_manifest(chunk_dir)
    
    # Find chunk files
    chunk_files = sorted(chunk_dir.glob("chunk_*.pt"))
    
    if not chunk_files:
        print(f"\n❌ No chunk files found in {chunk_dir}")
        return
    
    print(f"\n📁 Found {len(chunk_files)} chunk files")
    
    # Inspect first few chunks
    for chunk_file in chunk_files[:3]:
        inspect_chunk(chunk_file)
    
    if len(chunk_files) > 3:
        print(f"\n... and {len(chunk_files) - 3} more chunks")


if __name__ == "__main__":
    main()
    