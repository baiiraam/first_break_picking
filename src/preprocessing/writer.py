"""
Chunk writing logic for preprocessing pipeline.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from loguru import logger


class ChunkWriter:
    """
    Writes processed chunks to disk.
    """
    
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = Path(chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
    
    def write_chunk(
        self,
        data_batch: np.ndarray,
        mask_batch: np.ndarray,
        shot_ids: List[int],
        chunk_id: int,
        split: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Write a chunk to disk.
        
        Args:
            data_batch: (n_shots, target_traces, n_samples) float32
            mask_batch: (n_shots, target_traces, n_samples) int64
            shot_ids: List of shot IDs
            chunk_id: Chunk identifier
            split: 'train', 'val', or 'test'
            metadata: Additional metadata to save
        
        Returns:
            Path to saved chunk file
        """
        filename = f"chunk_{chunk_id:03d}_{split}.pt"
        filepath = self.chunk_dir / filename
        
        chunk_data = {
            'data': torch.tensor(data_batch, dtype=torch.float32),
            'mask': torch.tensor(mask_batch, dtype=torch.long),
            'shot_ids': shot_ids,
            'split': split,
            'chunk_id': chunk_id,
            'n_shots': len(shot_ids),
            'data_shape': list(data_batch.shape),
            'mask_shape': list(mask_batch.shape)
        }
        
        if metadata:
            chunk_data.update(metadata)
        
        torch.save(chunk_data, filepath)
        logger.debug(f"Written chunk: {filename} ({filepath.stat().st_size / (1024*1024):.1f} MB)")
        
        return filepath
    
    def write_all_chunks(
        self,
        chunks: Dict[str, List[Dict[str, Any]]],
        data_batches: Dict[int, np.ndarray],
        mask_batches: Dict[int, np.ndarray],
        shot_ids: Dict[int, List[int]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """
        Write all chunks from a preprocessing run.
        
        Returns:
            List of paths to saved chunk files
        """
        saved_paths = []
        
        total_chunks = sum(len(c) for c in chunks.values())
        pbar = tqdm(total=total_chunks, desc="Writing chunks")
        
        for split_name, chunk_list in chunks.items():
            for chunk in chunk_list:
                chunk_id = chunk['id']
                
                filepath = self.write_chunk(
                    data_batch=data_batches[chunk_id],
                    mask_batch=mask_batches[chunk_id],
                    shot_ids=shot_ids[chunk_id],
                    chunk_id=chunk_id,
                    split=split_name,
                    metadata=metadata
                )
                
                saved_paths.append(filepath)
                pbar.update(1)
                pbar.set_postfix({'current': filepath.name})
        
        pbar.close()
        logger.info(f"Written {len(saved_paths)} chunks to {self.chunk_dir}")
        
        return saved_paths
    
    def verify_chunk(self, filepath: Path) -> bool:
        """
        Verify a chunk file is valid and loadable.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            chunk = torch.load(filepath, map_location='cpu', weights_only=False)
            
            required_keys = ['data', 'mask', 'shot_ids', 'split', 'chunk_id', 'n_shots']
            for key in required_keys:
                if key not in chunk:
                    logger.error(f"Missing key in {filepath}: {key}")
                    return False
            
            if chunk['data'].shape[0] != chunk['n_shots']:
                logger.error(f"Data shape mismatch in {filepath}: data has {chunk['data'].shape[0]} shots, expected {chunk['n_shots']}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error verifying {filepath}: {e}")
            return False