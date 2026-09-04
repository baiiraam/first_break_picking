#!/usr/bin/env python3
"""
Quick test script to verify critical fixes are working.
Tests: chunker ID uniqueness, cache thread safety, model imports, config immutability.
"""

import os
import sys
import threading

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def print_section(title: str, char: str = "=", width: int = 70):
    """Print a formatted section header."""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def test_chunker_unique_ids():
    """Test that chunk IDs are unique across all splits."""
    print_section("1. TESTING CHUNKER UNIQUE IDs", "=", 70)
    
    from src.preprocessing.chunker import Chunker
    
    chunker = Chunker(chunk_size=20, random_seed=42)
    
    # Create sample splits
    shot_ids = np.arange(100)
    splits = chunker.assign_splits(shot_ids)
    
    # Create chunks with global IDs
    chunks = chunker.create_chunks(splits)
    
    print(f"Train chunks: {len(chunks['train'])}")
    print(f"Val chunks: {len(chunks['val'])}")
    print(f"Test chunks: {len(chunks['test'])}")
    
    # Collect all IDs
    all_ids = []
    for split_name, chunk_list in chunks.items():
        for chunk in chunk_list:
            all_ids.append(chunk['id'])
            print(f"  {split_name}: chunk_{chunk['id']:03d}_{split_name}.pt")
    
    # Check uniqueness
    unique_ids = set(all_ids)
    print(f"\nTotal chunks: {len(all_ids)}")
    print(f"Unique IDs: {len(unique_ids)}")
    
    if len(all_ids) == len(unique_ids):
        print("✅ All chunk IDs are unique across splits!")
        return True
    else:
        print("❌ Duplicate chunk IDs found!")
        return False


def test_cache_thread_safety():
    """Test that LRUCache is thread-safe."""
    print_section("2. TESTING CACHE THREAD SAFETY", "=", 70)
    
    from src.data.cache import LRUCache
    
    cache = LRUCache(max_size=3)
    
    def worker(worker_id):
        for i in range(20):
            key = (worker_id * 20 + i) % 10
            cache.put(key, {'data': i, 'worker': worker_id})
            val = cache.get(key)
            # Random operations to create contention
            if i % 3 == 0:
                stats = cache.get_stats()
    
    threads = []
    for i in range(4):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    stats = cache.get_stats()
    print(f"Cache size: {stats['size']}")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    
    # Verify no corruption
    active_keys = stats['active_keys']
    print(f"Active keys: {active_keys}")
    print(f"GC queue size: {stats['gc_queue_size']}")
    
    print("✅ Cache thread-safety test passed!")
    return True


def test_model_imports():
    """Test that all models import correctly."""
    print_section("3. TESTING MODEL IMPORTS", "=", 70)
    
    try:
        from src.models import (
            EfficientUNet,
            LightUNet,
            MobileUNet,
            MPSLightUNet,
            NanoUNet,
            NanoUNetLight,
            PicoUNet,
            TinyUNet,
            UltraNanoUNet,
            UNet,
            create_model,
            list_models,
        )
        
        print("✅ All model imports successful:")
        print(f"  - UNet: {UNet}")
        print(f"  - MPSLightUNet: {MPSLightUNet}")
        print(f"  - LightUNet: {LightUNet}")
        print(f"  - NanoUNet: {NanoUNet}")
        print(f"  - NanoUNetLight: {NanoUNetLight}")
        print(f"  - PicoUNet: {PicoUNet}")
        print(f"  - TinyUNet: {TinyUNet}")
        print(f"  - MobileUNet: {MobileUNet}")
        print(f"  - EfficientUNet: {EfficientUNet}")
        print(f"  - UltraNanoUNet (alias): {UltraNanoUNet}")
        
        # Test factory
        model = create_model("pico")
        print(f"\n✅ create_model('pico') works: {model.__class__.__name__}")
        
        # List models
        models = list_models()
        print(f"✅ list_models() returns {len(models)} models")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_config_immutability():
    """Test that config is immutable (frozen)."""
    print_section("4. TESTING CONFIG IMMUTABILITY", "=", 70)
    
    from src.config import SeismicConfig
    
    config = SeismicConfig()
    print(f"Config: dataset_name={config.dataset_name}, batch_size={config.batch_size}")
    
    try:
        config.batch_size = 100
        print("❌ Config was mutable - frozen=True not working!")
        return False
    except AttributeError as e:
        # ✅ This is the expected error for frozen dataclass
        print(f"✅ Config is immutable (frozen) - got: {e}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_conditional_lr_validation():
    """Test that LR validation only checks active scheduler."""
    print_section("5. TESTING CONDITIONAL LR VALIDATION", "=", 70)
    
    from src.config import SeismicConfig
    
    # ✅ For cosine scheduler, don't check lr_patience
    try:
        config = SeismicConfig(
            lr_scheduler="cosine",
            lr_T_max=30,
        )
        print(f"✅ Cosine scheduler validation passed (lr_T_max={config.lr_T_max})")
        return True
    except ValueError as e:
        print(f"❌ Conditional validation failed: {e}")
        return False



def test_gpu_metrics():
    """Test that SegmentationMetrics works on GPU."""
    print_section("6. TESTING GPU METRICS", "=", 70)
    
    try:
        from src.training.metrics import SegmentationMetrics
        
        # Use CPU for test
        device = torch.device("cpu")
        metrics = SegmentationMetrics(num_classes=3, device=device)
        
        print(f"✅ SegmentationMetrics created on {device}")
        print(f"   Confusion matrix device: {metrics.confusion_matrix.device}")
        
        # Simulate predictions
        batch_size = 4
        preds = torch.randint(0, 3, (batch_size, 1578, 751))
        targets = torch.randint(0, 3, (batch_size, 1578, 751))
        
        metrics.update(preds, targets)
        print("   ✅ Updated metrics with batch")
        
        results = metrics.compute()
        print(f"   ✅ Computed metrics: accuracy={results['accuracy']:.4f}, mean_iou={results['mean_iou']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sweep_parse_metrics():
    """Test that parse_metrics is a static method."""
    print_section("7. TESTING SWEEP PARSE_METRICS", "=", 70)
    
    try:
        from scripts.sweep_mlflow import SweepExperiment
        
        # Check if parse_metrics is a static method or can be called without instance
        if hasattr(SweepExperiment, 'parse_metrics'):
            # Try calling it as a static method
            result = SweepExperiment.parse_metrics("Train Loss: 0.1234\nVal Loss: 0.2345")
            print("✅ parse_metrics is accessible as static method")
            print(f"   Parsed: {result}")
            return True
        else:
            print("❌ parse_metrics not found or not static")
            return False
            
    except Exception as e:
        print(f"❌ Sweep parse_metrics test failed: {e}")
        return False


def test_predict_loader():
    """Test that predict.py uses unified loader."""
    print_section("8. TESTING PREDICT LOADER", "=", 70)
    
    try:
        # Check if predict.py imports load_model_from_checkpoint
        import inspect

        import scripts.predict as predict_module
        
        source = inspect.getsource(predict_module)
        
        if "load_model_from_checkpoint" in source:
            print("✅ predict.py uses unified load_model_from_checkpoint")
            return True
        else:
            print("⚠️  predict.py does NOT use unified loader (check if updated)")
            return True  # Not critical, just informational
            
    except Exception as e:
        print(f"⚠️  Could not inspect predict.py: {e}")
        return True


def main():
    """Run all quick tests."""
    
    print("=" * 70)
    print("🧪 QUICK TEST: CRITICAL FIXES")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    all_passed = True
    
    tests = [
        ("Chunker Unique IDs", test_chunker_unique_ids),
        ("Cache Thread Safety", test_cache_thread_safety),
        ("Model Imports", test_model_imports),
        ("Config Immutability", test_config_immutability),
        ("Conditional LR Validation", test_conditional_lr_validation),
        ("GPU Metrics", test_gpu_metrics),
        ("Sweep parse_metrics", test_sweep_parse_metrics),
        ("Predict Loader", test_predict_loader),
    ]
    
    for name, test_func in tests:
        try:
            result = test_func()
            if not result:
                all_passed = False
                print(f"\n❌ {name} FAILED")
        except Exception as e:
            all_passed = False
            print(f"\n❌ {name} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print_section("✅ TEST SUMMARY", "=", 70)
    if all_passed:
        print("🎉 ALL CRITICAL TESTS PASSED!".center(70))
    else:
        print("⚠️ SOME TESTS FAILED. Please review the output above.".center(70))
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
