#!/usr/bin/env python3
"""
Test script for model factory and all model architectures.
Verifies creation, forward pass, parameter counts, and channel adaptation.
"""

import os
import sys
import time

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.factory import (
    MODEL_REGISTRY,
    create_model,
    get_all_model_info,
    get_model_by_params,
    get_model_info,
)


def print_section(title: str, char: str = "=", width: int = 70):
    """Print a formatted section header."""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def test_model_creation():
    """Test that all models can be created."""
    print_section("1. TESTING MODEL CREATION", "=", 70)

    models = list(MODEL_REGISTRY.keys())
    print(f"Found {len(models)} models: {models}")

    failed_models = []

    for model_name in models:
        try:
            # Create with default params
            model = create_model(model_name)

            # Get info
            info = get_model_info(model_name)
            param_count = info.get("params", "N/A")

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())

            print(
                f"✅ {model_name:12s} | params: {total_params:>10,} | class: {model.__class__.__name__}"
            )

            # Check if model has expected attributes
            if not hasattr(model, "forward"):
                print(f"   ⚠️  Model {model_name} has no forward method")
                failed_models.append(model_name)

        except Exception as e:
            print(f"❌ {model_name:12s} | Failed to create: {e}")
            failed_models.append(model_name)

    if failed_models:
        print(f"\n⚠️  Failed models: {failed_models}")
    else:
        print("\n✅ All models created successfully!")

    return failed_models


def test_forward_pass():
    """Test forward pass with different input sizes."""
    print_section("2. TESTING FORWARD PASS", "=", 70)

    # Test with different input shapes
    test_shapes = [
        (1, 1, 1578, 751),  # Standard Halfmile shape
        (2, 1, 1000, 500),  # Smaller shape
        (1, 1, 2000, 1000),  # Larger shape
    ]

    models_to_test = ["pico", "tiny", "nano", "mpslight", "light", "unet"]

    for model_name in models_to_test:
        if model_name not in MODEL_REGISTRY:
            continue

        print(f"\n🔬 Testing {model_name}:")

        try:
            model = create_model(model_name)
            model.eval()

            for i, shape in enumerate(test_shapes):
                batch_size, channels, height, width = shape
                x = torch.randn(*shape)

                with torch.no_grad():
                    output = model(x)

                print(f"   Input {i + 1}: {shape} → Output: {tuple(output.shape)}")

                # Check output shape matches input spatial dimensions
                assert output.shape[0] == batch_size, (
                    f"Batch size mismatch: {output.shape[0]} vs {batch_size}"
                )
                assert output.shape[2] == height, (
                    f"Height mismatch: {output.shape[2]} vs {height}"
                )
                assert output.shape[3] == width, (
                    f"Width mismatch: {output.shape[3]} vs {width}"
                )

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False

    print("\n✅ All forward passes completed successfully!")
    return True


def test_pretrained_models():
    """Test pretrained models (EfficientUNet, MobileUNet)."""
    print_section("3. TESTING PRETRAINED MODELS", "=", 70)

    pretrained_models = ["efficient", "mobile"]
    test_shapes = [(1, 1, 1578, 751), (1, 1, 1000, 500)]

    for model_name in pretrained_models:
        if model_name not in MODEL_REGISTRY:
            continue

        print(f"\n🔬 Testing {model_name} with pretrained weights:")

        # Test with freeze_encoder=True (default)
        try:
            print("   With freeze_encoder=True (default):")
            model = create_model(model_name, pretrained=True, freeze_encoder=True)
            model.eval()

            # Check if encoder is frozen
            frozen_count = 0
            total_encoder_params = 0
            for name, param in model.encoder.named_parameters():
                total_encoder_params += 1
                if not param.requires_grad:
                    frozen_count += 1

            print(
                f"   Encoder frozen: {frozen_count}/{total_encoder_params} params frozen"
            )

            # Forward pass
            x = torch.randn(1, 1, 1578, 751)
            with torch.no_grad():
                output = model(x)
            print(f"   Forward pass output shape: {tuple(output.shape)}")

        except Exception as e:
            print(f"   ❌ Failed with freeze_encoder=True: {e}")
            return False

        # Test with freeze_encoder=False
        try:
            print("   With freeze_encoder=False (trainable):")
            model = create_model(model_name, pretrained=True, freeze_encoder=False)
            model.eval()

            # Check if encoder is trainable
            trainable_count = 0
            for name, param in model.encoder.named_parameters():
                if param.requires_grad:
                    trainable_count += 1

            print(f"   Encoder trainable: {trainable_count} params trainable")

            # Forward pass
            x = torch.randn(1, 1, 1578, 751)
            with torch.no_grad():
                output = model(x)
            print(f"   Forward pass output shape: {tuple(output.shape)}")

        except Exception as e:
            print(f"   ❌ Failed with freeze_encoder=False: {e}")
            return False

    print("\n✅ All pretrained models tested successfully!")
    return True


def test_channel_adaptation():
    """Test that models handle different input channels."""
    print_section("4. TESTING CHANNEL ADAPTATION", "=", 70)

    test_channels = [1, 3, 4]
    models_to_test = ["unet", "efficient", "mobile", "light"]

    for model_name in models_to_test:
        if model_name not in MODEL_REGISTRY:
            continue

        print(f"\n🔬 Testing {model_name} with different input channels:")

        for in_channels in test_channels:
            try:
                # Create model with custom in_channels
                model = create_model(model_name, in_channels=in_channels)

                # Forward pass
                x = torch.randn(1, in_channels, 100, 100)
                with torch.no_grad():
                    output = model(x)

                print(
                    f"   Input channels: {in_channels} → Output shape: {tuple(output.shape)}"
                )

            except Exception as e:
                print(f"   ❌ Failed with {in_channels} channels: {e}")
                return False

    print("\n✅ All channel adaptation tests passed!")
    return True


def test_model_filtering():
    """Test model filtering by parameters and features."""
    print_section("5. TESTING MODEL FILTERING", "=", 70)

    try:
        # Get models with param counts
        print("All models with parameter counts:")
        all_info = get_all_model_info()
        for name, info in all_info.items():
            if info["params"] is not None:
                print(
                    f"   {name:12s}: {info['params']:>12,} params - {info['description']}"
                )
            else:
                print(f"   {name:12s}: {'N/A':>12} params - {info['description']}")

        # Filter by parameter count
        print("\nModels with < 100K params:")
        small_models = get_model_by_params(max_params=100_000)
        for name in small_models:
            info = all_info[name]
            print(f"   {name:12s}: {info['params']:>12,} params")

        print("\nModels with 1M-10M params:")
        medium_models = get_model_by_params(min_params=1_000_000, max_params=10_000_000)
        for name in medium_models:
            info = all_info[name]
            print(f"   {name:12s}: {info['params']:>12,} params")

        print("\nModels with pretrained support:")
        pretrained_models = get_model_by_params(supports_pretrained=True)
        for name in pretrained_models:
            info = all_info[name]
            print(f"   {name:12s}: {info['params']:>12,} params")

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

    print("\n✅ All filtering tests passed!")
    return True


def test_memory_usage():
    """Test memory usage of different models."""
    print_section("6. TESTING MEMORY USAGE", "=", 70)

    try:
        import gc

        import psutil

        models_to_test = ["pico", "nano", "tiny", "mpslight", "light", "unet"]

        print(f"{'Model':<12} {'Params':>12} {'Memory (MB)':>15} {'Time (ms)':>12}")
        print("-" * 55)

        for model_name in models_to_test:
            if model_name not in MODEL_REGISTRY:
                continue

            # Force garbage collection
            gc.collect()

            # Measure memory before
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024

            # Create model
            model = create_model(model_name)
            params = sum(p.numel() for p in model.parameters())

            # Measure memory after
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024

            # Forward pass timing
            x = torch.randn(1, 1, 1578, 751)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start = time.perf_counter()
            with torch.no_grad():
                output = model(x)
            end = time.perf_counter()

            forward_time = (end - start) * 1000  # ms

            print(
                f"{model_name:<12} {params:>12,} {mem_after - mem_before:>14.1f} MB {forward_time:>10.2f} ms"
            )

            # Cleanup
            del model
            del x
            gc.collect()

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

    print("\n✅ Memory usage test completed!")
    return True


def test_device_compatibility():
    """Test models on different devices."""
    print_section("7. TESTING DEVICE COMPATIBILITY", "=", 70)

    # Detect available devices
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")

    print(f"Available devices: {devices}")

    models_to_test = ["pico", "mpslight", "unet"]

    for model_name in models_to_test:
        if model_name not in MODEL_REGISTRY:
            continue

        print(f"\n🔬 Testing {model_name}:")

        for device in devices:
            try:
                if device == "mps" and model_name == "unet":
                    print("   ⚠️  Skipping UNet on MPS (large model)")
                    continue

                device_obj = torch.device(device)
                model = create_model(model_name).to(device_obj)
                model.eval()

                x = torch.randn(1, 1, 100, 100).to(device_obj)

                with torch.no_grad():
                    output = model(x)

                print(f"   ✅ {device:>6}: Output shape {tuple(output.shape)}")

                # Cleanup
                del model
                del x
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()

            except Exception as e:
                print(f"   ❌ {device:>6}: Failed - {str(e)[:50]}")

    print("\n✅ Device compatibility test completed!")


def test_model_registry_consistency():
    """Test that model registry is consistent."""
    print_section("8. TESTING MODEL REGISTRY CONSISTENCY", "=", 70)

    issues = []

    for model_name, entry in MODEL_REGISTRY.items():
        # Check required keys
        required = ["class", "params", "description", "supports"]
        for key in required:
            if key not in entry:
                issues.append(f"Missing '{key}' in {model_name}")

        # Check that params match supports
        if "params" in entry and "supports" in entry:
            for param in entry["params"]:
                if param not in entry["supports"] and param not in [
                    "in_channels",
                    "out_channels",
                ]:
                    issues.append(
                        f"Parameter '{param}' in {model_name} not in supports list"
                    )

        # Check that class can be instantiated
        try:
            model_class = entry["class"]
            if "in_channels" not in entry["params"]:
                entry["params"]["in_channels"] = 1
            if "out_channels" not in entry["params"]:
                entry["params"]["out_channels"] = 3
            # Test instantiation
            model_class(**entry["params"])
        except Exception as e:
            issues.append(f"Cannot instantiate {model_name}: {e}")

    if issues:
        print("⚠️  Found issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Registry is consistent!")

    return len(issues) == 0


def main():
    """Run all tests."""
    print_section("🧪 TESTING MODEL FACTORY AND ARCHITECTURES", "=", 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    all_passed = True

    # Run tests
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Pretrained Models", test_pretrained_models),
        ("Channel Adaptation", test_channel_adaptation),
        ("Model Filtering", test_model_filtering),
        ("Memory Usage", test_memory_usage),
        ("Device Compatibility", test_device_compatibility),
        ("Registry Consistency", test_model_registry_consistency),
    ]

    for name, test_func in tests:
        try:
            result = test_func()
            if result is False:
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
        print("🎉 ALL TESTS PASSED! Model factory is working correctly.".center(70))
    else:
        print("⚠️ SOME TESTS FAILED. Please review the output above.".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
