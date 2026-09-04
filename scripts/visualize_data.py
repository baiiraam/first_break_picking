#!/usr/bin/env python3
"""
Visualize seismic data samples from preprocessed chunks.
Shows seismograms with ground truth masks and pick positions.
"""

import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def visualize_samples(
    dataset_name: str,
    split: str = "train",
    num_samples: int = 4,
    output_dir: str = "visualizations",
    show: bool = False,
):
    """
    Visualize samples from a dataset split.

    Args:
        dataset_name: Name of the dataset (Halfmile, Brunswick, etc.)
        split: 'train', 'val', or 'test'
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
        show: If True, display plots interactively
    """

    chunk_dir = Path(f"data/chunks/{dataset_name}")
    manifest_path = chunk_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"❌ No manifest found for {dataset_name}")
        return

    # Load manifest
    import json

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Get chunks for the specified split
    split_chunks = [c for c in manifest["chunks"] if c["split"] == split]

    if not split_chunks:
        print(f"❌ No chunks found for split '{split}' in {dataset_name}")
        return

    print(f"\n📊 Visualizing {num_samples} samples from {dataset_name} ({split} set)")
    print(f"   Found {len(split_chunks)} chunks")

    # Collect samples from chunks
    samples = []

    for chunk_info in tqdm(split_chunks, desc="Loading chunks"):
        chunk_path = chunk_dir / chunk_info["filename"]
        try:
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)
            data = chunk["data"]
            mask = chunk["mask"]
            shot_ids = chunk["shot_ids"]

            n_shots = data.shape[0]

            # Randomly select shots from this chunk
            indices = list(range(n_shots))
            random.shuffle(indices)

            for idx in indices[: min(5, n_shots)]:  # Take up to 5 per chunk
                samples.append(
                    {
                        "shot_id": int(shot_ids[idx])
                        if hasattr(shot_ids[idx], "item")
                        else shot_ids[idx],
                        "data": data[idx].numpy(),
                        "mask": mask[idx].numpy(),
                        "chunk": chunk_path.name,
                        "split": split,
                    }
                )

                if len(samples) >= num_samples * 3:  # Collect extra for selection
                    break

            if len(samples) >= num_samples * 3:
                break

        except Exception as e:
            print(f"   ⚠️  Error loading {chunk_path.name}: {e}")
            continue

    if not samples:
        print(f"❌ No samples loaded for {dataset_name} ({split})")
        return

    # Select diverse samples
    if len(samples) > num_samples:
        # Try to select samples with different pick positions
        samples.sort(key=lambda x: np.median(x["mask"] == 2))
        step = len(samples) // num_samples
        selected = [samples[i * step] for i in range(num_samples)]
    else:
        selected = samples[:num_samples]

    # Create output directory
    output_path = Path(output_dir) / dataset_name / split
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Saving visualizations to: {output_path}")

    # Visualize each sample
    for i, sample in enumerate(selected):
        fig = create_sample_figure(sample, dataset_name, i + 1, len(selected))

        # Save figure
        fig_path = output_path / f"sample_{i + 1:02d}_shot_{sample['shot_id']}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✅ Saved: {fig_path.name}")

    # Create summary page
    create_summary_page(selected, output_path, dataset_name, split)

    print(f"\n✅ Visualizations saved to: {output_path}")

    if show:
        # Display the last figure
        fig = create_sample_figure(
            selected[-1], dataset_name, len(selected), len(selected)
        )
        plt.show()
        plt.close(fig)


def create_sample_figure(sample, dataset_name, idx, total):
    """Create a single sample visualization."""

    data = sample["data"]  # (traces, samples)
    mask = sample["mask"]  # (traces, samples)
    shot_id = sample["shot_id"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # 1. Seismogram with pick overlay
    ax1 = axes[0]
    vmin = -np.percentile(np.abs(data), 95)
    vmax = np.percentile(np.abs(data), 95)
    im1 = ax1.imshow(data.T, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
    ax1.set_title(f"Seismogram (Shot {shot_id})", fontsize=14)
    ax1.set_xlabel("Trace", fontsize=12)
    ax1.set_ylabel("Sample", fontsize=12)
    plt.colorbar(im1, ax=ax1, label="Amplitude")

    # 2. Ground truth mask
    ax2 = axes[1]
    im2 = ax2.imshow(mask.T, cmap="tab10", aspect="auto", vmin=0, vmax=2)
    ax2.set_title("Ground Truth Mask", fontsize=14)
    ax2.set_xlabel("Trace", fontsize=12)
    ax2.set_ylabel("Sample", fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
    cbar2.set_label("Class: 0=Before, 1=After, 2=Strip", fontsize=10)

    # 3. Combined overlay
    ax3 = axes[2]
    # Create overlay: seismogram with strip highlighted
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    overlay = np.stack([data_norm, data_norm, data_norm], axis=-1)

    # Highlight strip in red
    strip_mask = (mask == 2).astype(bool)
    overlay[strip_mask, 0] = 1.0  # Red channel
    overlay[strip_mask, 1] = 0.0
    overlay[strip_mask, 2] = 0.0

    ax3.imshow(overlay, aspect="auto")
    ax3.set_title("Seismogram + Strip (Red)", fontsize=14)
    ax3.set_xlabel("Trace", fontsize=12)
    ax3.set_ylabel("Sample", fontsize=12)

    # Add sample info
    info_text = (
        f"Dataset: {dataset_name}\n"
        f"Split: {sample['split']}\n"
        f"Shot ID: {shot_id}\n"
        f"Chunk: {sample['chunk']}\n"
        f"Traces: {data.shape[0]}\n"
        f"Samples: {data.shape[1]}"
    )
    fig.text(
        0.02,
        0.98,
        info_text,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.suptitle(f"Sample {idx}/{total}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig


def create_summary_page(samples, output_path, dataset_name, split):
    """Create a summary HTML page for all samples."""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{dataset_name} - {split} Set Visualizations</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .sample {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .sample img {{ width: 100%; max-width: 600px; border-radius: 4px; }}
            .sample .info {{ margin-top: 10px; font-size: 14px; color: #555; }}
            .sample .info span {{ font-weight: bold; }}
            .summary {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>📊 {dataset_name} - {split.capitalize()} Set Samples</h1>
        
        <div class="summary">
            <p><strong>Total Samples:</strong> {len(samples)}</p>
            <p><strong>Generated:</strong> {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="container">
    """

    for i, sample in enumerate(samples):
        html_content += f"""
            <div class="sample">
                <img src="sample_{i + 1:02d}_shot_{sample["shot_id"]}.png" 
                     alt="Sample {i + 1} - Shot {sample["shot_id"]}">
                <div class="info">
                    <span>Shot:</span> {sample["shot_id"]} &nbsp;|&nbsp;
                    <span>Chunk:</span> {sample["chunk"]} &nbsp;|&nbsp;
                    <span>Traces:</span> {sample["data"].shape[0]} &nbsp;|&nbsp;
                    <span>Samples:</span> {sample["data"].shape[1]}
                </div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    html_path = output_path / "index.html"
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"   ✅ Summary page: {html_path}")


def main():
    """Main function to visualize all datasets."""

    print("=" * 70)
    print("🖼️  DATA VISUALIZATION")
    print("=" * 70)

    # Configuration
    datasets = ["Halfmile", "Brunswick", "Lalor", "Sudbury"]
    num_samples_per_split = 3

    for dataset in datasets:
        print(f"\n{'=' * 70}")
        print(f"📊 Processing: {dataset}")
        print(f"{'=' * 70}")

        # Check if data exists
        chunk_dir = Path(f"data/chunks/{dataset}")
        if not chunk_dir.exists():
            print(f"❌ No data found for {dataset}")
            continue

        # Visualize train, val, test sets
        for split in ["train", "val", "test"]:
            try:
                visualize_samples(
                    dataset_name=dataset,
                    split=split,
                    num_samples=num_samples_per_split,
                    output_dir="visualizations",
                    show=False,
                )
            except Exception as e:
                print(f"   ❌ Error visualizing {dataset} ({split}): {e}")

    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("📁 Results saved to: visualizations/")
    print("\nTo view the summary pages, open:")
    for dataset in datasets:
        print(f"   visualizations/{dataset}/train/index.html")
        print(f"   visualizations/{dataset}/val/index.html")
        print(f"   visualizations/{dataset}/test/index.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
