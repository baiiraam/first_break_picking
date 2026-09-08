# Seismic First Break Picking - Technical Explanation

## 📋 **What is First Break Picking?**

**First break picking** is the process of identifying the first arrival time of seismic waves on a seismogram. This is a critical step in seismic data processing because:
- It determines the **velocity model** for depth conversion
- It's used for **static corrections** (weathering and elevation)
- It helps identify **near-surface anomalies**
- It's essential for **seismic imaging** quality control

### **The Challenge**
Traditionally, first break picking is done manually by geophysicists, which is:
- **Time-consuming**: Thousands of traces per survey
- **Subjective**: Different interpreters may pick differently
- **Error-prone**: Noisy data makes picking difficult
- **Expensive**: Requires skilled personnel

---

## 🎯 **Our Approach: Deep Learning Segmentation**

Instead of directly predicting the pick time (regression), we reformulate the problem as **3-class semantic segmentation**:

### **Class Definitions**
| Class | Label | Description | Color |
|-------|-------|-------------|-------|
| **Class 0** | Before | Samples before the first break | Blue |
| **Class 1** | After | Samples after the first break | Green |
| **Class 2** | Strip | A narrow band around the first break | Red |

### **Why Segmentation?**
1. **Spatial Context**: The model considers neighboring traces and samples
2. **Robustness**: Segmentation is more robust to noise than direct pick prediction
3. **Uncertainty**: The strip width provides a measure of pick uncertainty
4. **Class Imbalance**: The strip class is rare, making it a focused learning target

---

## 🔬 **The Data Pipeline**

### **1. Input Data (HDF5 Format)**
Each dataset is stored as an HDF5 file with the following structure:
```
TRACE_DATA/
└── DEFAULT/
    ├── SHOTID      # Shot ID for each trace
    ├── data_array  # Seismic amplitudes (n_traces × n_samples)
    └── SPARE1      # First break picks (in milliseconds)
```

### **2. Shot-Level Processing**

For each shot, the pipeline:
1. **Extracts** the shot data and picks
2. **Converts** picks from milliseconds to samples
3. **Creates** a 3-class mask
4. **Pads/Crops** to a fixed number of traces
5. **Validates** mask quality

### **3. Critical: Unit Conversion**

The most important preprocessing step is converting `SPARE1` from **milliseconds to samples**:

```
Sampling Interval = Recording Time / Number of Samples
Sample Index = SPARE1 / Sampling Interval
```

**Example (Halfmile)**:
```
Recording Time: 1500ms
Samples: 751
Sampling Interval: 1500/751 = 2.0 ms/sample
SPARE1 = 881ms → 881/2.0 = 440 samples ✅ Valid (0-750)
```

**Without this conversion**, the model would learn the wrong positions!

### **4. Mask Creation**

The mask is created using vectorized operations:

```python
# Class mapping:
# -1: Unlabeled (ignored in training)
#  0: Before (samples < pick - half_width)
#  2: Strip (pick - half_width ≤ samples ≤ pick + half_width)
#  1: After (samples > pick + half_width)

for each trace:
    strip_mask = (samples >= pick - half_width) & (samples <= pick + half_width)
    after_mask = samples > pick + half_width
    
    mask[strip_mask] = 2
    mask[after_mask] = 1
    mask[invalid_picks] = -1
```

### **5. Chunking & Caching**

**Why chunking?**
- Individual HDF5 files are too large to load entirely in memory
- Training requires random access to shots
- Chunking enables efficient shuffling and batching

**Chunk Structure**:
```
chunk_001_train.pt
├── data: (69 × 1578 × 751) float32  # 69 shots
├── mask: (69 × 1578 × 751) int64
├── shot_ids: list[int]
├── split: "train"
└── chunk_id: 1
```

**LRU Cache**:
- Caches recently used chunks in memory
- Configurable size (default: 5 chunks)
- Automatically evicts oldest chunks when full
- Dramatically speeds up training

---

## 🏗️ **Model Architecture: U-Net Family**

### **Base U-Net Architecture**

U-Net is ideal for segmentation tasks because it:
1. **Preserves spatial information** through skip connections
2. **Captures multi-scale features** via the encoder-decoder structure
3. **Outputs same resolution** as input (pixel-wise classification)

```
Input: (B, 1, H, W)
         │
    ┌────▼────┐
    │ Encoder │  → Downsampling (captures context)
    │  (↓2×)  │
    └────┬────┘
         │
    ┌────▼────┐
    │Bottleneck│  → Deepest features
    └────┬────┘
         │
    ┌────▼────┐
    │ Decoder │  → Upsampling + Skip connections
    │  (↑2×)  │
    └────┬────┘
         │
    ┌────▼────┐
    │ Output  │  → (B, 3, H, W) segmentation mask
    └─────────┘
```

### **Model Variants**

| Model | Params | Speed | Memory | Best For |
|-------|--------|-------|--------|----------|
| **PicoUNet** | 2K | ⚡ Instant | 🟢 Low | Testing pipeline |
| **NanoUNet** | 10K | ⚡ Very Fast | 🟢 Low | Quick validation |
| **TinyUNet** | 50K | ⚡ Fast | 🟢 Low | Rapid prototyping |
| **MPSLightUNet** | 1.7M | 🔥 2× Faster | 🟡 Medium | **Apple Silicon** |
| **LightUNet** | 2.5M | 🔥 2× Faster | 🟡 Medium | Balanced performance |
| **MobileUNet** | 3.5M | 🟡 Moderate | 🟠 High | Transfer learning |
| **EfficientUNet** | 5M | 🟡 Moderate | 🟠 High | Transfer learning |
| **Full UNet** | 31M | 🐢 Slow | 🔴 Very High | Maximum capacity |

### **Why Multiple Variants?**

Different projects have different constraints:
- **Limited GPU memory** → Use smaller models
- **Training time constraints** → Use faster models
- **High accuracy requirement** → Use larger models
- **MPS (Apple Silicon)** → Use MPSLightUNet

---

## 🧮 **Loss Functions**

### **Problem: Class Imbalance**

The 3 classes are highly imbalanced:
- **Strip (Class 2)**: Only ~1% of pixels
- **Before (Class 0)**: ~20% of pixels
- **After (Class 1)**: ~70% of pixels

If we use standard Cross Entropy, the model will ignore the strip class.

### **Solution: Combo Loss**

We combine three loss functions:

**1. Weighted Cross Entropy**
```
Loss_CE = -Σ w_c * y_c * log(p_c)
```
- Gives higher weight to the strip class

**2. Focal Loss**
```
Loss_Focal = -α * (1 - p_t)^γ * log(p_t)
```
- Focuses on hard-to-classify examples
- Reduces the contribution of easy examples

**3. Dice Loss**
```
Loss_Dice = 1 - (2 * |A∩B|) / (|A| + |B|)
```
- Directly optimizes IoU
- Handles class imbalance naturally

**Final Loss**:
```
Loss_Combo = (1 - dice_weight) * (0.5*Loss_CE + 0.5*Loss_Focal) 
           + dice_weight * Loss_Dice
```

### **Ignore Index: Handling Unlabeled Data**

Unlabeled traces are assigned `-1` in the mask:
- They are **ignored** in loss computation
- They do **not** contribute to gradients
- This prevents the model from learning from bad data

**All loss functions correctly handle `-1` values.**

---

## 📊 **Evaluation Metrics**

### **Segmentation Metrics**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Pixel Accuracy** | Correct predictions / Total pixels | Overall performance |
| **Mean IoU** | Average IoU across classes | Segmentation quality |
| **Mean F1** | Average F1 across classes | Balance of precision/recall |
| **Class-wise IoU** | IoU per class | Strip IoU (most important!) |

**Class 2 (Strip) IoU is the most important metric** because it directly measures pick accuracy.

### **First Break Metrics**

From the segmentation mask, we extract the pick position:
```
pick = median(strip_indices)
```

Then we compute:
| Metric | Description |
|--------|-------------|
| **Mean Absolute Error (MAE)** | Average pick error in samples |
| **Std Error** | Variability of errors |
| **Accuracy within ±3 samples** | % of picks within 3 samples |
| **Error Distribution** | Percentiles of errors |

---

## 🚀 **Training Pipeline**

### **1. Hardware Detection**

The system automatically detects:
- **Device Type**: MPS, CUDA, or CPU
- **Memory Available**: RAM, GPU memory, MPS limits
- **Optimal Settings**: Batch size, cache size, memory limit

### **2. Auto-Configuration**

Based on hardware and dataset characteristics:

```python
available_mb = memory_gb * 1024
base_memory_mb = model_profile.base_memory_mb
safe_remaining_mb = (available_mb - base_memory_mb) * 0.8

optimal_batch = min(
    safe_remaining_mb / memory_per_batch_mb,
    recommended_batch,
    total_shots
)

optimal_cache = min(
    remaining_mb / memory_per_cache_mb,
    recommended_cache,
    num_chunks
)
```

### **3. MPS Warmup**

On Apple Silicon, the first forward pass triggers JIT compilation:
- **Duration**: 2-10 minutes (normal!)
- **Why**: Shaders are compiled on first use
- **After warmup**: Training is 2× faster than CPU

### **4. Training Loop**

Each epoch:
1. **Training Phase**:
   - Forward pass → Compute loss → Backward pass
   - Update metrics (IoU, F1, Accuracy)
   - Log to TensorBoard & MLflow

2. **Validation Phase**:
   - No gradients (evaluation mode)
   - Compute validation loss and metrics
   - Check for early stopping

3. **Logging**:
   - TensorBoard: Loss curves, metrics, images
   - MLflow: Parameters, metrics, model versions
   - Console: Progress bars, summary

### **5. Error Recovery**

If memory error occurs:
1. **Clear memory**: Empty cache, garbage collect
2. **Try next variant**: Reduce batch size, cache size, or memory limit
3. **Progress fallback**: 5 levels (Optimal → Minimal)
4. **Skip dataset**: Continue to next dataset if all variants fail

---

## 📦 **Model Versioning (MLflow)**

### **Model Registry**

Each trained model is registered with:
- **Version**: Auto-incremented
- **Aliases**: `champion`, `challenger`, `staging`
- **Tags**: Dataset, model type, epoch, loss, metrics
- **Artifacts**: Model weights, config, sample predictions

### **Alias Management**

| Alias | Description |
|-------|-------------|
| **champion** | Best performing model |
| **challenger** | Current model being evaluated |
| **staging** | Latest model (intermediate) |

When a new model outperforms the champion:
1. Old champion → `challenger`
2. New model → `champion`
3. Latest model → `staging`

---

## 🔧 **Key Technical Decisions**

### **1. Why Segmentation over Regression?**

| Aspect | Regression | Segmentation |
|--------|------------|--------------|
| **Output** | Single value | Full spatial map |
| **Context** | Limited | Full trace context |
| **Robustness** | Lower | Higher |
| **Uncertainty** | Hard to estimate | Strip width provides uncertainty |

### **2. Why 3 Classes instead of Binary?**

Binary (Before/After) would lose information about pick location. The strip class:
- Provides a **target region** for the model to learn
- Gives a **measure of uncertainty** (strip width)
- Makes the problem **easier to learn** (8 samples vs exact point)

### **3. Why Ignore Unlabeled Traces?**

Some traces have invalid picks:
- **No pick**: SPARE1 = 0 or -1
- **Bad pick**: Out of range
- **Low quality**: Unreliable label

Ignoring them:
- Prevents the model from learning from bad data
- Maintains training stability
- Does not waste capacity on poor labels

### **4. Why Chunking?**

| Without Chunking | With Chunking |
|------------------|---------------|
| Load entire dataset | Load chunks on-demand |
| High memory usage | Memory efficient |
| Slow shuffling | Fast random access |
| Limited dataset size | Handles large datasets |

---

## 🎯 **Expected Results**

### **Typical Performance**

| Model | Epochs | Val IoU | Strip IoU | MAE (samples) |
|-------|--------|---------|-----------|---------------|
| PicoUNet | 2 | 0.3-0.4 | 0.2-0.3 | 3-5 |
| PicoUNet | 30 | 0.4-0.5 | 0.3-0.4 | 2-3 |
| MPSLightUNet | 30 | 0.5-0.6 | 0.4-0.5 | 1-2 |
| UNet | 30 | 0.6-0.7 | 0.5-0.6 | <1 |

### **Training Time**

| Model | Epoch Time | 30 Epochs |
|-------|------------|-----------|
| PicoUNet | ~10 seconds | ~5 minutes |
| MPSLightUNet | ~5 minutes | ~2.5 hours |
| UNet | ~10 minutes | ~5 hours |

---

## ✅ **Summary: Preprocessing Fix Confirmed**

| Issue | Status | Verification |
|-------|--------|--------------|
| **Unit mismatch** | ✅ Fixed | SPARE1 converted from ms to samples |
| **All picks valid** | ✅ Verified | 0 out-of-bounds errors |
| **Mask classes** | ✅ Correct | -1, 0, 1, 2 present |
| **Strip placement** | ✅ Accurate | 1.07% of pixels (matches strip_width) |
| **Ignore index** | ✅ Working | Loss functions handle -1 |
| **Sampling interval** | ✅ Configurable | Per-dataset settings |