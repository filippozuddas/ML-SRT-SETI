# Preprocessing Pipeline

This document describes the data preprocessing steps applied before model inference.

## Overview

The preprocessing pipeline transforms raw radio telescope data into a format suitable for the VAE encoder:

```
Raw: (6, 16, 4096) → Downscale → (6, 16, 512) → Normalize → (6, 16, 512, 1)
```

## Step 1: Downscaling

**Purpose**: Reduce frequency resolution while preserving signal structure.

```python
from src.utils.preprocessing import downscale

# Input: (6, 16, 4096) or (batch, 6, 16, 4096)
# Output: (6, 16, 512) or (batch, 6, 16, 512)
downscaled = downscale(data, factor=8)
```

- Uses local mean pooling (`skimage.transform.downscale_local_mean`)
- Factor of 8: 4096 → 512 frequency bins
- Preserves narrowband signal visibility

## Step 2: Per-Snippet Normalization

**Purpose**: Normalize all 6 observations together to preserve relative contrast.

```python
from src.utils.preprocessing import preprocess

# Input: (6, 16, 512) - single snippet
# Output: (6, 16, 512, 1) - with channel dimension
processed = preprocess(data, add_channel=True)
```

### Algorithm

```python
def normalize_log(data):
    # 1. Log transform
    log_data = np.log(np.abs(data) + 1e-10)
    
    # 2. Shift to zero
    log_data = log_data - log_data.min()
    
    # 3. Scale to [0, 1]
    if log_data.max() > 0:
        log_data = log_data / log_data.max()
    
    return log_data
```

### Key: Per-Snippet vs Per-Observation

| Approach | How | Result |
|----------|-----|--------|
| **Per-snippet** ✓ | Normalize all 6 obs together | Preserves ON/OFF contrast |
| Per-observation ✗ | Normalize each obs separately | Loses relative scaling |

**Why per-snippet matters**: If a signal appears only in ON observations, per-snippet normalization keeps it brighter than OFF. Per-observation would scale both to the same range.

## Complete Pipeline

```python
from src.utils.preprocessing import preprocess, downscale

# Raw observation data: (6, 16, 4096)
raw_snippet = load_observations(files)

# Step 1: Downscale
downscaled = downscale(raw_snippet, factor=8)  # (6, 16, 512)

# Step 2: Normalize + add channel
processed = preprocess(downscaled, add_channel=True)  # (6, 16, 512, 1)

# Ready for encoder
latent = encoder.predict(processed)
```

## Batch Processing

For multiple snippets:

```python
# Input: (batch, 6, 16, 4096)
downscaled = downscale(batch_data, factor=8)  # (batch, 6, 16, 512)
processed = preprocess(downscaled, add_channel=True)  # (batch, 6, 16, 512, 1)
```

## Visual Comparison

See `notebooks/visualize_plates.ipynb` for raw vs preprocessed comparison.

## Code Reference

All preprocessing functions are in: `src/utils/preprocessing.py`

| Function | Purpose |
|----------|---------|
| `downscale(data, factor)` | Frequency downsampling |
| `preprocess(data, add_channel)` | Full preprocessing pipeline |
| `normalize_log(data)` | Log normalization |
| `preprocess_batch(data)` | Numba-optimized batch processing |
