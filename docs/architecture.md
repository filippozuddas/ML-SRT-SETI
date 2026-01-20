# System Architecture

This document describes the technical architecture of the ML-SRT-SETI signal detection pipeline.

## Overview

The pipeline uses a two-stage approach:
1. **β-VAE** learns a compressed latent representation of radio observations
2. **Random Forest** classifies cadence patterns as ETI or RFI

## Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                           INPUT                                  │
│      6 observations × 16 time bins × 4096 frequency channels     │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                                │
│  1. Downscale 8x: (6, 16, 4096) → (6, 16, 512)                   │
│  2. Log normalize per-snippet: all 6 obs together                │
│  3. Add channel dim: (6, 16, 512, 1)                             │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       VAE ENCODER                                │
│  9 Conv2D layers → Flatten → Dense(512) → z_mean, z_log_var      │
│  Output: 8-dimensional latent vector per observation             │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LATENT COMBINATION                            │
│  Concatenate 6 latent vectors: 6 × 8 = 48 dimensions             │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RANDOM FOREST                                 │
│  Input: 48D vector                                               │
│  Output: P(ETI) probability                                      │
└──────────────────────────────────────────────────────────────────┘
```

## VAE Encoder Architecture

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Input | (16, 512, 1) | - |
| Conv1 (3×3, 16) stride 2×2 | (8, 256, 16) | 160 |
| Conv2 (3×3, 16) | (8, 256, 16) | 2,320 |
| Conv3 (3×3, 32) stride 2×2 | (4, 128, 32) | 4,640 |
| Conv4 (3×3, 32) | (4, 128, 32) | 9,248 |
| Conv5 (3×3, 32) | (4, 128, 32) | 9,248 |
| Conv6 (3×3, 64) stride 2×2 | (2, 64, 64) | 18,496 |
| Conv7 (3×3, 64) | (2, 64, 64) | 36,928 |
| Conv8 (3×3, 128) | (2, 64, 128) | 73,856 |
| Conv9 (3×3, 256) stride 2×2 | (1, 32, 256) | 295,168 |
| Flatten | (8192,) | - |
| Dense | (512,) | 4,194,816 |
| z_mean | (8,) | 4,104 |
| z_log_var | (8,) | 4,104 |

**Total parameters**: 4,653,088 (17.75 MB)

## Loss Function

The VAE uses a composite loss:

```
Total Loss = Reconstruction + β × KL + α × (True_Clustering + False_Clustering)
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| Reconstruction | 1.0 | Faithful data reconstruction |
| KL Divergence | β=1.5 | Regularize latent space |
| True Clustering | α=10 | ON observations cluster together, separate from OFF |
| False Clustering | α=10 | All observations cluster together (RFI pattern) |

### Clustering Loss Details

**True samples (ETI pattern)**:
- Obs 1, 3, 5 (ON) should cluster together
- Obs 2, 4, 6 (OFF) should cluster together
- ON and OFF clusters should be far apart

**False samples (RFI pattern)**:
- All 6 observations should cluster together

## Random Forest Classifier

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | None |
| min_samples_split | 2 |
| Input features | 48 |
| Output | P(ETI) |

## Key Design Decisions

### Per-Snippet Normalization
The entire 6-observation snippet (6×16×512) is normalized together, not per-observation. This preserves the relative contrast between ON and OFF observations, which is crucial for detecting signals that appear only in ON.

### Latent Dimension (8)
Compact enough to avoid overfitting, expressive enough to capture relevant features.

### 8x Downscaling
Reduces computational load while preserving signal structure. Narrowband signals are still detectable after downscaling.

## Code References

- **VAE Model**: `src/models/vae.py`
- **Preprocessing**: `src/utils/preprocessing.py`
- **Training**: `scripts/train_large_scale.py`
- **Inference**: `src/inference/pipeline_optimized.py`
