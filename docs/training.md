# Training Guide

This guide covers training the ML-SRT-SETI model from scratch.

## Prerequisites

- Python 3.10+
- TensorFlow 2.15+
- 2x NVIDIA GPU recommended (24GB VRAM each)
- SRT background data plate

## Quick Training

```bash
python scripts/train_large_scale.py \
    --batches 15 \
    --samples 2500 \
    --epochs 100 \
    --plate data/srt_training/srt_backgrounds.npz \
    --output results/my_model/
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batches` | 15 | Number of training batches |
| `--samples` | 2500 | Samples per batch |
| `--epochs` | 100 | Max epochs per batch |
| `--plate` | None | Path to SRT backgrounds .npz |
| `--output` | results/ | Output directory |
| `--batch-size` | 500 | Per-GPU batch size |
| `--resume` | None | Resume from encoder checkpoint |
| `--start-batch` | 0 | Starting batch (for resume) |

## Training Data

### SRT Backgrounds Plate

The plate contains real SRT backgrounds for signal injection:

```python
# Expected format
plate = np.load("srt_backgrounds.npz")
backgrounds = plate["backgrounds"]  # (N, 6, 16, 4096)
```

### Data Generation

Each batch generates fresh synthetic data:
- **True samples**: ETI-like signal in ON observations only
- **False samples**: RFI patterns (signal in all observations)
- **VAE samples**: Mixed data for reconstruction learning

## Multi-GPU Training

Training automatically uses `MirroredStrategy` for multi-GPU:

```
✓ Using MirroredStrategy with 2 GPUs
  Per-GPU batch size: 500
  Effective batch size: 1000
```

## Checkpointing

Saved automatically to output directory:

| File | Description |
|------|-------------|
| `encoder_batch_N.keras` | Encoder after batch N |
| `decoder_batch_N.keras` | Decoder after batch N |
| `encoder_global_best.keras` | Best performing encoder |
| `encoder_final.keras` | Final encoder |
| `random_forest.joblib` | Trained RF classifier |

## Resuming Training

```bash
python scripts/train_large_scale.py \
    --batches 15 \
    --samples 2500 \
    --epochs 100 \
    --plate data/srt_training/srt_backgrounds.npz \
    --output results/my_model/ \
    --resume results/my_model/encoder_global_best.keras \
    --start-batch 10
```

## Monitoring

### Early Stopping
- Monitors `val_false_loss`
- Patience: 15 epochs
- Restores best weights

### Catastrophic Degradation Detection
If a batch performs >5% worse than global best, weights are rolled back.

## Expected Output

```
======================================================================
VAE TRAINING COMPLETE
Total training time: 126.7 minutes (2.1 hours)
Global best: batch 15 with val_loss=0.46
======================================================================

Random Forest Results:
  Accuracy: 0.9771
  AUC-ROC:  0.9958
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| β (KL weight) | 1.5 | Paper default |
| α (clustering weight) | 10.0 | Paper default |
| Learning rate | 1e-4 | Adam optimizer |
| Latent dimension | 8 | Compact but expressive |
| SNR range | 10-50 | Signal strength variation |

## Code Reference

Main training script: `scripts/train_large_scale.py`
