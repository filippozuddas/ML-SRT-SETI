# Training Guide

This guide covers training the ML-SRT-SETI model from scratch.

## Development Environment

The model was developed and trained using **Google Colab** with a local runtime, connected to a dedicated server with the following hardware:

| Component | Specification |
|-----------|---------------|
| GPU | 2× NVIDIA RTX 4090 (24 GB VRAM each) |
| RAM | 250 GB |
| Training strategy | `tf.distribute.MirroredStrategy` (dual GPU) |

Software stack:
- **Python** 3.10
- **TensorFlow** ≥ 2.15 with CUDA support
- **setigen** for synthetic signal injection
- **blimpy** for filterbank I/O

Full dependency list: see `environment.yml` or `requirements.txt`.

## Quick Training

```bash
python experiments/train_large_scale.py \
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
python experiments/train_large_scale.py \
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

## End-to-End Pipeline

The training script runs **three phases** automatically:

1. **Phase 1 — VAE Training**: the β-VAE is trained across multiple batches with fresh synthetic data each batch, using early stopping and automatic rollback to the global best checkpoint if a catastrophic degradation is detected.
2. **Phase 2 — Latent Evaluation**: 2 000 fresh cadences (true + false) are generated and encoded through the trained encoder. The resulting latent vectors are used to produce a latent-space visualization.
3. **Phase 3 — Random Forest**: the 6 per-observation latent vectors of each cadence are concatenated into a single 48D feature vector. A `RandomForestClassifier` (1 000 trees, `max_features='sqrt'`) is trained on a 70/30 split and saved as `.joblib`.

All three outputs — encoder, decoder, and classifier — are saved automatically at the end of the script. No separate step is required.

## Code Reference

Main training script: `experiments/train_large_scale.py`

For model hyperparameters (β, α, latent dimension, etc.), see [Architecture Details](architecture.md).
