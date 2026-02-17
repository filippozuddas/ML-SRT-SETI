# ML-SRT-SETI: Machine Learning Signal Detection for SETI

A semi-supervised deep learning pipeline for detecting technosignatures in radio telescope observations. Originally developed for the Green Bank Telescope (GBT), adapted for the Sardinian Radio Telescope (SRT).

## References

- Original paper: [**"A deep-learning search for technosignatures of 820 nearby stars"**](https://arxiv.org/abs/2301.12670)
- Original repository: [**PetchMa/ML_GBT_SETI**](https://github.com/PetchMa/ML_GBT_SETI)

## Overview

This pipeline uses a custom **β-VAE** (Variational Autoencoder) combined with a **Random Forest classifier** to detect potential extraterrestrial signals that:
- Appear only in "ON-source" observations (when pointing at a target)
- Disappear in "OFF-source" observations (when pointing away)
- Show Doppler drift due to non-uniform relative motion between the signal source and the receiver

The system achieves **~99% accuracy** in distinguishing ETI-like signals from RFI (Radio Frequency Interference) on synthetic test data.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/ML-SRT-SETI.git
cd ML-SRT-SETI

# Create environment
conda env create -f environment.yml
conda activate ml-srt-seti

# Or with pip
pip install -r requirements.txt
```

### Run Inference on SRT Data

```bash
# Process observations from a listfile
python -m src.inference.cli listfile \
    --list-file data/targets.txt \
    --encoder models/encoder_final.keras \
    --classifier models/random_forest.joblib \
    --optimized --batch-size 2048 \
    --threshold 0.7 \
    --output results/
```

### Train Your Own Model

```bash
# Large-scale training with SRT backgrounds
python experiments/train_large_scale.py \
    --batches 15 \
    --samples 2500 \
    --epochs 100 \
    --plate data/srt_training/srt_backgrounds.npz \
    --output models/
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│   .h5 files → Downscale 8x → Per-snippet Normalize → VAE    │
│                                                             │
│    Encoder → Latent (8D) → Combine 6 obs → RF → P(ETI)      │
└─────────────────────────────────────────────────────────────┘
```

| Component | Description |
|-----------|-------------|
| **Preprocessing** | Log normalization per-snippet (6×16×512), 8x frequency downscaling |
| **β-VAE Encoder** | 9-layer CNN → 8-dimensional latent space |
| **Contrastive Loss** | Clustering loss to separate ON/OFF patterns |
| **Random Forest** | Classifies 48D cadence features (6 obs × 8 latent dims) |

## Project Structure

```
ML-SRT-SETI/
├── analysis/         # Scripts for post-processing and analysis (e.g., evaluating cadences, analyzing ON-OFF patterns, and debugging features)
├── configs/          # YAML configuration files for training and inference pipelines (default, SRT-specific, testing)
├── data/             # Datasets, metadata JSONs, and target lists for inference across different bands (e.g., 6GHz, 18GHz)
├── docs/             # Detailed project documentation (architecture, preprocessing, training, inference)
├── experiments/      # Executable scripts for running specific tasks (full training pipelines, evaluations, Voyager tests)
├── notebooks/        # Jupyter notebooks for interactive data exploration and sample visualization
├── results/          # Generated outputs and artifacts:
│   ├── inference/    # Outputs from inference runs on target data
│   ├── models/       # Saved trained model weights (.keras, .h5) and Random Forest classifiers (.joblib)
│   ├── plots/        # Visualizations of candidates and model comparisons
│   └── visualizations/ # Latent space, traversal, and SNR sensitivity plots
├── src/              # Core source code of the project:
│   ├── data/         # Data loading, dataset building, and synthetic signal/noise generation
│   ├── inference/    # Inference pipelines and plotting utilities 
│   ├── models/       # Neural network architectures (encoders, decoders, sampling layers)
│   └── utils/        # Shared utilities for configuration, preprocessing, and visualization
├── environment.yml   # Conda environment specification
└── requirements.txt  # Python pip dependencies
```

## Documentation

- [Architecture Details](docs/architecture.md)
- [Data & Preprocessing](docs/data.md)
- [Training Guide](docs/training.md)
- [Inference Guide](docs/inference.md)


## Key Features

- **Optimized Pipeline**: Process 67M-channel files with chunked loading
- **Multi-GPU Training**: MirroredStrategy for dual-GPU training
- **SRT Adaptation**: Real SRT backgrounds for realistic training data
- **Per-snippet Normalization**: Preserves ON/OFF relative contrast 
- **Overlap Mode**: 50% overlapping windows for better signal coverage
