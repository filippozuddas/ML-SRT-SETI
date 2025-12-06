# SETI Signal Detector

A modern reimplementation of the semi-unsupervised SETI signal detection algorithm 
using β-VAE and Random Forest classifiers.

## Overview

This project implements a machine learning pipeline for detecting technosignatures
in radio telescope data, following the ON-OFF cadence observation pattern used in SETI.

## Architecture

1. **β-VAE Encoder**: Extracts 8-dimensional latent representations from spectrograms
2. **Random Forest Classifier**: Classifies cadence patterns based on latent vectors

## Project Structure

```
seti_detector/
├── configs/           # Configuration files
├── src/
│   ├── data/          # Data generation and loading
│   ├── models/        # VAE and classifier models
│   ├── training/      # Training loops and losses
│   ├── inference/     # Search pipeline
│   └── utils/         # Utilities and preprocessing
├── scripts/           # Training and inference scripts
├── tests/             # Unit and integration tests
└── notebooks/         # Jupyter notebooks for analysis
```

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate seti_detector

# Or with pip
pip install -r requirements.txt
```

## Usage

### Phase 1: Training with Simulated Data

```bash
# Train the VAE
python scripts/train_vae.py --config configs/default.yaml

# Train the Random Forest classifier
python scripts/train_classifier.py --config configs/default.yaml
```

### Phase 2: Training with Real SRT Data

```bash
# Train with real observation backgrounds
python scripts/train_vae.py --config configs/srt_config.yaml --plates /path/to/plates
```

## Hardware Requirements

- GPU with 24+ GB VRAM recommended (tested on RTX 4090)
- Multi-GPU training supported via tf.distribute.MirroredStrategy

## References

- Original paper: [2301.12670](https://arxiv.org/abs/2301.12670)
- Original repository: [PetchMa/ML_GBT_SETI](https://github.com/PetchMa/ML_GBT_SETI)
