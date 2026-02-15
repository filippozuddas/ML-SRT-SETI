# Inference Guide

This guide covers running inference on SRT observations.

## Prerequisites

Running inference requires the three artifacts produced by the training script (`experiments/train_large_scale.py`):

| Artifact | File | Description |
|----------|------|-------------|
| Encoder | `encoder_final.keras` | Trained VAE encoder |
| Classifier | `random_forest.joblib` | Random Forest trained on latent features |
| Listfile | `targets.txt` | Target names and 6 `.h5` observation paths per line |

## Basic Usage

```bash
python -m src.inference.cli listfile \
    --list-file data/targets.txt \
    --encoder models/encoder_final.keras \
    --classifier models/random_forest.joblib \
    --output results/
```

## Listfile Format

Each line contains a target name and 6 observation file paths:

```
TARGET_NAME|path/to/obs1.fil,path/to/obs2.fil,path/to/obs3.fil,path/to/obs4.fil,path/to/obs5.fil,path/to/obs6.fil
```

Example:
```
TIC123456789|/data/obs_A1.fil,/data/obs_B.fil,/data/obs_A2.fil,/data/obs_C.fil,/data/obs_A3.fil,/data/obs_D.fil
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--list-file` | Required | Path to listfile |
| `--encoder` | Required | Path to encoder model |
| `--classifier` | Required | Path to RF classifier |
| `--threshold` | 0.5 | ETI detection threshold |
| `--output` | None | Output directory |
| `--optimized` | False | Use chunked pipeline for large files |
| `--chunks` | 8 | Number of frequency chunks |
| `--batch-size` | 256 | GPU batch size for encoding |
| `--overlap` | False | Use 50% overlapping windows |

## Optimized Pipeline

For large files (>1M channels), use the optimized chunked pipeline:

```bash
python -m src.inference.cli listfile \
    --list-file data/targets.txt \
    --encoder models/encoder_final.keras \
    --classifier models/random_forest.joblib \
    --optimized --chunks 4 --batch-size 2048 \
    --threshold 0.7 \
    --overlap \
    --output results/
```

Benefits:
- Processes files chunk by chunk (reduces memory)
- Larger batch sizes for faster GPU encoding
- Overlap mode for better signal coverage (see below)

## Overlapping Search

### The Problem

Without overlap, the frequency band is sliced into **non-overlapping 4096-channel windows**. A real signal whose energy straddles the boundary between two adjacent windows may be split across both, reducing the SNR in each and potentially causing a missed detection.

```
Non-overlapping (default):
|----4096----|----4096----|----4096----|
             ↑ signal split here
```

### The Solution

With `--overlap`, windows advance by **2048 channels** (50% overlap). Every point in the band is covered by at least two windows, so a signal that is split in one window will be fully captured in the next:

```
Overlapping (--overlap):
|----4096----|
       |----4096----|
              |----4096----|
                    ↑ signal always fully inside at least one window
```

### Usage

```bash
python -m src.inference.cli listfile \
    --list-file data/targets.txt \
    --encoder models/encoder_final.keras \
    --classifier models/random_forest.joblib \
    --optimized --overlap \
    --output results/
```

### Trade-offs

| | Without overlap | With overlap |
|--|-----------------|--------------|
| Snippets | N | ~2N |
| Coverage | Signals at boundaries may be split | ≥99.99% coverage |
| Runtime | Baseline | ~2× |

## Output Files

```
results/
├── inference_summary.csv       # Summary of all targets
├── TARGET_NAME/
│   ├── metadata.json           # Target metadata
│   ├── all_snippets.csv        # All processed snippets
│   └── candidates.csv          # Candidates above threshold
```

### Summary CSV Columns

| Column | Description |
|--------|-------------|
| target | Target name |
| n_snippets | Total snippets processed |
| n_detections | Candidates above threshold |
| top_freq_mhz | Frequency of top candidate |
| top_probability | P(ETI) of top candidate |
| status | Processing status |

## Parallel Processing

### Two Sessions (Append Mode)

Results automatically append to the same summary file:

```bash
# Session 1
CUDA_VISIBLE_DEVICES=0 python -m src.inference.cli listfile \
    --list-file targets_part1.txt ...

# Session 2
CUDA_VISIBLE_DEVICES=1 python -m src.inference.cli listfile \
    --list-file targets_part2.txt ...
```

Both write to the same `inference_summary.csv`.

## Plotting Candidates

After inference, generate candidate plots:

```bash
python -m src.inference.plot_candidates \
    --candidates results/TARGET_NAME/candidates.csv \
    --files obs1.fil,obs2.fil,obs3.fil,obs4.fil,obs5.fil,obs6.fil \
    --output results/TARGET_NAME/plots/
```

## Threshold Selection

The `--threshold` parameter controls the ETI detection sensitivity. A value of **0.7** is recommended for balanced precision/recall. Use lower values (0.5) for maximum recall or higher values (0.9) for high precision.

## VRAM Usage

Typical VRAM usage with optimized pipeline:
- ~3 GB per process
- Multiple processes can share a 24GB GPU

## Code Reference

- CLI: `src/inference/cli.py`
- Pipeline: `src/inference/pipeline_optimized.py`
- Plotting: `src/inference/plot_candidates.py`
