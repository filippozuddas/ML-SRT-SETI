#!/usr/bin/env python3
"""
Visualize where the top 8 candidates fall in the latent space.
This helps diagnose domain shift issues between training and test data.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import preprocess, downscale
from src.data.cadence_generator import CadenceGenerator, CadenceParams
from multiprocessing import Pool

print("=" * 70)
print("LATENT SPACE DIAGNOSTIC - TOP 8 CANDIDATES")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = Path("/content/filippo/ML-SRT-SETI/assets/top_8_data.pkl")
ENCODER_PATH = Path("/content/filippo/ML-SRT-SETI/results/real_obs_training/encoder_final.keras")
PLATE_PATH = Path("/content/filippo/ML-SRT-SETI/data/srt_training/srt_backgrounds.npz")
OUTPUT_PATH = Path("/content/filippo/ML-SRT-SETI/results/real_obs_training/latent_diagnostic.png")

# ============================================
# LOAD ENCODER
# ============================================
print("\nLoading encoder...")
encoder = tf.keras.models.load_model(ENCODER_PATH)
print("  ✓ Encoder loaded")

# ============================================
# GENERATE REFERENCE DATA (small sample for visualization)
# ============================================
print("\n" + "=" * 70)
print("GENERATING REFERENCE DATA")
print("=" * 70)

# Load plate
print("\nLoading SRT plate...")
plate_data = np.load(PLATE_PATH)
plate = plate_data['backgrounds']
print(f"  Plate shape: {plate.shape}")

# Global for workers
_WORKER_PLATE = plate

def _generate_true(args):
    seed, = args
    params = CadenceParams(fchans=4096, tchans=16, snr_base=20, snr_range=30)
    gen = CadenceGenerator(params, plate=_WORKER_PLATE, seed=seed)
    return gen.create_true_sample_fast()

def _generate_false(args):
    seed, = args
    params = CadenceParams(fchans=4096, tchans=16, snr_base=20, snr_range=30)
    gen = CadenceGenerator(params, plate=_WORKER_PLATE, seed=seed)
    return gen.create_false_sample()

# Generate samples
N_SAMPLES = 500
print(f"\nGenerating {N_SAMPLES} True/False samples each...")

true_samples = [_generate_true((i,)) for i in range(N_SAMPLES)]
false_samples = [_generate_false((i + 10000,)) for i in range(N_SAMPLES)]

true_samples = np.array(true_samples)  # (N, 6, 16, 4096)
false_samples = np.array(false_samples)

# Preprocess
print("Preprocessing reference data...")
true_ds = downscale(true_samples, factor=8)
false_ds = downscale(false_samples, factor=8)

true_proc = preprocess(true_ds, add_channel=True)  # (N, 6, 16, 512, 1)
false_proc = preprocess(false_ds, add_channel=True)

# Flatten cadences for encoder
true_flat = true_proc.reshape(-1, 16, 512, 1)  # (N*6, 16, 512, 1)
false_flat = false_proc.reshape(-1, 16, 512, 1)

# Extract latents
print("Extracting latents for reference data...")
true_latents = encoder.predict(true_flat, batch_size=256, verbose=0)[2]
false_latents = encoder.predict(false_flat, batch_size=256, verbose=0)[2]

print(f"  True latents shape: {true_latents.shape}")
print(f"  False latents shape: {false_latents.shape}")

# ============================================
# LOAD AND PROCESS TOP 8 CANDIDATES
# ============================================
print("\n" + "=" * 70)
print("PROCESSING TOP 8 CANDIDATES")
print("=" * 70)

print(f"\nLoading from {DATA_PATH}...")
with open(DATA_PATH, 'rb') as f:
    candidates_data = pickle.load(f)

candidate_latents = {}

for name in sorted(candidates_data.keys()):
    print(f"\n  Processing {name}...")
    data = candidates_data[name]['data']
    
    # Preprocess: (6, 16, 1, 4096) -> (6, 16, 512, 1)
    data_squeezed = data.squeeze(axis=2)  # (6, 16, 4096)
    data_batch = data_squeezed[np.newaxis, ...]  # (1, 6, 16, 4096)
    data_ds = downscale(data_batch, factor=8)  # (1, 6, 16, 512)
    data_proc = preprocess(data_ds, add_channel=True)[0]  # (6, 16, 512, 1)
    
    # Get latents
    z = encoder.predict(data_proc, verbose=0)[2]  # (6, 8)
    candidate_latents[name] = z
    
    print(f"    Latent mean: {z.mean():.3f}, std: {z.std():.3f}")

# ============================================
# VISUALIZE LATENT SPACE
# ============================================
print("\n" + "=" * 70)
print("CREATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
dim_pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3), (0, 4)]

colors = plt.cm.tab10(np.linspace(0, 1, 8))

for ax, (d1, d2) in zip(axes.flat, dim_pairs):
    # Plot reference distributions
    ax.scatter(false_latents[:, d1], false_latents[:, d2], 
               alpha=0.3, s=10, c='blue', label='False (RFI)', zorder=1)
    ax.scatter(true_latents[:, d1], true_latents[:, d2], 
               alpha=0.3, s=10, c='red', label='True (ETI)', zorder=2)
    
    # Plot candidates
    for i, (name, z) in enumerate(candidate_latents.items()):
        # Each candidate has 6 observations
        ax.scatter(z[:, d1], z[:, d2], 
                   c=[colors[i]], s=100, marker='X', edgecolors='black',
                   label=name, zorder=3)
    
    ax.set_xlabel(f'Latent {d1}', fontsize=12)
    ax.set_ylabel(f'Latent {d2}', fontsize=12)
    ax.grid(True, alpha=0.3)

# Add legend to first subplot
axes[0, 0].legend(loc='upper left', fontsize=8)

plt.suptitle('Latent Space: Training Data vs Top 8 Candidates', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved to {OUTPUT_PATH}")

# ============================================
# STATISTICS
# ============================================
print("\n" + "=" * 70)
print("LATENT STATISTICS")
print("=" * 70)

print("\n┌──────────┬───────────┬───────────┬───────────────────────────────────────┐")
print("│ Candidate│ Mean      │ Std       │ In True Region?                       │")
print("├──────────┼───────────┼───────────┼───────────────────────────────────────┤")

# Compute True region statistics
true_mean = true_latents.mean(axis=0)
true_std = true_latents.std(axis=0)

for name, z in candidate_latents.items():
    z_mean = z.mean()
    z_std = z.std()
    
    # Check how many observations fall within 2 std of True mean
    in_true = 0
    for obs in z:
        dist = np.abs(obs - true_mean) / (true_std + 1e-6)
        if dist.mean() < 2.0:
            in_true += 1
    
    region = f"{in_true}/6 observations in True region"
    print(f"│ {name:8} │ {z_mean:9.4f} │ {z_std:9.4f} │ {region:37} │")

print("└──────────┴───────────┴───────────┴───────────────────────────────────────┘")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
