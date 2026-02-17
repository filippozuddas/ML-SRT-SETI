#!/usr/bin/env python3
"""
Regenerate the Latent Space Visualization without re-training.

Loads a trained encoder + plate, generates evaluation samples,
extracts latent vectors, and produces a scatter plot with clear
True vs False distinction.

Usage:
    python plot_latent_space_from_model.py \
        --encoder results/band_c/encoder_final.keras \
        --plate data/band_c/srt_backgrounds.npz \
        --output results/band_c/

    python plot_latent_space_from_model.py \
        --encoder results/band_k/encoder_final.keras \
        --plate data/band_k/srt_backgrounds.npz \
        --output results/band_k/
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import preprocess, downscale, combine_cadences, recombine_latents
from src.data.cadence_generator import CadenceGenerator, CadenceParams

# ============================================
# ARGUMENT PARSING
# ============================================
parser = argparse.ArgumentParser(
    description="Regenerate the Latent Space plot from a trained encoder (no re-training needed)"
)
parser.add_argument(
    "--encoder", type=str, required=True,
    help="Path to the encoder model (.keras)"
)
parser.add_argument(
    "--plate", type=str, required=True,
    help="Path to the SRT backgrounds plate (.npz)"
)
parser.add_argument(
    "--output", type=str, required=True,
    help="Output directory for the plot"
)
parser.add_argument(
    "--n-samples", type=int, default=2000,
    help="Number of cadences per class (default: 2000)"
)
parser.add_argument(
    "--filename", type=str, default="latent_space_final.png",
    help="Output filename (default: latent_space_final.png)"
)
args = parser.parse_args()

ENCODER_PATH = Path(args.encoder)
PLATE_PATH = Path(args.plate)
OUTPUT_DIR = Path(args.output)
N_SAMPLES = args.n_samples

# Validate paths
for p, name in [(ENCODER_PATH, "Encoder"), (PLATE_PATH, "Plate")]:
    if not p.exists():
        print(f"ERROR: {name} not found at: {p}")
        sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("LATENT SPACE VISUALIZATION (no re-training)")
print("=" * 70)
print(f"\n  Encoder:  {ENCODER_PATH}")
print(f"  Plate:    {PLATE_PATH}")
print(f"  Output:   {OUTPUT_DIR / args.filename}")
print(f"  Samples:  {N_SAMPLES} per class")

# ============================================
# LOAD MODEL & DATA
# ============================================
print("\nLoading encoder...")
encoder = tf.keras.models.load_model(ENCODER_PATH)
print("  ✓ Encoder loaded")

print("Loading SRT plate...")
plate = np.load(PLATE_PATH)['backgrounds']
print(f"  Plate shape: {plate.shape}")

# ============================================
# GENERATE EVALUATION DATA
# ============================================
print(f"\nGenerating {N_SAMPLES} True samples...")
true_samples = []
for i in range(N_SAMPLES):
    params = CadenceParams(fchans=4096, tchans=16, snr_base=20, snr_range=30)
    gen = CadenceGenerator(params, plate=plate, seed=i + 50000)
    true_samples.append(gen.create_true_sample_fast())
true_samples = np.array(true_samples)

print(f"Generating {N_SAMPLES} False samples...")
false_samples = []
for i in range(N_SAMPLES):
    params = CadenceParams(fchans=4096, tchans=16, snr_base=20, snr_range=30)
    gen = CadenceGenerator(params, plate=plate, seed=i + 60000)
    false_samples.append(gen.create_false_sample())
false_samples = np.array(false_samples)

print(f"Raw shapes: True={true_samples.shape}, False={false_samples.shape}")

# Preprocess
print("Preprocessing...")
true_ds = downscale(true_samples, factor=8)
false_ds = downscale(false_samples, factor=8)

true_proc = preprocess(true_ds, add_channel=True)
false_proc = preprocess(false_ds, add_channel=True)

# Flatten cadences: (N, 6, H, W, C) -> (N*6, H, W, C)
true_flat = combine_cadences(true_proc)
false_flat = combine_cadences(false_proc)

print(f"Flattened shapes: True={true_flat.shape}, False={false_flat.shape}")

# ============================================
# EXTRACT LATENTS
# ============================================
print("\nEncoding True samples...")
true_latents = encoder.predict(true_flat, batch_size=512, verbose=1)[2]

print("Encoding False samples...")
false_latents = encoder.predict(false_flat, batch_size=512, verbose=1)[2]

print(f"Latent shapes: True={true_latents.shape}, False={false_latents.shape}")

# ============================================
# PLOT LATENT SPACE
# ============================================
print("\nGenerating plot...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
dim_pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3), (0, 4)]

for ax, (d1, d2) in zip(axes.flat, dim_pairs):
    # True first (background) - small, semi-transparent
    ax.scatter(true_latents[:, d1], true_latents[:, d2],
               alpha=0.15, s=8, c='#FF6B6B', marker='o', label='True', zorder=1)
    # False on top - larger, more opaque, distinct diamond marker
    ax.scatter(false_latents[:, d1], false_latents[:, d2],
               alpha=0.7, s=30, c='#1B4F72', marker='D', label='False',
               edgecolors='white', linewidths=0.3, zorder=2)
    ax.set_xlabel(f'Latent {d1}')
    ax.set_ylabel(f'Latent {d2}')
    ax.legend(markerscale=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Latent Space Visualization (Final Model)', fontsize=14)
plt.tight_layout()

out_path = OUTPUT_DIR / args.filename
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Observation-level plot saved to: {out_path}")

# ============================================
# CADENCE-LEVEL LATENT SPACE (recombined 48D)
# ============================================
print("\n" + "=" * 70)
print("CADENCE-LEVEL LATENT VISUALIZATION (recombined)")
print("=" * 70)

# Recombine: (N*6, latent_dim) -> (N, latent_dim*6)
true_latents_cadence = recombine_latents(true_latents)
false_latents_cadence = recombine_latents(false_latents)

print(f"  Cadence latent shapes: True={true_latents_cadence.shape}, False={false_latents_cadence.shape}")

# Combine for dimensionality reduction
X_all = np.vstack([true_latents_cadence, false_latents_cadence])
labels = np.array(['True'] * len(true_latents_cadence) + ['False'] * len(false_latents_cadence))

# Handle NaN/Inf
if not np.isfinite(X_all).all():
    print("  Warning: Found NaN/Inf, replacing...")
    X_all = np.nan_to_num(X_all, nan=0, posinf=100, neginf=-100)

# --- PCA ---
print("\nComputing PCA projection...")
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all)

true_pca = X_pca[labels == 'True']
false_pca = X_pca[labels == 'False']

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(true_pca[:, 0], true_pca[:, 1],
           alpha=0.3, s=12, c='#FF6B6B', marker='o', label='True', zorder=1)
ax.scatter(false_pca[:, 0], false_pca[:, 1],
           alpha=0.7, s=30, c='#1B4F72', marker='D', label='False',
           edgecolors='white', linewidths=0.3, zorder=2)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('Cadence-Level Latent Space — PCA Projection', fontsize=14)
ax.legend(markerscale=2, fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

pca_path = OUTPUT_DIR / 'latent_space_cadence_pca.png'
plt.savefig(pca_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ PCA plot saved to: {pca_path}")

# --- t-SNE ---
print("\nComputing t-SNE projection (this may take a minute)...")
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_all)

true_tsne = X_tsne[labels == 'True']
false_tsne = X_tsne[labels == 'False']

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(true_tsne[:, 0], true_tsne[:, 1],
           alpha=0.3, s=12, c='#FF6B6B', marker='o', label='True', zorder=1)
ax.scatter(false_tsne[:, 0], false_tsne[:, 1],
           alpha=0.7, s=30, c='#1B4F72', marker='D', label='False',
           edgecolors='white', linewidths=0.3, zorder=2)
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title('Cadence-Level Latent Space — t-SNE Projection', fontsize=14)
ax.legend(markerscale=2, fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

tsne_path = OUTPUT_DIR / 'latent_space_cadence_tsne.png'
plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ t-SNE plot saved to: {tsne_path}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("ALL PLOTS GENERATED")
print("=" * 70)
print(f"""
  1. {out_path}
     → Observation-level latent pairs (individual frames)

  2. {pca_path}
     → Cadence-level PCA (recombined 48D → 2D)

  3. {tsne_path}
     → Cadence-level t-SNE (recombined 48D → 2D)

  Plots 2 & 3 show the actual separation used by the Random Forest.
  True/False cadences should be clearly separable there.
""")

