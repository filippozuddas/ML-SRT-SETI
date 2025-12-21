#!/usr/bin/env python3
"""
Cadence-Level Latent Space Visualization.

This script visualizes the FULL 48-dimensional feature vector (6 obs × 8 latent dims)
that the Random Forest uses for classification, showing proper class separation.

This is different from per-observation visualization and shows how the model
actually classifies cadences.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cadence_generator import CadenceGenerator, CadenceParams
from src.utils.preprocessing import preprocess, downscale

print("=" * 70)
print("CADENCE-LEVEL LATENT SPACE VISUALIZATION")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
ENCODER_PATH = Path(__file__).parent.parent / "results" / "large_scale_training" / "encoder_final.keras"
RF_PATH = Path(__file__).parent.parent / "results" / "large_scale_training" / "random_forest.joblib"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "latent_space_viz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 300
SNR_BASE = 20
SNR_RANGE = 30

print(f"\nConfiguration:")
print(f"  Samples per class: {N_SAMPLES}")
print(f"  Output: {OUTPUT_DIR}")

# ============================================
# LOAD MODELS
# ============================================
print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

encoder = tf.keras.models.load_model(ENCODER_PATH)
rf = joblib.load(RF_PATH)
print("  Models loaded successfully")

# ============================================
# GENERATE CADENCES AND EXTRACT FEATURES
# ============================================
print("\n" + "=" * 70)
print("GENERATING CADENCES")
print("=" * 70)

params = CadenceParams(fchans=4096, tchans=16, snr_base=SNR_BASE, snr_range=SNR_RANGE)
cadence_gen = CadenceGenerator(params)

def get_cadence_features(encoder, cadence):
    """Extract 48D feature vector from a full cadence (what RF sees)."""
    # Preprocess cadence
    cadence_batch = np.array([cadence])  # (1, 6, 16, 4096)
    cadence_ds = downscale(cadence_batch, factor=8)  # (1, 6, 16, 512)
    processed = preprocess(cadence_ds, add_channel=True)[0]  # (6, 16, 512, 1)
    
    # Get latent vectors for all 6 observations
    outputs = encoder.predict(processed, verbose=0)
    if isinstance(outputs, list):
        z = outputs[2]  # (6, 8)
    else:
        z = outputs
    
    # Flatten to 48D feature vector (what RF uses!)
    return z.flatten()  # (48,)

# Generate TRUE cadences (ETI pattern: signal in ON only)
print(f"\nGenerating {N_SAMPLES} TRUE cadences...")
true_cadence_features = []
for i in tqdm(range(N_SAMPLES), desc="TRUE"):
    cadence = cadence_gen.create_true_sample_fast()
    features_48d = get_cadence_features(encoder, cadence)
    true_cadence_features.append(features_48d)
true_cadence_features = np.array(true_cadence_features)

# Generate FALSE cadences (RFI: signal in ALL observations)
print(f"Generating {N_SAMPLES} FALSE cadences...")
false_cadence_features = []
for i in tqdm(range(N_SAMPLES), desc="FALSE"):
    cadence = cadence_gen.create_false_sample()
    features_48d = get_cadence_features(encoder, cadence)
    false_cadence_features.append(features_48d)
false_cadence_features = np.array(false_cadence_features)

# Generate NOISE cadences (pure background)
print(f"Generating {N_SAMPLES} NOISE cadences...")
noise_cadence_features = []
for i in tqdm(range(N_SAMPLES), desc="NOISE"):
    cadence = cadence_gen._get_background()
    features_48d = get_cadence_features(encoder, cadence)
    noise_cadence_features.append(features_48d)
noise_cadence_features = np.array(noise_cadence_features)

print(f"\nCadence feature shapes (48D each):")
print(f"  TRUE:  {true_cadence_features.shape}")
print(f"  FALSE: {false_cadence_features.shape}")
print(f"  NOISE: {noise_cadence_features.shape}")

# ============================================
# t-SNE ON 48D CADENCE FEATURES
# ============================================
print("\n" + "=" * 70)
print("t-SNE ON CADENCE FEATURES (48D → 2D)")
print("=" * 70)

all_cadence_features = np.vstack([true_cadence_features, false_cadence_features, noise_cadence_features])

print("  Running t-SNE on 48D cadence features...")
tsne_48d = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
cadence_2d = tsne_48d.fit_transform(all_cadence_features)

# Plot
fig, ax = plt.subplots(figsize=(12, 10))

colors = ['#e74c3c', '#3498db', '#2ecc71']
labels = ['TRUE (ETI)', 'FALSE (RFI)', 'NOISE']

for i, (label, color) in enumerate(zip(labels, colors)):
    start = i * N_SAMPLES
    end = start + N_SAMPLES
    ax.scatter(cadence_2d[start:end, 0], cadence_2d[start:end, 1], 
               c=color, label=label, alpha=0.6, s=60, edgecolors='white', linewidth=0.5)

ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.set_title('Cadence-Level Latent Space (48D → 2D)\nWhat the Random Forest Sees', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cadence_tsne_48d.png", dpi=200, bbox_inches='tight')
print(f"  Saved to {OUTPUT_DIR / 'cadence_tsne_48d.png'}")
plt.close()

# ============================================
# PCA ON 48D FEATURES
# ============================================
print("\n" + "=" * 70)
print("PCA ON CADENCE FEATURES")
print("=" * 70)

pca = PCA(n_components=2)
cadence_pca = pca.fit_transform(all_cadence_features)

print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

fig, ax = plt.subplots(figsize=(12, 10))

for i, (label, color) in enumerate(zip(labels, colors)):
    start = i * N_SAMPLES
    end = start + N_SAMPLES
    ax.scatter(cadence_pca[start:end, 0], cadence_pca[start:end, 1], 
               c=color, label=label, alpha=0.6, s=60, edgecolors='white', linewidth=0.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('Cadence-Level PCA (48D → 2D)\nWhat the Random Forest Sees', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cadence_pca_48d.png", dpi=200, bbox_inches='tight')
print(f"  Saved to {OUTPUT_DIR / 'cadence_pca_48d.png'}")
plt.close()

# ============================================
# ON vs OFF FEATURE COMPARISON
# ============================================
print("\n" + "=" * 70)
print("ON vs OFF OBSERVATION COMPARISON")
print("=" * 70)

# For TRUE samples: compare ON (obs 0,2,4) vs OFF (obs 1,3,5)
on_indices = [0, 2, 4]
off_indices = [1, 3, 5]

# Reshape to (N, 6, 8)
true_reshaped = true_cadence_features.reshape(-1, 6, 8)
false_reshaped = false_cadence_features.reshape(-1, 6, 8)
noise_reshaped = noise_cadence_features.reshape(-1, 6, 8)

# Calculate mean latent vector for ON and OFF observations
true_on_mean = true_reshaped[:, on_indices, :].mean(axis=1)  # (N, 8)
true_off_mean = true_reshaped[:, off_indices, :].mean(axis=1)  # (N, 8)
true_on_off_diff = np.linalg.norm(true_on_mean - true_off_mean, axis=1)  # (N,)

false_on_mean = false_reshaped[:, on_indices, :].mean(axis=1)
false_off_mean = false_reshaped[:, off_indices, :].mean(axis=1)
false_on_off_diff = np.linalg.norm(false_on_mean - false_off_mean, axis=1)

noise_on_mean = noise_reshaped[:, on_indices, :].mean(axis=1)
noise_off_mean = noise_reshaped[:, off_indices, :].mean(axis=1)
noise_on_off_diff = np.linalg.norm(noise_on_mean - noise_off_mean, axis=1)

# Plot histogram of ON-OFF differences
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(true_on_off_diff, bins=30, alpha=0.7, label=f'TRUE (ETI) - mean: {true_on_off_diff.mean():.2f}', color='#e74c3c', density=True)
ax.hist(false_on_off_diff, bins=30, alpha=0.7, label=f'FALSE (RFI) - mean: {false_on_off_diff.mean():.2f}', color='#3498db', density=True)
ax.hist(noise_on_off_diff, bins=30, alpha=0.7, label=f'NOISE - mean: {noise_on_off_diff.mean():.2f}', color='#2ecc71', density=True)

ax.set_xlabel('||ON_mean - OFF_mean|| (Latent Distance)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('ON vs OFF Observation Difference in Latent Space\n(Key discriminating feature)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "on_off_difference.png", dpi=200, bbox_inches='tight')
print(f"  Saved to {OUTPUT_DIR / 'on_off_difference.png'}")
plt.close()

# ============================================
# RF DECISION BOUNDARY VISUALIZATION
# ============================================
print("\n" + "=" * 70)
print("RF PREDICTION PROBABILITIES")
print("=" * 70)

# Get RF predictions for all samples
all_proba = rf.predict_proba(all_cadence_features)
eti_proba = all_proba[:, 1]  # Probability of ETI class

fig, ax = plt.subplots(figsize=(12, 6))

for i, (label, color) in enumerate(zip(labels, colors)):
    start = i * N_SAMPLES
    end = start + N_SAMPLES
    ax.hist(eti_proba[start:end], bins=30, alpha=0.7, label=label, color=color, density=True)

ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax.set_xlabel('ETI Probability (RF Output)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Random Forest ETI Probability Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rf_probability_distribution.png", dpi=200, bbox_inches='tight')
print(f"  Saved to {OUTPUT_DIR / 'rf_probability_distribution.png'}")
plt.close()

# ============================================
# SUMMARY STATISTICS
# ============================================
print("\n" + "=" * 70)
print("CLASSIFICATION STATISTICS")
print("=" * 70)

# Calculate accuracy
true_correct = np.sum(eti_proba[:N_SAMPLES] > 0.5)
false_correct = np.sum(eti_proba[N_SAMPLES:2*N_SAMPLES] < 0.5)
noise_correct = np.sum(eti_proba[2*N_SAMPLES:] < 0.5)

print(f"\n  TRUE (ETI) correctly classified: {true_correct}/{N_SAMPLES} ({100*true_correct/N_SAMPLES:.1f}%)")
print(f"  FALSE (RFI) correctly classified: {false_correct}/{N_SAMPLES} ({100*false_correct/N_SAMPLES:.1f}%)")
print(f"  NOISE correctly classified: {noise_correct}/{N_SAMPLES} ({100*noise_correct/N_SAMPLES:.1f}%)")

total_correct = true_correct + false_correct + noise_correct
total = 3 * N_SAMPLES
print(f"\n  Overall accuracy: {total_correct}/{total} ({100*total_correct/total:.1f}%)")

print(f"\n  ON-OFF difference statistics:")
print(f"    TRUE:  mean={true_on_off_diff.mean():.3f}, std={true_on_off_diff.std():.3f}")
print(f"    FALSE: mean={false_on_off_diff.mean():.3f}, std={false_on_off_diff.std():.3f}")
print(f"    NOISE: mean={noise_on_off_diff.mean():.3f}, std={noise_on_off_diff.std():.3f}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(f"""
  Generated visualizations:
    1. {OUTPUT_DIR / 'cadence_tsne_48d.png'} - t-SNE of full 48D features
    2. {OUTPUT_DIR / 'cadence_pca_48d.png'} - PCA of full 48D features  
    3. {OUTPUT_DIR / 'on_off_difference.png'} - ON vs OFF latent difference
    4. {OUTPUT_DIR / 'rf_probability_distribution.png'} - RF output probabilities
    
  Key insight:
    The RF classifier sees the FULL 48D feature vector, not 2D projections.
    This allows it to capture the ON/OFF pattern that distinguishes ETI from RFI.
""")

print("=" * 70)
print("DONE")
print("=" * 70)
