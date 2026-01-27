#!/usr/bin/env python3
"""
Debug script to investigate if the problem is in VAE encoder or RF classifier.

This script:
1. Extracts latent features from top 8 GBT candidates
2. Generates synthetic ETI signals with SRT-like characteristics  
3. Compares latent distributions
4. Tests if retraining RF on GBT-like data helps
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pickle
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import preprocess, downscale

print("=" * 70)
print("INVESTIGATING VAE ENCODER vs RF CLASSIFIER")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = Path(__file__).parent.parent / "assets" / "top_8_data.pkl"
ENCODER_PATH = Path(__file__).parent.parent / "results" / "models" / "encoder_final.keras"
RF_PATH = Path(__file__).parent.parent / "results" / "models" / "random_forest.joblib"

# Load models
print("\nLoading models...")
encoder = tf.keras.models.load_model(ENCODER_PATH)
rf = joblib.load(RF_PATH)
print("  ✓ Models loaded")

# Load top 8 data
print("\nLoading top 8 candidates...")
with open(DATA_PATH, 'rb') as f:
    candidates_data = pickle.load(f)
print(f"  ✓ Loaded {len(candidates_data)} candidates")

# ============================================
# STEP 1: Extract latent features from GBT candidates
# ============================================
print("\n" + "=" * 70)
print("STEP 1: Extracting latent features from GBT candidates")
print("=" * 70)

def preprocess_observation(data):
    """Preprocess a single observation."""
    # (6, 16, 1, 4096) → (6, 16, 512, 1)
    data_reshaped = data.squeeze(axis=2)  # (6, 16, 4096)
    data_batch = data_reshaped[np.newaxis, ...]  # (1, 6, 16, 4096)
    data_ds = downscale(data_batch, factor=8)  # (1, 6, 16, 512)
    processed = preprocess(data_ds, add_channel=True)[0]  # (6, 16, 512, 1)
    return processed

gbt_latents = {}
gbt_features = []

for name in sorted(candidates_data.keys()):
    data = candidates_data[name]['data']
    processed = preprocess_observation(data)
    
    # Get latent representation
    z_mean, z_log_var, z = encoder.predict(processed, verbose=0)
    
    # Store individual observations and combined features
    gbt_latents[name] = {
        'z_mean': z_mean,
        'z': z,
        'combined': z.flatten()
    }
    gbt_features.append(z.flatten())

gbt_features = np.array(gbt_features)
print(f"\nGBT latent features shape: {gbt_features.shape}")
print(f"  Range: [{gbt_features.min():.3f}, {gbt_features.max():.3f}]")
print(f"  Mean: {gbt_features.mean():.3f}, Std: {gbt_features.std():.3f}")

# ============================================
# STEP 2: Generate synthetic ETI with similar characteristics
# ============================================
print("\n" + "=" * 70)
print("STEP 2: Generating synthetic ETI samples for comparison")
print("=" * 70)

from src.data.cadence_generator import CadenceGenerator

# Create generator (no plate needed for pure synthetic)
gen = CadenceGenerator()

# Generate true ETI samples
n_samples = 50
true_latents = []
false_latents = []

print(f"Generating {n_samples} synthetic samples...")
for i in range(n_samples):
    # True ETI: signal only in ON observations
    true_cadence, _ = gen.create_true_sample(snr=30)
    if true_cadence is not None:
        true_ds = downscale(true_cadence[np.newaxis, ...], factor=8)
        true_proc = preprocess(true_ds, add_channel=True)[0]
        _, _, z_true = encoder.predict(true_proc, verbose=0)
        true_latents.append(z_true.flatten())
    
    # False: signal in all observations (RFI pattern)
    false_cadence = gen.create_false_sample(snr=30)
    if false_cadence is not None:
        false_ds = downscale(false_cadence[np.newaxis, ...], factor=8)
        false_proc = preprocess(false_ds, add_channel=True)[0]
        _, _, z_false = encoder.predict(false_proc, verbose=0)
        false_latents.append(z_false.flatten())

true_latents = np.array(true_latents)
false_latents = np.array(false_latents)

print(f"\nSynthetic TRUE latents: {true_latents.shape}")
print(f"  Range: [{true_latents.min():.3f}, {true_latents.max():.3f}]")
print(f"  Mean: {true_latents.mean():.3f}, Std: {true_latents.std():.3f}")

print(f"\nSynthetic FALSE latents: {false_latents.shape}")
print(f"  Range: [{false_latents.min():.3f}, {false_latents.max():.3f}]")
print(f"  Mean: {false_latents.mean():.3f}, Std: {false_latents.std():.3f}")

# ============================================
# STEP 3: Compare distributions
# ============================================
print("\n" + "=" * 70)
print("STEP 3: Comparing latent distributions")
print("=" * 70)

# Calculate distances
from scipy.spatial.distance import cdist

# Distance from GBT to synthetic true
dist_to_true = cdist(gbt_features, true_latents, 'euclidean').mean(axis=1)
dist_to_false = cdist(gbt_features, false_latents, 'euclidean').mean(axis=1)

print("\nDistance to synthetic distributions:")
print("┌──────────┬────────────────┬────────────────┬─────────────┐")
print("│ Candidate│ Dist to TRUE   │ Dist to FALSE  │ Closer to?  │")
print("├──────────┼────────────────┼────────────────┼─────────────┤")
for i, name in enumerate(sorted(gbt_latents.keys())):
    closer = "TRUE" if dist_to_true[i] < dist_to_false[i] else "FALSE"
    print(f"│ {name:8} │ {dist_to_true[i]:14.3f} │ {dist_to_false[i]:14.3f} │ {closer:11} │")
print("└──────────┴────────────────┴────────────────┴─────────────┘")

# ============================================
# STEP 4: Test RF with GBT features directly
# ============================================
print("\n" + "=" * 70)
print("STEP 4: Testing RF on synthetic data")
print("=" * 70)

# Test RF on synthetic data
true_preds = rf.predict_proba(true_latents)[:, 1]
false_preds = rf.predict_proba(false_latents)[:, 1]

print(f"\nRF predictions on synthetic TRUE (should be ~1.0):")
print(f"  Mean: {true_preds.mean():.3f}, Std: {true_preds.std():.3f}")
print(f"  Range: [{true_preds.min():.3f}, {true_preds.max():.3f}]")

print(f"\nRF predictions on synthetic FALSE (should be ~0.0):")
print(f"  Mean: {false_preds.mean():.3f}, Std: {false_preds.std():.3f}")
print(f"  Range: [{false_preds.min():.3f}, {false_preds.max():.3f}]")

# ============================================
# STEP 5: Visualize latent space (PCA)
# ============================================
print("\n" + "=" * 70)
print("STEP 5: Visualizing latent space with PCA")
print("=" * 70)

from sklearn.decomposition import PCA

# Combine all features
all_features = np.vstack([gbt_features, true_latents, false_latents])
labels = ['GBT'] * len(gbt_features) + ['Synth TRUE'] * len(true_latents) + ['Synth FALSE'] * len(false_latents)

# PCA to 2D
pca = PCA(n_components=2)
features_2d = pca.fit_transform(all_features)

plt.figure(figsize=(12, 8))

# Plot synthetic first (smaller markers, background)
synth_true_2d = features_2d[len(gbt_features):len(gbt_features)+len(true_latents)]
synth_false_2d = features_2d[len(gbt_features)+len(true_latents):]
gbt_2d = features_2d[:len(gbt_features)]

plt.scatter(synth_false_2d[:, 0], synth_false_2d[:, 1], 
            c='blue', alpha=0.3, s=30, label='Synthetic FALSE (RFI)')
plt.scatter(synth_true_2d[:, 0], synth_true_2d[:, 1], 
            c='green', alpha=0.3, s=30, label='Synthetic TRUE (ETI)')

# Plot GBT candidates (larger markers, with names)
plt.scatter(gbt_2d[:, 0], gbt_2d[:, 1], 
            c='red', s=150, marker='*', edgecolors='black', linewidths=1,
            label='GBT Top 8 Candidates')

# Add names
for i, name in enumerate(sorted(gbt_latents.keys())):
    plt.annotate(name, (gbt_2d[i, 0], gbt_2d[i, 1]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Latent Space: GBT Candidates vs Synthetic Data')
plt.legend()
plt.grid(alpha=0.3)

output_path = Path(__file__).parent.parent / "results" / "visualizations" / "latent_space" /"latent_space_comparison_synt_model.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_path}")

plt.close()

# ============================================
# CONCLUSIONS
# ============================================
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

# Check if GBT candidates are closer to TRUE or FALSE distribution
n_closer_to_true = sum(1 for d_t, d_f in zip(dist_to_true, dist_to_false) if d_t < d_f)
n_closer_to_false = len(dist_to_true) - n_closer_to_true

print(f"\n📊 Distance Analysis:")
print(f"   GBT candidates closer to synthetic TRUE: {n_closer_to_true}/8")
print(f"   GBT candidates closer to synthetic FALSE: {n_closer_to_false}/8")

if n_closer_to_true >= 6:
    print("\n✅ VAE ENCODER is working correctly!")
    print("   GBT candidates are in the TRUE (ETI) region of latent space.")
    print("   → Problem is likely in RANDOM FOREST classifier.")
    print("   → Solution: Fine-tune RF on GBT-like data.")
else:
    print("\n⚠️ VAE ENCODER may have domain shift issues.")
    print("   GBT candidates are not clearly in TRUE region.")
    print("   → Problem is in VAE encoder generalization.")
    print("   → Solution: Retrain VAE with diverse data.")

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)
