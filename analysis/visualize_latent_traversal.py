#!/usr/bin/env python3
"""
Latent Space Traversal Visualization.

This script perturbs each latent dimension and decodes the results,
revealing what physical property each dimension has learned to encode.

Based on the paper's "Latent Space Analysis" section.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cadence_generator import CadenceGenerator, CadenceParams
from src.utils.preprocessing import preprocess, downscale

print("=" * 70)
print("LATENT SPACE TRAVERSAL VISUALIZATION")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
ENCODER_PATH = Path(__file__).parent.parent / "results" / "large_scale_training" / "encoder_final.keras"
DECODER_PATH = Path(__file__).parent.parent / "results" / "large_scale_training" / "decoder_final.keras"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "latent_traversal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATENT_DIM = 8  # VAE latent dimension
TRAVERSAL_RANGE = np.linspace(-3, 3, 11)  # Values to traverse

print(f"\nConfiguration:")
print(f"  Encoder: {ENCODER_PATH}")
print(f"  Decoder: {DECODER_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Traversal range: {TRAVERSAL_RANGE[0]} to {TRAVERSAL_RANGE[-1]}")

# ============================================
# LOAD MODELS
# ============================================
print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

encoder = tf.keras.models.load_model(ENCODER_PATH)
decoder = tf.keras.models.load_model(DECODER_PATH)
print("  Encoder and Decoder loaded successfully")

# Print model info
print(f"\n  Encoder input: {encoder.input_shape}")
print(f"  Encoder output shapes: {[o.shape for o in encoder.output]}")
print(f"  Decoder input: {decoder.input_shape}")
print(f"  Decoder output: {decoder.output_shape}")

# ============================================
# GET REFERENCE LATENT VECTOR
# ============================================
print("\n" + "=" * 70)
print("GENERATING REFERENCE SAMPLE")
print("=" * 70)

params = CadenceParams(fchans=4096, tchans=16, snr_base=30, snr_range=10)
cadence_gen = CadenceGenerator(params)

# Generate a TRUE sample as reference
reference_cadence = cadence_gen.create_true_sample_fast()
ref_batch = np.array([reference_cadence])
ref_ds = downscale(ref_batch, factor=8)
ref_processed = preprocess(ref_ds, add_channel=True)[0]  # (6, 16, 512, 1)

# Get latent representation
outputs = encoder.predict(ref_processed, verbose=0)
if isinstance(outputs, list):
    z_mean = outputs[0]
    z_log_var = outputs[1]
    z = outputs[2]
else:
    z_mean = outputs
    z = outputs

# Use mean of all 6 observations as reference
z_reference = z.mean(axis=0)  # (8,)
print(f"  Reference z vector: {z_reference}")

# ============================================
# LATENT TRAVERSAL FOR EACH DIMENSION
# ============================================
print("\n" + "=" * 70)
print("PERFORMING LATENT TRAVERSAL")
print("=" * 70)

# Create a grid figure: rows = latent dimensions, cols = traversal steps
fig, axes = plt.subplots(LATENT_DIM, len(TRAVERSAL_RANGE), figsize=(22, 16))
fig.suptitle('Latent Space Traversal (Perturbing Each Dimension)', fontsize=16, fontweight='bold')

for dim in tqdm(range(LATENT_DIM), desc="Traversing dimensions"):
    for col, value in enumerate(TRAVERSAL_RANGE):
        # Create perturbed z vector
        z_perturbed = z_reference.copy()
        z_perturbed[dim] = value
        
        # Decode
        z_input = z_perturbed.reshape(1, LATENT_DIM)
        decoded = decoder.predict(z_input, verbose=0)[0]
        
        # Remove channel dimension if present
        if decoded.ndim == 3 and decoded.shape[-1] == 1:
            decoded = decoded[:, :, 0]
        
        # Plot
        ax = axes[dim, col]
        ax.imshow(decoded, aspect='auto', cmap='hot', origin='upper')
        ax.axis('off')
        
        if dim == 0:
            ax.set_title(f'{value:.1f}', fontsize=10)
        if col == 0:
            ax.set_ylabel(f'z{dim}', fontsize=12, rotation=0, ha='right', va='center')

plt.tight_layout(rect=[0.02, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / "latent_traversal_all.png", dpi=200, bbox_inches='tight')
print(f"  Saved traversal grid to {OUTPUT_DIR / 'latent_traversal_all.png'}")
plt.close()

# ============================================
# DETAILED VIEW OF FIRST 3 DIMENSIONS
# ============================================
print("\n" + "=" * 70)
print("DETAILED VISUALIZATION (z0, z1, z2)")
print("=" * 70)

for dim in range(min(3, LATENT_DIM)):
    fig, axes = plt.subplots(1, len(TRAVERSAL_RANGE), figsize=(20, 3))
    fig.suptitle(f'Latent Dimension z{dim} Traversal', fontsize=14, fontweight='bold')
    
    for col, value in enumerate(TRAVERSAL_RANGE):
        z_perturbed = z_reference.copy()
        z_perturbed[dim] = value
        
        z_input = z_perturbed.reshape(1, LATENT_DIM)
        decoded = decoder.predict(z_input, verbose=0)[0]
        
        if decoded.ndim == 3 and decoded.shape[-1] == 1:
            decoded = decoded[:, :, 0]
        
        ax = axes[col]
        ax.imshow(decoded, aspect='auto', cmap='hot', origin='upper')
        ax.set_title(f'z{dim}={value:.1f}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"latent_traversal_z{dim}.png", dpi=200, bbox_inches='tight')
    print(f"  Saved z{dim} traversal to {OUTPUT_DIR / f'latent_traversal_z{dim}.png'}")
    plt.close()

# ============================================
# LATENT DIMENSION CORRELATION WITH PROPERTIES
# ============================================
print("\n" + "=" * 70)
print("ANALYZING LATENT DIMENSIONS")
print("=" * 70)

# Generate samples with different SNR and analyze correlation
print("  Generating samples with varying SNR...")
snr_values = [10, 20, 30, 40, 50, 60]
snr_latents = {snr: [] for snr in snr_values}

for snr in snr_values:
    params_snr = CadenceParams(fchans=4096, tchans=16, snr_base=snr, snr_range=5)
    gen_snr = CadenceGenerator(params_snr)
    
    for _ in range(30):
        cadence = gen_snr.create_true_sample_fast()
        batch = np.array([cadence])
        ds = downscale(batch, factor=8)
        processed = preprocess(ds, add_channel=True)[0]
        
        outputs = encoder.predict(processed, verbose=0)
        if isinstance(outputs, list):
            z = outputs[2]
        else:
            z = outputs
        
        snr_latents[snr].append(z.mean(axis=0))

# Calculate mean latent for each SNR
snr_mean_latents = {snr: np.mean(latents, axis=0) for snr, latents in snr_latents.items()}

# Plot SNR vs each latent dimension
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for dim in range(LATENT_DIM):
    ax = axes[dim]
    dim_values = [snr_mean_latents[snr][dim] for snr in snr_values]
    ax.plot(snr_values, dim_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('SNR')
    ax.set_ylabel(f'z{dim} mean')
    ax.set_title(f'Dimension z{dim} vs SNR')
    ax.grid(True, alpha=0.3)
    
    # Calculate correlation
    corr = np.corrcoef(snr_values, dim_values)[0, 1]
    ax.text(0.05, 0.95, f'r={corr:.2f}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Latent Dimension Correlation with SNR', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "latent_snr_correlation.png", dpi=200, bbox_inches='tight')
print(f"  Saved SNR correlation to {OUTPUT_DIR / 'latent_snr_correlation.png'}")
plt.close()

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("LATENT TRAVERSAL COMPLETE")
print("=" * 70)

print(f"""
  Generated visualizations:
    1. {OUTPUT_DIR / 'latent_traversal_all.png'} - All 8 dimensions traversal grid
    2. {OUTPUT_DIR / 'latent_traversal_z0.png'} - Detailed z0 traversal
    3. {OUTPUT_DIR / 'latent_traversal_z1.png'} - Detailed z1 traversal
    4. {OUTPUT_DIR / 'latent_traversal_z2.png'} - Detailed z2 traversal
    5. {OUTPUT_DIR / 'latent_snr_correlation.png'} - SNR correlation analysis
    
  Interpretation:
    - Look for dimensions that control signal intensity (SNR)
    - Look for dimensions that control signal position
    - Look for dimensions that control noise/artifacts
    - Strong correlation with SNR suggests that dimension encodes signal strength
""")

print("=" * 70)
print("DONE")
print("=" * 70)
