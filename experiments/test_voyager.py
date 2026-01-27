#!/usr/bin/env python3
"""
Test the model on real Voyager 1 observation data from GBT.
This is a complete ON/OFF cadence observing Voyager 1 in July 2020.

The data is a single coarse channel observation.
Pattern: ON/OFF/ON/OFF/ON/OFF (6 five-minute scans)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from blimpy import Waterfall
import matplotlib.pyplot as plt

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import normalize_log

print("=" * 70)
print("TESTING ON VOYAGER 1 OBSERVATION DATA")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
# Voyager data files (ON/OFF/ON/OFF/ON/OFF pattern)
DATA_DIR = Path(__file__).parent.parent / "data" / "voyager"

VOYAGER_FILES = [
    DATA_DIR / "single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5",   # ON
    DATA_DIR / "single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_OFF_0012.rawspec.0000.h5", # OFF
    DATA_DIR / "single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5",   # ON
    DATA_DIR / "single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_OFF_0014.rawspec.0000.h5", # OFF
    DATA_DIR / "single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5",   # ON
    DATA_DIR / "single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_OFF_0016.rawspec.0000.h5", # OFF
]

ENCODER_PATH = Path(__file__).parent.parent / "results" / "real_obs_training" / "C_band" / "encoder_final.keras"
RF_PATH = Path(__file__).parent.parent / "results" / "real_obs_training" / "C_band" / "random_forest.joblib"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "tests" / "voyager"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.5

print(f"\nConfiguration:")
print(f"  Data Directory: {DATA_DIR}")
print(f"  Encoder: {ENCODER_PATH}")
print(f"  RF: {RF_PATH}")
print(f"  Threshold: {THRESHOLD}")

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
# LOAD VOYAGER DATA
# ============================================
print("\n" + "=" * 70)
print("LOADING VOYAGER OBSERVATION DATA")
print("=" * 70)

# First, explore the data structure
print("\nExploring first file...")
first_file = VOYAGER_FILES[0]
if first_file.exists():
    wf = Waterfall(str(first_file))
    print(f"  File: {first_file.name}")
    print(f"  Header info:")
    wf.info()

    wf.plot_spectrum(logged=True)
    plt.savefig(OUTPUT_DIR / "spectrum.png", dpi=300, bbox_inches="tight")
    plt.close()

    wf.plot_waterfall(f_start=8419.542, f_stop=8419.543)
    plt.savefig(OUTPUT_DIR / "waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    print(f"  ❌ File not found: {first_file}")
    print("  Please check the path and try again.")
    sys.exit(1)

# Now load all 6 observations
print("\nLoading all 6 observations...")
cadence_data = []

for i, filepath in enumerate(VOYAGER_FILES):
    obs_type = "ON" if "OFF" not in filepath.name else "OFF"
    print(f"  Loading {i+1}/6 ({obs_type}): {filepath.name}")
    
    wf = Waterfall(str(filepath))
    data = wf.data  # Shape: (time, pol?, freq)
    
    print(f"    Raw shape: {data.shape}")
    print(f"    Freq range: {wf.header['fch1']:.6f} - {wf.header['fch1'] + wf.header['nchans'] * wf.header['foff']:.6f} MHz")
    
    # Squeeze polarization if present
    if len(data.shape) == 3:
        data = data.squeeze()  # Remove polarization axis
    
    print(f"    Squeezed shape: {data.shape}")
    cadence_data.append(data)

# ============================================
# FIND VOYAGER SIGNAL LOCATION
# ============================================
print("\n" + "=" * 70)
print("DETECTING VOYAGER SIGNAL LOCATION")
print("=" * 70)

# Find the Voyager signal by comparing ON vs OFF spectra
on_spectrum = (cadence_data[0].mean(axis=0) + cadence_data[2].mean(axis=0) + cadence_data[4].mean(axis=0)) / 3
off_spectrum = (cadence_data[1].mean(axis=0) + cadence_data[3].mean(axis=0) + cadence_data[5].mean(axis=0)) / 3
diff_spectrum = on_spectrum - off_spectrum

# Find peak (location of Voyager signal)
peak_idx = np.argmax(diff_spectrum)
print(f"  Voyager signal detected at frequency bin: {peak_idx}")

# Get frequency info from header
wf_info = Waterfall(str(VOYAGER_FILES[0]), load_data=False)
fch1 = wf_info.header['fch1']  # MHz
foff = wf_info.header['foff']  # MHz per channel
peak_freq = fch1 + peak_idx * foff
print(f"  Estimated signal frequency: {peak_freq:.6f} MHz")

# ============================================
# PREPROCESS DATA
# ============================================
print("\n" + "=" * 70)
print("PREPROCESSING DATA")
print("=" * 70)

# Check shapes
shapes = [d.shape for d in cadence_data]
print(f"  Observation shapes: {shapes}")

# =====================================================
# CORRECT PIPELINE: Extract 4096-channel snippet, then downsample 8x to 512
# This matches the training pipeline!
# =====================================================

SNIPPET_WIDTH = 4096  # Same as training
DOWNSAMPLE_FACTOR = 8  # Same as training
FINAL_FREQ_BINS = SNIPPET_WIDTH // DOWNSAMPLE_FACTOR  # 512

print(f"\n  Pipeline: Extract {SNIPPET_WIDTH} channels around signal → downsample {DOWNSAMPLE_FACTOR}x → {FINAL_FREQ_BINS} channels")

# Use peak_idx found earlier to center the snippet
# peak_idx is the location of the Voyager signal in the raw data
snippet_half = SNIPPET_WIDTH // 2
snippet_start = max(0, peak_idx - snippet_half)
snippet_end = min(cadence_data[0].shape[1], peak_idx + snippet_half)

# Adjust if we hit boundaries
if snippet_end - snippet_start < SNIPPET_WIDTH:
    if snippet_start == 0:
        snippet_end = SNIPPET_WIDTH
    else:
        snippet_start = snippet_end - SNIPPET_WIDTH

print(f"  Extracting snippet: bins {snippet_start} to {snippet_end} (centered on signal at {peak_idx})")

# Store snippets before and after downsampling for visualization
snippets_before_ds = []  # 16x4096 (before downsampling)
processed_cadence = []   # 16x512 (after downsampling)

for i, data in enumerate(cadence_data):
    # data shape: (time, freq) after squeeze
    time_bins = data.shape[0]
    
    # Step 1: Extract the 4096-channel snippet around the signal
    snippet = data[:, snippet_start:snippet_end]
    print(f"  Obs {i+1}: Extracted snippet shape: {snippet.shape}")
    
    # Step 2: Downsample time if needed (should already be 16 for Voyager)
    if time_bins > 16:
        time_factor = time_bins // 16
        new_time = time_bins // time_factor
        snippet = snippet[:new_time * time_factor, :].reshape(new_time, time_factor, snippet.shape[1]).mean(axis=1)
    
    # Store snippet BEFORE frequency downsampling (for visualization)
    snippets_before_ds.append(snippet.copy())
    
    # Step 3: Downsample frequency by factor 8 (4096 → 512)
    snippet = snippet.reshape(snippet.shape[0], FINAL_FREQ_BINS, DOWNSAMPLE_FACTOR).mean(axis=2)
    
    print(f"           After downsampling: {snippet.shape}")
    
    # Apply log normalization (same as training!)
    snippet = normalize_log(snippet.astype(np.float64))
    
    # Add channel dimension
    snippet = snippet[..., np.newaxis]
    
    processed_cadence.append(snippet)

# Stack into cadence
snippets_before_ds = np.array(snippets_before_ds)  # (6, 16, 4096)
processed_cadence = np.array(processed_cadence, dtype=np.float32)  # (6, 16, 512, 1)
print(f"\n  Final processed shape: {processed_cadence.shape}")

# ============================================
# PLOT BEFORE DOWNSAMPLING (16x4096 snippet)
# ============================================
print("\n" + "=" * 70)
print("CREATING PRE-DOWNSAMPLING PLOT (16x4096 snippet)")
print("=" * 70)

fig_pre, axes_pre = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
fig_pre.subplots_adjust(hspace=0)

for i in range(6):
    ax = axes_pre[i]
    obs_type = "ON" if i % 2 == 0 else "OFF"
    
    # Plot snippet before downsampling (apply log for visualization)
    data_plot = np.log10(np.abs(snippets_before_ds[i]) + 1e-10)
    
    img = ax.imshow(data_plot, aspect='auto', cmap='hot', origin='upper')
    
    ax.text(0.02, 0.85, f'Obs {i+1} ({obs_type})', transform=ax.transAxes, 
            fontsize=10, fontweight='bold', color='white',
            bbox=dict(boxstyle='square', facecolor='black', alpha=0.5))
    
    ax.set_ylabel('Time bin')
    if i < 5:
        ax.tick_params(labelbottom=False)

axes_pre[5].set_xlabel('Frequency bins (4096 total - BEFORE downsampling)')

cbar = fig_pre.colorbar(img, ax=axes_pre, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('log10(Power)')

fig_pre.suptitle('Voyager 1 BEFORE Downsampling (16x4096) - 4096-channel Snippet', 
                 fontsize=14, fontweight='bold')

plt.savefig(OUTPUT_DIR / "voyager_before_downsampling.png", dpi=200, bbox_inches='tight')
print(f"  Saved pre-downsampling plot to {OUTPUT_DIR / 'voyager_before_downsampling.png'}")
plt.close(fig_pre)

# ============================================
# PLOT AFTER DOWNSAMPLING (16x512 - what model sees)
# ============================================
print("\n" + "=" * 70)
print("CREATING POST-DOWNSAMPLING PLOT (16x512)")
print("=" * 70)

fig_post, axes_post = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
fig_post.subplots_adjust(hspace=0)

for i in range(6):
    ax = axes_post[i]
    obs_type = "ON" if i % 2 == 0 else "OFF"
    
    # Plot processed data (already log-normalized to [0,1])
    data_plot = processed_cadence[i, :, :, 0]
    
    img = ax.imshow(data_plot, aspect='auto', cmap='hot', origin='upper',
                    extent=[0, 512, data_plot.shape[0], 0],
                    vmin=0, vmax=1)
    
    ax.text(0.02, 0.85, f'Obs {i+1} ({obs_type})', transform=ax.transAxes, 
            fontsize=10, fontweight='bold', color='white',
            bbox=dict(boxstyle='square', facecolor='black', alpha=0.5))
    
    ax.set_ylabel('Time bin')
    if i < 5:
        ax.tick_params(labelbottom=False)

axes_post[5].set_xlabel('Frequency bins (512 total - AFTER 8x downsampling)')

cbar = fig_post.colorbar(img, ax=axes_post, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('Log-Normalized Power [0-1]')

fig_post.suptitle('Voyager 1 AFTER Downsampling (16x512) - What the Model Sees', 
                  fontsize=14, fontweight='bold')

plt.savefig(OUTPUT_DIR / "voyager_after_downsampling.png", dpi=200, bbox_inches='tight')
print(f"  Saved post-downsampling plot to {OUTPUT_DIR / 'voyager_after_downsampling.png'}")
plt.close(fig_post)

# ============================================
# PREDICT
# ============================================
print("\n" + "=" * 70)
print("RUNNING PREDICTION")
print("=" * 70)

# Get latent representations
outputs = encoder.predict(processed_cadence, verbose=0)
if isinstance(outputs, list):
    z = outputs[2]
else:
    z = outputs

print(f"  Latent vectors shape: {z.shape}")

# Flatten for RF
features = z.flatten().reshape(1, -1)
print(f"  Features shape: {features.shape}")

# Predict
proba = rf.predict_proba(features)[0]
prediction = proba[1] > THRESHOLD

print(f"\n  Probability (ETI): {proba[1]:.4f}")
print(f"  Probability (Noise): {proba[0]:.4f}")

if prediction:
    print(f"\n  🚀 ✅ VOYAGER SIGNAL DETECTED! (prob: {proba[1]*100:.1f}%)")
else:
    print(f"\n  ❌ Signal not detected (prob: {proba[1]*100:.1f}%)")

# (Removed redundant 2x3 grid visualization - using stacked vertical plots above)

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("VOYAGER TEST SUMMARY")
print("=" * 70)

print(f"""
  Data: Voyager 1 observation from GBT (July 2020)
  Pattern: ON/OFF/ON/OFF/ON/OFF (6 x 5-minute scans)
  
  Expected: ETI signal (Voyager transmits at ~8.4 GHz)
  Result: {'✅ DETECTED' if prediction else '❌ NOT DETECTED'}
  
  Probability (ETI): {proba[1]*100:.1f}%
  Probability (Noise): {proba[0]*100:.1f}%
""")

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
