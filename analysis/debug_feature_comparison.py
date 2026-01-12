#!/usr/bin/env python3
"""
Debug: Compare features from chunked vs non-chunked pipeline.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
from blimpy import Waterfall
import warnings
import tensorflow as tf

# Import preprocessing
from src.utils.preprocessing import preprocess, downscale

# Path to Voyager files
VOYAGER_DIR = Path("data")
voyager_files = sorted(VOYAGER_DIR.glob("single*.h5"))[:6]

if len(voyager_files) < 6:
    print("ERROR: Voyager files not found in data/")
    exit(1)

# Load encoder
ENCODER_PATH = "results/real_obs_training/C_band/encoder_final.keras"
print(f"Loading encoder from {ENCODER_PATH}...")
encoder = tf.keras.models.load_model(ENCODER_PATH)

print("="*60)
print("COMPARING CHUNKED VS NON-CHUNKED PROCESSING")
print("="*60)

# Get file info
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    wf = Waterfall(str(voyager_files[0]), load_data=False)
    
fch1 = wf.header['fch1']
foff = wf.header['foff']
nchans = wf.header['nchans']

print(f"\nFile info: fch1={fch1:.4f}, foff={foff}, nchans={nchans}")

# ============================================
# Method 1: NON-CHUNKED (load full file)
# ============================================
print("\n" + "="*60)
print("METHOD 1: NON-CHUNKED (full load)")
print("="*60)

# Load all 6 files fully
full_data_list = []
for f in voyager_files:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = Waterfall(str(f))
    full_data_list.append(wf.data.squeeze())

full_cadence = np.stack(full_data_list, axis=0)  # (6, 16, nchans)
print(f"Full cadence shape: {full_cadence.shape}")

# Extract first snippet (channels 0:4096)
snippet_full = full_cadence[:, :, 0:4096]  # (6, 16, 4096)
print(f"Snippet shape: {snippet_full.shape}")

# Downsample and preprocess
snippet_full_ds = downscale(snippet_full, factor=8)  # (6, 16, 512)
snippet_full_proc = preprocess(snippet_full_ds, add_channel=True)  # (6, 16, 512, 1)
print(f"Preprocessed shape: {snippet_full_proc.shape}")
print(f"Preprocessed stats: min={snippet_full_proc.min():.4f}, max={snippet_full_proc.max():.4f}, mean={snippet_full_proc.mean():.4f}")

# Encode
latents_full = []
for i in range(6):
    obs = snippet_full_proc[i:i+1]  # (1, 16, 512, 1)
    z = encoder.predict(obs, verbose=0)
    if isinstance(z, list):
        z = z[2]  # Get z from VAE output
    latents_full.append(z[0])

latents_full = np.concatenate(latents_full)  # (48,)
print(f"Latent features shape: {latents_full.shape}")
print(f"Latent features: {latents_full[:8]}...")  # Show first 8

# ============================================
# Method 2: CHUNKED (load with f_start/f_stop)
# ============================================
print("\n" + "="*60)
print("METHOD 2: CHUNKED (f_start/f_stop load)")
print("="*60)

# Calculate frequency range for first 4096 channels
f_start_chan = 0
f_end_chan = 4096

f_start = fch1 + f_start_chan * foff
f_end = fch1 + f_end_chan * foff

# Swap if needed
if f_start > f_end:
    f_start, f_end = f_end, f_start
    
print(f"Loading freq range: {f_start:.4f} - {f_end:.4f} MHz")

# Load chunk from all 6 files
chunk_data_list = []
for f in voyager_files:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf = Waterfall(str(f), f_start=f_start, f_stop=f_end)
    chunk_data_list.append(wf.data.squeeze())

chunk_cadence = np.stack(chunk_data_list, axis=0)  # (6, 16, ~4096)
print(f"Chunk cadence shape: {chunk_cadence.shape}")

# Extract snippet
snippet_chunk = chunk_cadence[:, :, 0:4096]  # (6, 16, 4096)
print(f"Snippet shape: {snippet_chunk.shape}")

# Downsample and preprocess
snippet_chunk_ds = downscale(snippet_chunk, factor=8)  # (6, 16, 512)
snippet_chunk_proc = preprocess(snippet_chunk_ds, add_channel=True)  # (6, 16, 512, 1)
print(f"Preprocessed shape: {snippet_chunk_proc.shape}")
print(f"Preprocessed stats: min={snippet_chunk_proc.min():.4f}, max={snippet_chunk_proc.max():.4f}, mean={snippet_chunk_proc.mean():.4f}")

# Encode
latents_chunk = []
for i in range(6):
    obs = snippet_chunk_proc[i:i+1]  # (1, 16, 512, 1)
    z = encoder.predict(obs, verbose=0)
    if isinstance(z, list):
        z = z[2]
    latents_chunk.append(z[0])

latents_chunk = np.concatenate(latents_chunk)  # (48,)
print(f"Latent features shape: {latents_chunk.shape}")
print(f"Latent features: {latents_chunk[:8]}...")

# ============================================
# COMPARE
# ============================================
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

# Compare raw data
raw_match = np.allclose(snippet_full, snippet_chunk, rtol=1e-5)
print(f"Raw snippet match: {raw_match}")

# Compare preprocessed
proc_match = np.allclose(snippet_full_proc, snippet_chunk_proc, rtol=1e-5)
print(f"Preprocessed match: {proc_match}")

# Compare latents
latent_match = np.allclose(latents_full, latents_chunk, rtol=1e-5)
print(f"Latent features match: {latent_match}")

if not raw_match:
    print("\n⚠️  RAW DATA DIFFERS!")
    diff = np.abs(snippet_full - snippet_chunk)
    print(f"   Max diff: {diff.max()}")
    print(f"   Mean diff: {diff.mean()}")
    
if not proc_match:
    print("\n⚠️  PREPROCESSED DATA DIFFERS!")
    diff = np.abs(snippet_full_proc - snippet_chunk_proc)
    print(f"   Max diff: {diff.max()}")
    print(f"   Mean diff: {diff.mean()}")
    
if not latent_match:
    print("\n⚠️  LATENT FEATURES DIFFER!")
    diff = np.abs(latents_full - latents_chunk)
    print(f"   Max diff: {diff.max()}")
    print(f"   Mean diff: {diff.mean()}")

print("\nDone!")
