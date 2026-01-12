#!/usr/bin/env python3
"""
Debug script to investigate chunked vs non-chunked data loading.
"""

import numpy as np
from pathlib import Path
from blimpy import Waterfall
import warnings

# Path to Voyager files
VOYAGER_DIR = Path("data")
voyager_files = sorted(VOYAGER_DIR.glob("single*.h5"))[:6]

if len(voyager_files) < 6:
    print("ERROR: Voyager files not found in data/")
    exit(1)

print("Testing blimpy data loading behavior...")
print(f"File: {voyager_files[0]}")

# Load full file
print("\n1. Loading FULL file...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    wf_full = Waterfall(str(voyager_files[0]))

fch1 = wf_full.header['fch1']
foff = wf_full.header['foff']
nchans = wf_full.header['nchans']
full_data = wf_full.data.squeeze()

print(f"   fch1: {fch1}")
print(f"   foff: {foff}")
print(f"   nchans: {nchans}")
print(f"   Data shape: {full_data.shape}")

# Test all 8 chunks
N_CHUNKS = 8
chunk_size = nchans // N_CHUNKS

print(f"\n2. Testing {N_CHUNKS} chunks (size={chunk_size} channels each)...")

for chunk_idx in range(N_CHUNKS):
    f_start_chan = chunk_idx * chunk_size
    f_end_chan = (chunk_idx + 1) * chunk_size if chunk_idx < N_CHUNKS - 1 else nchans
    
    # Calculate frequency range
    f_start = fch1 + f_start_chan * foff
    f_end = fch1 + f_end_chan * foff
    
    # Swap if needed (as pipeline does)
    if f_start > f_end:
        f_start, f_end = f_end, f_start
    
    print(f"\n--- Chunk {chunk_idx + 1}/{N_CHUNKS} ---")
    print(f"   Channel range: [{f_start_chan}:{f_end_chan}]")
    print(f"   Freq range: {f_start:.4f} - {f_end:.4f} MHz")
    
    # Load chunk
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wf_chunk = Waterfall(str(voyager_files[0]), f_start=f_start, f_stop=f_end)
    
    chunk_data = wf_chunk.data.squeeze()
    print(f"   Loaded shape: {chunk_data.shape}")
    
    # Extract corresponding slice from full data
    slice_data = full_data[:, f_start_chan:f_end_chan]
    
    # Compare
    if chunk_data.shape == slice_data.shape:
        forward_match = np.allclose(chunk_data, slice_data, rtol=1e-5)
        reversed_match = np.allclose(chunk_data, slice_data[:, ::-1], rtol=1e-5)
        
        if forward_match:
            print(f"   ✅ Match: Forward")
        elif reversed_match:
            print(f"   ⚠️  Match: REVERSED!")
        else:
            print(f"   ❌ NO MATCH!")
            # More debugging
            print(f"   Chunk mean: {chunk_data.mean():.2f}")
            print(f"   Slice mean: {slice_data.mean():.2f}")
            print(f"   Chunk[0,:5]: {chunk_data[0,:5]}")
            print(f"   Slice[0,:5]: {slice_data[0,:5]}")
    else:
        print(f"   ❌ Shape mismatch: chunk={chunk_data.shape}, slice={slice_data.shape}")

print("\n3. Summary of potential issue...")
print("   If all chunks match forward, the issue is elsewhere.")
print("   If some chunks are reversed, we need to flip them in the pipeline.")

print("\nDone!")

