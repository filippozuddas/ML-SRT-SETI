#!/usr/bin/env python3
"""
Test the trained model on the 8 candidate signals from the paper.
These are real GBT observations that were identified as potential ETI signals.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pickle
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))



print("=" * 70)
print("TESTING ON TOP 8 CANDIDATES FROM PAPER")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = Path(__file__).parent.parent / "assets" / "top_8_data.pkl"
ENCODER_PATH = Path(__file__).parent.parent / "results" / "real_obs_training" / "encoder_final.keras"
RF_PATH = Path(__file__).parent.parent / "results" / "real_obs_training" / "random_forest.joblib"
THRESHOLD = 0.5

print(f"\nConfiguration:")
print(f"  Data: {DATA_PATH}")
print(f"  Encoder: {ENCODER_PATH}")
print(f"  RF: {RF_PATH}")
print(f"  Threshold: {THRESHOLD}")

# ============================================
# LOAD MODELS
# ============================================
print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

print("\nLoading encoder...")
encoder = tf.keras.models.load_model(ENCODER_PATH)
print(f"  Encoder loaded successfully")

print("\nLoading Random Forest...")
rf = joblib.load(RF_PATH)
print(f"  RF loaded successfully")

# ============================================
# LOAD TOP 8 CANDIDATES
# ============================================
print("\n" + "=" * 70)
print("LOADING TOP 8 CANDIDATES")
print("=" * 70)

print(f"\nLoading from {DATA_PATH}...")
with open(DATA_PATH, 'rb') as f:
    candidates_data = pickle.load(f)

print(f"  Found {len(candidates_data)} candidates: {list(candidates_data.keys())}")

# ============================================
# TEST EACH CANDIDATE
# ============================================
print("\n" + "=" * 70)
print("TESTING CANDIDATES")
print("=" * 70)

from src.utils.preprocessing import preprocess, downscale

def preprocess_observation(data):
    """Preprocess a single observation for the model using same pipeline as training.
    
    Input: (6, 16, 1, 4096) - 6 observations, 16 time bins, 1 pol, 4096 freq bins
    Output: (6, 16, 512, 1) - ready for the model
    """
    print(f"    Raw data shape: {data.shape}")
    
    if data.shape != (6, 16, 1, 4096):
        print(f"    ⚠️ Unexpected shape, expected (6, 16, 1, 4096)")
        return None
    
    # Reshape: (6, 16, 1, 4096) → (1, 6, 16, 4096) for downscale function
    data_reshaped = data.squeeze(axis=2)  # (6, 16, 4096)
    data_batch = data_reshaped[np.newaxis, ...]  # (1, 6, 16, 4096)
    
    # Step 1: Downscale by factor 8 (4096 → 512) - same as training!
    data_ds = downscale(data_batch, factor=8)  # (1, 6, 16, 512)
    
    # Step 2: Apply log normalization with preprocess - same as training!
    processed = preprocess(data_ds, add_channel=True)[0]  # (6, 16, 512, 1)
    
    print(f"    Processed data shape: {processed.shape}")
    
    return processed

def predict_candidate(encoder, rf, data, threshold=0.5):
    """
    Predict if a candidate is ETI signal.
    Returns probability and classification.
    """
    # Process each observation
    processed = preprocess_observation(data)
    if processed is None:
        return None, None, "Failed to preprocess"
    
    # Flatten cadences for VAE: (6, 16, 512, 1) -> (6, 16, 512, 1)
    samples = processed  # Already in right shape
    
    # Get latent representations
    z_mean, z_log_var, z = encoder.predict(samples, verbose=0)
    
    # Recombine for RF: (6, 8) -> (1, 48)
    features = z.flatten().reshape(1, -1)
    
    # Predict
    proba = rf.predict_proba(features)[0]
    prediction = proba[1] > threshold  # Class 1 = ETI
    
    return proba, prediction, z

results = []

for name in sorted(candidates_data.keys()):
    print(f"\n--- {name} ---")
    
    candidate = candidates_data[name]
    data = candidate['data']
    freq = candidate.get('frequency', 'Unknown')
    
    print(f"  Frequency: {freq}")
    
    try:
        proba, prediction, z = predict_candidate(encoder, rf, data, THRESHOLD)
        
        if proba is not None:
            print(f"  Probability (ETI): {proba[1]:.4f}")
            print(f"  Classification: {'✅ ETI DETECTED' if prediction else '❌ Not ETI'}")
            results.append({
                'name': name,
                'probability': proba[1],
                'classification': prediction,
                'status': 'Success'
            })
        else:
            print(f"  ⚠️ Could not process")
            results.append({
                'name': name,
                'probability': None,
                'classification': None,
                'status': 'Failed'
            })
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append({
            'name': name,
            'probability': None,
            'classification': None,
            'status': f'Error: {e}'
        })

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n┌──────────┬─────────────┬────────────────┐")
print("│ Candidate│ Probability │ Classification │")
print("├──────────┼─────────────┼────────────────┤")

detected = 0
total = 0

for r in results:
    if r['probability'] is not None:
        cls = "ETI" if r['classification'] else "Not ETI"
        print(f"│ {r['name']:8} │ {r['probability']:11.4f} │ {cls:14} │")
        total += 1
        if r['classification']:
            detected += 1
    else:
        print(f"│ {r['name']:8} │      N/A    │ {r['status']:14} │")

print("└──────────┴─────────────┴────────────────┘")

if total > 0:
    print(f"\n✓ Detected {detected}/{total} candidates as ETI signals ({100*detected/total:.1f}%)")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
