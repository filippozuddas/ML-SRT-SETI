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
DATA_PATH = Path(__file__).parent.parent / "top_8_data.pkl"
ENCODER_PATH = Path(__file__).parent.parent / "checkpoints" / "large_scale_training" / "encoder_final.keras"
RF_PATH = Path(__file__).parent.parent / "checkpoints" / "large_scale_training" / "random_forest.joblib"
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

def preprocess_observation(data):
    """Preprocess a single observation for the model."""
    # data shape: (6, time, 1, freq) or similar
    # We need to reshape to (6, 16, 512, 1)
    
    print(f"    Raw data shape: {data.shape}")
    
    # If data has wrong dimensions, try to reshape
    if len(data.shape) == 4:
        # Assume (6, time, ?, freq)
        cadences = []
        for i in range(6):
            obs = data[i]
            # Squeeze any singleton dimensions
            obs = np.squeeze(obs)
            
            # If 2D (time, freq), add channel
            if len(obs.shape) == 2:
                # Downsample to 16x512 if needed
                if obs.shape[0] != 16 or obs.shape[1] != 512:
                    # Simple downsampling via slicing/binning
                    from scipy.ndimage import zoom
                    zoom_factors = (16/obs.shape[0], 512/obs.shape[1])
                    obs = zoom(obs, zoom_factors, order=1)
                obs = obs[..., np.newaxis]
            cadences.append(obs)
        
        processed = np.array(cadences)  # Shape: (6, 16, 512, 1)
        
        # Normalize
        processed = (processed - processed.mean()) / (processed.std() + 1e-8)
        
        print(f"    Processed data shape: {processed.shape}")

        return processed
    
    return None

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
