#!/usr/bin/env python3
"""
Debug script to understand why the model has high false positive rate.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cadence_generator import CadenceGenerator, CadenceParams
# Using same preprocessing as training

print("=" * 70)
print("DEBUGGING HIGH FALSE POSITIVE RATE")
print("=" * 70)

# Load models
ENCODER_PATH = Path(__file__).parent.parent / "results" / "models" / "encoder_final.keras"
RF_PATH = Path(__file__).parent.parent / "results" / "models" / "random_forest.joblib"

encoder = tf.keras.models.load_model(ENCODER_PATH)
rf = joblib.load(RF_PATH)
print("Models loaded")

# Generate samples
params = CadenceParams(fchans=4096, tchans=16, snr_base=20, snr_range=30)
cadence_gen = CadenceGenerator(params)

from src.utils.preprocessing import preprocess, downscale

def get_features(encoder, sample):
    outputs = encoder.predict(sample, verbose=0)
    if isinstance(outputs, list):
        z = outputs[2]
    else:
        z = outputs
    return z.flatten()

print("\n" + "=" * 70)
print("ANALYZING LATENT SPACE DISTRIBUTION")
print("=" * 70)

# Generate samples of each type
n_samples = 50

print(f"\nGenerating {n_samples} TRUE samples...")
true_features = []
for i in range(n_samples):
    cadence = cadence_gen.create_true_sample_fast()
    cadence_ds = downscale(np.array([cadence]), factor=8)  # Add batch dim, downscale
    processed = preprocess(cadence_ds, add_channel=True)[0]  # Preprocess and remove batch dim
    features = get_features(encoder, processed)
    true_features.append(features)
true_features = np.array(true_features)

print(f"Generating {n_samples} FALSE samples...")
false_features = []
for i in range(n_samples):
    cadence = cadence_gen.create_false_sample()
    cadence_ds = downscale(np.array([cadence]), factor=8)
    processed = preprocess(cadence_ds, add_channel=True)[0]
    features = get_features(encoder, processed)
    false_features.append(features)
false_features = np.array(false_features)

print(f"Generating {n_samples} PURE NOISE samples...")
noise_features = []
for i in range(n_samples):
    cadence = cadence_gen._get_background()  # Pure noise
    cadence_ds = downscale(np.array([cadence]), factor=8)
    processed = preprocess(cadence_ds, add_channel=True)[0]
    features = get_features(encoder, processed)
    noise_features.append(features)
noise_features = np.array(noise_features)

# Analyze distributions
print("\n" + "=" * 70)
print("LATENT SPACE STATISTICS")
print("=" * 70)

print("\nMean of features (first 5 dimensions):")
print(f"  TRUE samples:  {true_features[:, :5].mean(axis=0)}")
print(f"  FALSE samples: {false_features[:, :5].mean(axis=0)}")
print(f"  NOISE samples: {noise_features[:, :5].mean(axis=0)}")

print("\nStd of features (first 5 dimensions):")
print(f"  TRUE samples:  {true_features[:, :5].std(axis=0)}")
print(f"  FALSE samples: {false_features[:, :5].std(axis=0)}")
print(f"  NOISE samples: {noise_features[:, :5].std(axis=0)}")

# Check RF predictions
print("\n" + "=" * 70)
print("RANDOM FOREST PREDICTIONS")
print("=" * 70)

true_probs = rf.predict_proba(true_features)[:, 1]
false_probs = rf.predict_proba(false_features)[:, 1]
noise_probs = rf.predict_proba(noise_features)[:, 1]

print(f"\nMean probability of ETI class:")
print(f"  TRUE samples:  {true_probs.mean():.4f} (expected: ~1.0)")
print(f"  FALSE samples: {false_probs.mean():.4f} (expected: ~0.0)")
print(f"  NOISE samples: {noise_probs.mean():.4f} (expected: ~0.0)")

print(f"\nProbability distributions:")
print(f"  TRUE:  min={true_probs.min():.4f}, max={true_probs.max():.4f}")
print(f"  FALSE: min={false_probs.min():.4f}, max={false_probs.max():.4f}")
print(f"  NOISE: min={noise_probs.min():.4f}, max={noise_probs.max():.4f}")

# Check what FALSE samples look like
print("\n" + "=" * 70)
print("ANALYZING FALSE SAMPLE GENERATION")
print("=" * 70)

# Check if FALSE samples contain signals
false_sample = cadence_gen.create_false_sample()
print(f"\nFalse sample shape: {false_sample.shape}")
print(f"False sample stats: min={false_sample.min():.2f}, max={false_sample.max():.2f}, mean={false_sample.mean():.2f}")

# Check individual observations in false sample
print("\nPer-observation stats for FALSE sample:")
for i in range(6):
    obs = false_sample[i]
    print(f"  Obs {i}: max={obs.max():.2f}, mean={obs.mean():.2f}, max/mean ratio: {obs.max()/obs.mean():.2f}")

# Compare with pure noise
pure_noise = cadence_gen._get_background()
print("\nPer-observation stats for PURE NOISE:")
for i in range(6):
    obs = pure_noise[i]
    print(f"  Obs {i}: max={obs.max():.2f}, mean={obs.mean():.2f}, max/mean ratio: {obs.max()/obs.mean():.2f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if false_probs.mean() > 0.5:
    print("\n⚠️ FALSE samples are being classified as ETI!")
    print("   Possible causes:")
    print("   1. FALSE samples contain signals (RFI in ALL observations)")
    print("   2. RF is not trained to distinguish noise from ETI pattern")
    print("   3. The encoder produces similar features for all inputs")
    
    # Check if encoder differentiates
    true_mean = true_features.mean(axis=0)
    false_mean = false_features.mean(axis=0)
    noise_mean = noise_features.mean(axis=0)
    
    true_false_dist = np.linalg.norm(true_mean - false_mean)
    true_noise_dist = np.linalg.norm(true_mean - noise_mean)
    false_noise_dist = np.linalg.norm(false_mean - noise_mean)
    
    print(f"\n   Distance between class means:")
    print(f"     TRUE-FALSE:  {true_false_dist:.4f}")
    print(f"     TRUE-NOISE:  {true_noise_dist:.4f}")
    print(f"     FALSE-NOISE: {false_noise_dist:.4f}")
