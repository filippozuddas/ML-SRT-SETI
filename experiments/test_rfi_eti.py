#!/usr/bin/env python3
"""
Test the model on a challenging sample with both RFI and ETI signals.
This tests the model's ability to detect ETI in the presence of interference.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cadence_generator import CadenceGenerator, CadenceParams
from src.utils.preprocessing import preprocess, downscale

print("=" * 70)
print("TESTING ON NOISY SAMPLE WITH RFI + ETI")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
ENCODER_PATH = Path(__file__).parent.parent / "results" / "real_obs_training" / "C_band" / "encoder_final.keras"
RF_PATH = Path(__file__).parent.parent / "results" / "real_obs_training" / "C_band" / "random_forest.joblib"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "rfi_eti_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.5

print(f"\nConfiguration:")
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
# GENERATE CHALLENGING SAMPLES
# ============================================
print("\n" + "=" * 70)
print("GENERATING TEST SAMPLES")
print("=" * 70)

params = CadenceParams(
    fchans=4096,
    tchans=16,
    snr_base=15,  # Lower SNR for ETI (harder to detect)
    snr_range=10
)
cadence_gen = CadenceGenerator(params)

def add_rfi_pattern(cadence, rfi_strength=2.0, num_rfi=5):
    """Add RFI patterns to all 6 observations (mimics real interference)."""
    noisy = cadence.copy()
    
    for _ in range(num_rfi):
        # Random frequency position for RFI
        freq_pos = np.random.randint(100, cadence.shape[2] - 100)
        freq_width = np.random.randint(5, 30)
        
        # RFI is present in ALL observations (unlike ETI which is ON only)
        for obs_idx in range(6):
            # Add broadband RFI
            intensity = np.random.uniform(0.5, 1.0) * rfi_strength
            noisy[obs_idx, :, freq_pos:freq_pos+freq_width] += intensity * np.abs(np.random.randn())
            
            # Add narrowband RFI spikes
            if np.random.random() > 0.5:
                spike_pos = np.random.randint(0, cadence.shape[2])
                noisy[obs_idx, :, spike_pos] += intensity * 2
    
    return noisy

def preprocess_sample(cadence):
    """Preprocess cadence for model input using same pipeline as training."""
    # Add batch dimension, downscale, then preprocess with log normalization
    cadence_ds = downscale(np.array([cadence]), factor=8)  # (1, 6, 16, 512)
    processed = preprocess(cadence_ds, add_channel=True)[0]  # (6, 16, 512, 1)
    return processed

def predict_sample(encoder, rf, sample, threshold=0.5):
    """Predict if sample contains ETI."""
    # Get latent representation
    outputs = encoder.predict(sample, verbose=0)
    if isinstance(outputs, list):
        z = outputs[2]
    else:
        z = outputs
    
    # Flatten for RF
    features = z.flatten().reshape(1, -1)
    
    # Predict
    proba = rf.predict_proba(features)[0]
    prediction = proba[1] > threshold
    
    return proba[1], prediction

# Test cases
test_cases = []

# Case 1: Pure noise (should be FALSE)
print("\n1. Generating PURE NOISE sample...")
noise_only = cadence_gen._get_background()
test_cases.append(("Pure Noise", noise_only, False))

# Case 2: Noise + RFI only (should be FALSE) 
print("2. Generating NOISE + RFI sample...")
noise_rfi = add_rfi_pattern(cadence_gen._get_background(), rfi_strength=3.0, num_rfi=10)
test_cases.append(("Noise + RFI", noise_rfi, False))

# Case 3: ETI signal only (should be TRUE)
print("3. Generating ETI signal sample...")
eti_only = cadence_gen.create_true_sample_fast()
test_cases.append(("ETI Only", eti_only, True))

# Case 4: ETI + light RFI (should be TRUE)
print("4. Generating ETI + light RFI sample...")
eti_light_rfi = add_rfi_pattern(cadence_gen.create_true_sample_fast(), rfi_strength=1.5, num_rfi=3)
test_cases.append(("ETI + Light RFI", eti_light_rfi, True))

# Case 5: ETI + heavy RFI (challenging - should be TRUE)
print("5. Generating ETI + HEAVY RFI sample (challenging)...")
eti_heavy_rfi = add_rfi_pattern(cadence_gen.create_true_sample_fast(), rfi_strength=4.0, num_rfi=15)
test_cases.append(("ETI + Heavy RFI", eti_heavy_rfi, True))

# Case 6: Very weak ETI + moderate RFI (very challenging)
print("6. Generating WEAK ETI + moderate RFI sample (very challenging)...")
weak_params = CadenceParams(fchans=4096, tchans=16, snr_base=8, snr_range=5)
weak_gen = CadenceGenerator(weak_params)
weak_eti_rfi = add_rfi_pattern(weak_gen.create_true_sample_fast(), rfi_strength=2.0, num_rfi=8)
test_cases.append(("Weak ETI + RFI", weak_eti_rfi, True))

# ============================================
# RUN TESTS
# ============================================
print("\n" + "=" * 70)
print("TESTING SAMPLES")
print("=" * 70)

results = []

for name, raw_cadence, expected in test_cases:
    processed = preprocess_sample(raw_cadence)
    prob, prediction = predict_sample(encoder, rf, processed, THRESHOLD)
    
    correct = prediction == expected
    status = "✅" if correct else "❌"
    
    print(f"\n{name}:")
    print(f"  Probability: {prob:.4f}")
    print(f"  Prediction: {'ETI' if prediction else 'Not ETI'}")
    print(f"  Expected: {'ETI' if expected else 'Not ETI'}")
    print(f"  Result: {status} {'CORRECT' if correct else 'WRONG'}")
    
    results.append({
        "name": name,
        "probability": prob,
        "prediction": prediction,
        "expected": expected,
        "correct": correct
    })

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n┌─────────────────────────┬─────────────┬────────────┬──────────┬────────┐")
print("│ Test Case               │ Probability │ Prediction │ Expected │ Result │")
print("├─────────────────────────┼─────────────┼────────────┼──────────┼────────┤")

correct_count = 0
for r in results:
    pred = "ETI" if r["prediction"] else "Not ETI"
    exp = "ETI" if r["expected"] else "Not ETI"
    status = "✅" if r["correct"] else "❌"
    if r["correct"]:
        correct_count += 1
    print(f"│ {r['name']:23} │ {r['probability']:11.4f} │ {pred:10} │ {exp:8} │ {status:5} │")

print("└─────────────────────────┴─────────────┴────────────┴──────────┴────────┘")

print(f"\n✓ Accuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)")

# ============================================
# VISUALIZE ONE CHALLENGING CASE
# ============================================
print("\n" + "=" * 70)
print("SAVING VISUALIZATION")
print("=" * 70)

# Plot ETI + Heavy RFI case
heavy_rfi_case = test_cases[4][1]  # ETI + Heavy RFI

heavy_rfi_prob = results[4]["probability"]

# Stack all 6 observations vertically
stacked = np.vstack([heavy_rfi_case[i] for i in range(6)])

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(stacked, aspect='auto', cmap='viridis', interpolation='nearest')

# Add horizontal lines to separate observations
obs_labels = ["ON₁", "OFF₁", "ON₂", "OFF₂", "ON₃", "OFF₃"]
tchans = heavy_rfi_case.shape[1]
for i in range(1, 6):
    ax.axhline(y=i * tchans - 0.5, color='white', linewidth=1.5, linestyle='--', alpha=0.7)

# Add observation labels on the left
for i, label in enumerate(obs_labels):
    ax.text(-50, i * tchans + tchans/2, label, fontsize=12, fontweight='bold',
            ha='right', va='center', color='white' if i % 2 == 0 else 'lightblue')

ax.set_xlabel("Frequency Bins", fontsize=12)
ax.set_ylabel("Time Bins (6 observations stacked)", fontsize=12)
ax.set_title(f"ETI + Heavy RFI Sample (Challenging) - P(ETI) = {heavy_rfi_prob:.4f}", fontsize=14)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Intensity", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eti_heavy_rfi_sample.png", dpi=150, bbox_inches='tight')
print(f"  Saved visualization to {OUTPUT_DIR / 'eti_heavy_rfi_sample.png'}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
