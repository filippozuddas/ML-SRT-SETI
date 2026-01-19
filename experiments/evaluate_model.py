#!/usr/bin/env python3
"""
Evaluate the trained model and compute metrics (Accuracy, AUC-ROC, etc.)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Add parent for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import preprocess, downscale, combine_cadences, recombine_latents
from src.data.cadence_generator import CadenceGenerator, CadenceParams

print("=" * 70)
print("MODEL EVALUATION - METRICS")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
ENCODER_PATH = Path("/content/filippo/ML-SRT-SETI/results/real_obs_training/K_band_2/encoder_final.keras")
RF_PATH = Path("/content/filippo/ML-SRT-SETI/results/real_obs_training/K_band_2/random_forest.joblib")
PLATE_PATH = Path("/content/filippo/ML-SRT-SETI/data/srt_training/backgrounds_18GHz.npz")
OUTPUT_DIR = Path("/content/filippo/ML-SRT-SETI/results/real_obs_training/K_band_2")

N_EVAL_SAMPLES = 2000  # Number of cadences for evaluation

# ============================================
# LOAD MODELS
# ============================================
print("\nLoading models...")
encoder = tf.keras.models.load_model(ENCODER_PATH)
rf = joblib.load(RF_PATH)
print("  ✓ Encoder loaded")
print("  ✓ Random Forest loaded")

# Load plate
print("\nLoading SRT plate...")
plate_data = np.load(PLATE_PATH)
plate = plate_data['backgrounds']
print(f"  Plate shape: {plate.shape}")

# ============================================
# GENERATE EVALUATION DATA
# ============================================
print("\n" + "=" * 70)
print("GENERATING EVALUATION DATA")
print("=" * 70)

print(f"\nGenerating {N_EVAL_SAMPLES} True samples...")
true_samples = []
for i in range(N_EVAL_SAMPLES):
    params = CadenceParams(fchans=4096, tchans=16, snr_base=20, snr_range=30)
    gen = CadenceGenerator(params, plate=plate, seed=i + 50000)
    true_samples.append(gen.create_true_sample_fast())
true_samples = np.array(true_samples)

print(f"Generating {N_EVAL_SAMPLES} False samples...")
false_samples = []
for i in range(N_EVAL_SAMPLES):
    params = CadenceParams(fchans=4096, tchans=16, snr_base=20, snr_range=30)
    gen = CadenceGenerator(params, plate=plate, seed=i + 60000)
    false_samples.append(gen.create_false_sample())
false_samples = np.array(false_samples)

print(f"\nRaw shapes: True={true_samples.shape}, False={false_samples.shape}")

# Preprocess
print("Preprocessing...")
true_ds = downscale(true_samples, factor=8)
false_ds = downscale(false_samples, factor=8)

true_proc = preprocess(true_ds, add_channel=True)
false_proc = preprocess(false_ds, add_channel=True)

# Flatten for encoder
true_flat = combine_cadences(true_proc)  # (N*6, 16, 512, 1)
false_flat = combine_cadences(false_proc)

print(f"Preprocessed shapes: True={true_flat.shape}, False={false_flat.shape}")

# ============================================
# EXTRACT LATENTS
# ============================================
print("\n" + "=" * 70)
print("EXTRACTING LATENT REPRESENTATIONS")
print("=" * 70)

print("\nEncoding True samples...")
true_latents = encoder.predict(true_flat, batch_size=512, verbose=1)[2]

print("Encoding False samples...")
false_latents = encoder.predict(false_flat, batch_size=512, verbose=1)[2]

# Recombine for RF
true_latents_cadence = recombine_latents(true_latents)  # (N, 48)
false_latents_cadence = recombine_latents(false_latents)

print(f"\nCadence latent shapes: True={true_latents_cadence.shape}, False={false_latents_cadence.shape}")

# ============================================
# EVALUATE RANDOM FOREST
# ============================================
print("\n" + "=" * 70)
print("RANDOM FOREST EVALUATION")
print("=" * 70)

# Prepare data
X = np.vstack([true_latents_cadence, false_latents_cadence])
y = np.concatenate([np.ones(len(true_latents_cadence)), np.zeros(len(false_latents_cadence))])

# Handle NaN/Inf
if not np.isfinite(X).all():
    print("Warning: Found NaN/Inf, replacing...")
    X = np.nan_to_num(X, nan=0, posinf=100, neginf=-100)

# Predict
print("\nMaking predictions...")
y_pred = rf.predict(X)
y_prob = rf.predict_proba(X)[:, 1]

# ============================================
# COMPUTE METRICS
# ============================================
print("\n" + "=" * 70)
print("METRICS")
print("=" * 70)

accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
cm = confusion_matrix(y, y_pred)

print(f"\n{'='*50}")
print(f"  ACCURACY:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  AUC-ROC:   {auc:.4f}")
print(f"{'='*50}")

print("\nConfusion Matrix:")
print(f"                  Predicted")
print(f"                  RFI    ETI")
print(f"  Actual RFI   {cm[0,0]:5}  {cm[0,1]:5}")
print(f"  Actual ETI   {cm[1,0]:5}  {cm[1,1]:5}")

print("\n" + "-" * 50)
print("Classification Report:")
print("-" * 50)
print(classification_report(y, y_pred, target_names=['RFI (False)', 'ETI (True)']))

# ============================================
# SAVE METRICS
# ============================================
metrics = {
    'accuracy': accuracy,
    'auc_roc': auc,
    'confusion_matrix': cm.tolist(),
    'n_samples': N_EVAL_SAMPLES
}

import json
with open(OUTPUT_DIR / 'evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to {OUTPUT_DIR / 'evaluation_metrics.json'}")

# ============================================
# ROC CURVE
# ============================================
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - ETI Signal Detection', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=150)
print(f"ROC curve saved to {OUTPUT_DIR / 'roc_curve.png'}")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
