"""
SNR Sensitivity Testing.

This script evaluates the model's detection performance across
different signal-to-noise ratios to determine the detection threshold.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=" * 70)
print("SETI Signal Detector - SNR Sensitivity Test")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
SNR_VALUES = [10, 12, 15, 20, 25, 30, 40, 50]  
SAMPLES_PER_SNR = 100  # Samples per SNR level
PROBABILITY_THRESHOLD = 0.5

# Model paths - adjust if needed
MODEL_DIR = Path("checkpoints/large_scale_training")
ENCODER_PATH = MODEL_DIR / "encoder_final.keras"
RF_PATH = MODEL_DIR / "random_forest.joblib"

OUTPUT_DIR = Path("checkpoints/snr_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  SNR values: {SNR_VALUES}")
print(f"  Samples per SNR: {SAMPLES_PER_SNR}")
print(f"  Threshold: {PROBABILITY_THRESHOLD}")

# ============================================
# IMPORTS
# ============================================
from src.data.cadence_generator import CadenceGenerator, CadenceParams
from src.models.sampling import Sampling
from src.utils.preprocessing import preprocess, downscale, combine_cadences, recombine_latents
from multiprocessing import Pool, cpu_count

# Number of parallel workers
NUM_WORKERS = max(1, cpu_count() - 2)
print(f"Using {NUM_WORKERS} CPU workers for parallel generation")

# Worker functions for multiprocessing
def _generate_true_sample_snr(args):
    """Worker function for generating a true sample at specific SNR."""
    seed, snr_base = args
    params = CadenceParams(fchans=4096, tchans=16, snr_base=snr_base, snr_range=0)
    cadence_gen = CadenceGenerator(params, seed=seed)
    return cadence_gen.create_true_sample_fast()

def _generate_false_sample_snr(args):
    """Worker function for generating a FALSE sample (pure noise).
    
    For SNR sensitivity testing, we use pure noise (no signal) as FALSE.
    This provides a cleaner measurement of detection vs noise.
    """
    seed, _ = args  # snr_base not used for pure noise
    params = CadenceParams(fchans=4096, tchans=16, snr_base=10, snr_range=0)
    cadence_gen = CadenceGenerator(params, seed=seed)
    # Return pure background noise (no signal)
    return cadence_gen._get_background()

# ============================================
# LOAD MODELS
# ============================================
print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

print("\nLoading encoder...")
encoder = tf.keras.models.load_model(ENCODER_PATH, custom_objects={'Sampling': Sampling})
print(f"  Encoder loaded: {ENCODER_PATH}")

print("\nLoading Random Forest...")
rf = joblib.load(RF_PATH)
print(f"  RF loaded: {RF_PATH}")

# ============================================
# SNR SENSITIVITY TESTING
# ============================================
print("\n" + "=" * 70)
print("SNR SENSITIVITY TESTING")
print("=" * 70)

results = {
    'snr': [],
    'tpr': [],        # True Positive Rate (recall for TRUE class)
    'fpr': [],        # False Positive Rate
    'precision': [],
    'accuracy': [],
    'f1': []
}

for snr in SNR_VALUES:
    print(f"\n--- Testing SNR = {snr} ---")
    
    # Prepare arguments for parallel generation
    base_seed = snr * 1000  # Different base seed per SNR level
    true_args = [(base_seed + i, snr) for i in range(SAMPLES_PER_SNR)]
    false_args = [(base_seed + 10000 + i, snr) for i in range(SAMPLES_PER_SNR)]
    
    # Generate TRUE and FALSE samples in parallel
    with Pool(NUM_WORKERS) as pool:
        true_samples = list(tqdm(pool.imap(_generate_true_sample_snr, true_args), 
                                  total=SAMPLES_PER_SNR, desc=f"TRUE SNR={snr}", leave=False))
        false_samples = list(tqdm(pool.imap(_generate_false_sample_snr, false_args), 
                                   total=SAMPLES_PER_SNR, desc=f"FALSE SNR={snr}", leave=False))
    
    true_samples = np.array(true_samples)
    false_samples = np.array(false_samples)
    
    # Preprocess
    true_samples = downscale(true_samples, factor=8)
    false_samples = downscale(false_samples, factor=8)
    true_samples = preprocess(true_samples, add_channel=True)
    false_samples = preprocess(false_samples, add_channel=True)
    
    # Extract latents
    true_flat = combine_cadences(true_samples).astype(np.float32)
    false_flat = combine_cadences(false_samples).astype(np.float32)
    
    true_latents = encoder.predict(true_flat, batch_size=100, verbose=0)[2]
    false_latents = encoder.predict(false_flat, batch_size=100, verbose=0)[2]
    
    # Recombine for cadence-level
    true_latents_cadence = recombine_latents(true_latents)
    false_latents_cadence = recombine_latents(false_latents)
    
    # Handle NaN/Inf
    true_latents_cadence = np.nan_to_num(true_latents_cadence, nan=0, posinf=100, neginf=-100)
    false_latents_cadence = np.nan_to_num(false_latents_cadence, nan=0, posinf=100, neginf=-100)
    
    # Classify
    X_test = np.vstack([true_latents_cadence, false_latents_cadence])
    y_true = np.concatenate([np.ones(SAMPLES_PER_SNR), np.zeros(SAMPLES_PER_SNR)])
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    # TPR = True Positive Rate = TP / (TP + FN) = Recall for positive class
    # FPR = False Positive Rate = FP / (FP + TN)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    results['snr'].append(snr)
    results['tpr'].append(tpr)
    results['fpr'].append(fpr)
    results['precision'].append(precision)
    results['accuracy'].append(accuracy)
    results['f1'].append(f1)
    
    print(f"  TPR (Detection Rate): {tpr:.4f}")
    print(f"  FPR (False Alarm):    {fpr:.4f}")
    print(f"  Accuracy:             {accuracy:.4f}")

# ============================================
# RESULTS SUMMARY
# ============================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n┌─────────┬─────────┬─────────┬───────────┬──────────┐")
print("│   SNR   │   TPR   │   FPR   │ Precision │ Accuracy │")
print("├─────────┼─────────┼─────────┼───────────┼──────────┤")
for i in range(len(results['snr'])):
    print(f"│ {results['snr'][i]:7} │ {results['tpr'][i]:7.4f} │ {results['fpr'][i]:7.4f} │ "
          f"{results['precision'][i]:9.4f} │ {results['accuracy'][i]:8.4f} │")
print("└─────────┴─────────┴─────────┴───────────┴──────────┘")

# Find detection threshold (SNR where TPR >= 0.9)
for i, (snr, tpr) in enumerate(zip(results['snr'], results['tpr'])):
    if tpr >= 0.9:
        print(f"\n✓ Detection threshold (TPR ≥ 90%): SNR ≥ {snr}")
        break
else:
    print("\n⚠ Model does not achieve 90% TPR at any tested SNR")

# ============================================
# PLOT RESULTS
# ============================================
print("\nGenerating plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: TPR and FPR vs SNR
axes[0].plot(results['snr'], results['tpr'], 'g-o', linewidth=2, markersize=8, label='TPR (Detection Rate)')
axes[0].plot(results['snr'], results['fpr'], 'r-s', linewidth=2, markersize=8, label='FPR (False Alarm)')
axes[0].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% TPR Target')
axes[0].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='1% FPR Target')
axes[0].set_xlabel('SNR', fontsize=12)
axes[0].set_ylabel('Rate', fontsize=12)
axes[0].set_title('Detection Rate vs SNR', fontsize=14)
axes[0].legend(loc='center right')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# Plot 2: Accuracy and F1 vs SNR
axes[1].plot(results['snr'], results['accuracy'], 'b-o', linewidth=2, markersize=8, label='Accuracy')
axes[1].plot(results['snr'], results['f1'], 'm-^', linewidth=2, markersize=8, label='F1 Score')
axes[1].set_xlabel('SNR', fontsize=12)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Performance Metrics vs SNR', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

# Plot 3: Precision-Recall tradeoff
axes[2].plot(results['snr'], results['precision'], 'c-o', linewidth=2, markersize=8, label='Precision')
axes[2].plot(results['snr'], results['tpr'], 'y-s', linewidth=2, markersize=8, label='Recall (TPR)')
axes[2].set_xlabel('SNR', fontsize=12)
axes[2].set_ylabel('Score', fontsize=12)
axes[2].set_title('Precision & Recall vs SNR', fontsize=14)
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'snr_sensitivity.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to {OUTPUT_DIR / 'snr_sensitivity.png'}")

# Save results to CSV
import csv
with open(OUTPUT_DIR / 'snr_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['SNR', 'TPR', 'FPR', 'Precision', 'Accuracy', 'F1'])
    for i in range(len(results['snr'])):
        writer.writerow([
            results['snr'][i],
            results['tpr'][i],
            results['fpr'][i],
            results['precision'][i],
            results['accuracy'][i],
            results['f1'][i]
        ])
print(f"Results saved to {OUTPUT_DIR / 'snr_results.csv'}")

print("\n" + "=" * 70)
print("SNR SENSITIVITY TEST COMPLETE")
print("=" * 70)
