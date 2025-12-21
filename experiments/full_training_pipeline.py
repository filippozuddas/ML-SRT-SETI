"""
Full Training and Evaluation Pipeline.

This script:
1. Trains the VAE for sufficient epochs
2. Visualizes the latent space to verify TRUE/FALSE separation
3. Trains and evaluates the Random Forest classifier
4. Generates performance metrics
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import time
import gc

print("=" * 70)
print("SETI Signal Detector - Full Training & Evaluation Pipeline")
print("=" * 70)

# ============================================
# CONFIGURATION
# ============================================
NUM_TRAIN_SAMPLES = 500       # Samples for VAE training
NUM_EVAL_SAMPLES = 200        # Samples for evaluation
BATCH_SIZE = 25
VAE_EPOCHS = 20               # Total VAE epochs
LEARNING_RATE = 0.001
RF_ESTIMATORS = 500
SNR_BASE = 30
SNR_RANGE = 30

OUTPUT_DIR = Path("checkpoints/full_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Training samples: {NUM_TRAIN_SAMPLES}")
print(f"  Eval samples: {NUM_EVAL_SAMPLES}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  VAE epochs: {VAE_EPOCHS}")
print(f"  SNR range: {SNR_BASE} ± {SNR_RANGE}")

# ============================================
# IMPORTS
# ============================================
from src.data.cadence_generator import CadenceGenerator, CadenceParams
from src.models.vae import build_vae
from src.utils.preprocessing import preprocess, downscale, combine_cadences, recombine_latents

# ============================================
# DATA GENERATION
# ============================================
print("\n" + "=" * 70)
print("PHASE 1: DATA GENERATION")
print("=" * 70)

params = CadenceParams(fchans=4096, tchans=16, snr_base=SNR_BASE, snr_range=SNR_RANGE)
cadence_gen = CadenceGenerator(params, seed=42)

print("\nGenerating training data...")
start_time = time.time()

true_train = []
false_train = []
for i in tqdm(range(NUM_TRAIN_SAMPLES), desc="Training samples"):
    true_train.append(cadence_gen.create_true_sample_fast())
    false_train.append(cadence_gen.create_false_sample())

true_train = np.array(true_train)
false_train = np.array(false_train)

print(f"Generated in {time.time() - start_time:.1f}s")
print(f"  True training: {true_train.shape}")
print(f"  False training: {false_train.shape}")

# Preprocess
print("\nPreprocessing...")
true_train = downscale(true_train, factor=8)
false_train = downscale(false_train, factor=8)
true_train = preprocess(true_train, add_channel=True)
false_train = preprocess(false_train, add_channel=True)

vae_train = combine_cadences(true_train)
print(f"  VAE input shape: {vae_train.shape}")

# Generate evaluation data
print("\nGenerating evaluation data...")
cadence_gen_eval = CadenceGenerator(params, seed=999)

true_eval = []
false_eval = []
for i in tqdm(range(NUM_EVAL_SAMPLES), desc="Eval samples"):
    true_eval.append(cadence_gen_eval.create_true_sample_fast())
    false_eval.append(cadence_gen_eval.create_false_sample())

true_eval = np.array(true_eval)
false_eval = np.array(false_eval)

true_eval = downscale(true_eval, factor=8)
false_eval = downscale(false_eval, factor=8)
true_eval = preprocess(true_eval, add_channel=True)
false_eval = preprocess(false_eval, add_channel=True)

print(f"  True eval: {true_eval.shape}")
print(f"  False eval: {false_eval.shape}")

# ============================================
# VAE TRAINING
# ============================================
print("\n" + "=" * 70)
print("PHASE 2: VAE TRAINING")
print("=" * 70)

model = build_vae(
    input_shape=(16, 512, 1),
    latent_dim=8,
    dense_units=512,
    alpha=10,
    beta=1,
    learning_rate=LEARNING_RATE
)

num_batches = NUM_TRAIN_SAMPLES // BATCH_SIZE
history = {'loss': [], 'recon': [], 'kl': [], 'true': [], 'false': []}

print(f"\nTraining for {VAE_EPOCHS} epochs...")
print("-" * 70)

for epoch in range(VAE_EPOCHS):
    epoch_losses = []
    indices = np.random.permutation(NUM_TRAIN_SAMPLES)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch_indices = indices[start_idx:end_idx]
        
        vae_indices = []
        for bi in batch_indices:
            vae_indices.extend(range(bi * 6, (bi + 1) * 6))
        
        vae_batch = vae_train[vae_indices].astype(np.float32)
        true_batch = true_train[batch_indices].astype(np.float32)
        false_batch = false_train[batch_indices].astype(np.float32)
        
        loss_dict = model.train_step(((vae_batch, true_batch, false_batch), vae_batch))
        epoch_losses.append(float(loss_dict['loss']))
    
    mean_loss = np.mean(epoch_losses)
    history['loss'].append(mean_loss)
    history['recon'].append(float(loss_dict['reconstruction_loss']))
    history['kl'].append(float(loss_dict['kl_loss']))
    history['true'].append(float(loss_dict['true_loss']))
    history['false'].append(float(loss_dict['false_loss']))
    
    print(f"Epoch {epoch + 1:3d}/{VAE_EPOCHS} - loss: {mean_loss:.2f} - "
          f"recon: {history['recon'][-1]:.4f} - kl: {history['kl'][-1]:.2f} - "
          f"true: {history['true'][-1]:.2f} - false: {history['false'][-1]:.2f}")

print("-" * 70)

# Save VAE
model.encoder.save(str(OUTPUT_DIR / 'encoder.keras'))
model.decoder.save(str(OUTPUT_DIR / 'decoder.keras'))
print(f"VAE saved to {OUTPUT_DIR}")

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(history['loss'], 'b-')
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history['recon'], 'g-', label='Reconstruction')
axes[0, 1].plot(history['kl'], 'r-', label='KL Divergence')
axes[0, 1].set_title('VAE Losses')
axes[0, 1].legend()
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history['true'], 'c-', label='True Clustering')
axes[1, 0].plot(history['false'], 'm-', label='False Clustering')
axes[1, 0].set_title('Contrastive Losses')
axes[1, 0].legend()
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(np.array(history['true']) - np.array(history['false']), 'k-')
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('True - False Loss (should → 0)')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=150)
print(f"Training plot saved")

# Plot reconstructions
print("Generating reconstruction examples...")
sample_indices = np.random.choice(len(vae_train), 6, replace=False)
sample_input = vae_train[sample_indices].astype(np.float32)
_, _, _, reconstructed = model(sample_input, training=False)
reconstructed = reconstructed.numpy()

fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for i in range(6):
    axes[0, i].imshow(sample_input[i, :, :, 0], aspect='auto', cmap='hot')
    axes[0, i].set_title(f'Original {i+1}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(reconstructed[i, :, :, 0], aspect='auto', cmap='hot')
    axes[1, i].set_title(f'Reconstructed {i+1}')
    axes[1, i].axis('off')

plt.suptitle('VAE Reconstruction Quality', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'reconstructions.png', dpi=150)
print(f"Reconstruction plot saved")

# ============================================
# LATENT SPACE VISUALIZATION
# ============================================
print("\n" + "=" * 70)
print("PHASE 3: LATENT SPACE ANALYSIS")
print("=" * 70)

print("\nExtracting latent representations...")

# Extract latents from evaluation data
true_flat = combine_cadences(true_eval).astype(np.float32)
false_flat = combine_cadences(false_eval).astype(np.float32)

true_latents = model.encoder.predict(true_flat, batch_size=100)[2]  # z
false_latents = model.encoder.predict(false_flat, batch_size=100)[2]

print(f"  True latents: {true_latents.shape}")
print(f"  False latents: {false_latents.shape}")

# Recombine for cadence-level analysis
true_latents_cadence = recombine_latents(true_latents)  # (N, 48)
false_latents_cadence = recombine_latents(false_latents)  # (N, 48)

print(f"  True cadence latents: {true_latents_cadence.shape}")
print(f"  False cadence latents: {false_latents_cadence.shape}")

# Visualize latent space (2D projection)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot different dimension pairs
dim_pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3), (0, 4)]

for ax, (d1, d2) in zip(axes.flat, dim_pairs):
    ax.scatter(false_latents[:, d1], false_latents[:, d2], 
               alpha=0.3, s=10, c='blue', label='False')
    ax.scatter(true_latents[:, d1], true_latents[:, d2], 
               alpha=0.3, s=10, c='red', label='True')
    ax.set_xlabel(f'Latent {d1}')
    ax.set_ylabel(f'Latent {d2}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Latent Space Visualization (Individual Observations)', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'latent_space.png', dpi=150)
print(f"Latent space plot saved")

# ============================================
# RANDOM FOREST TRAINING
# ============================================
print("\n" + "=" * 70)
print("PHASE 4: RANDOM FOREST CLASSIFIER")
print("=" * 70)

# Prepare training data for RF
X = np.vstack([true_latents_cadence, false_latents_cadence])
y = np.concatenate([np.ones(len(true_latents_cadence)), np.zeros(len(false_latents_cadence))])

# Handle NaN/Inf
if not np.isfinite(X).all():
    print("Warning: Found NaN/Inf, replacing...")
    X = np.nan_to_num(X, nan=0, posinf=100, neginf=-100)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nRandom Forest training...")
print(f"  Train: {X_train.shape} - Test: {X_test.shape}")

rf = RandomForestClassifier(
    n_estimators=RF_ESTIMATORS,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n--- Random Forest Results ---")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  AUC-ROC:  {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['False (RFI)', 'True (ETI)']))

# Save RF
joblib.dump(rf, OUTPUT_DIR / 'random_forest.joblib')
print(f"Random Forest saved")

# Plot confusion matrix style results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Probability distributions
axes[0].hist(y_prob[y_test == 0], bins=30, alpha=0.6, label='False samples', color='blue')
axes[0].hist(y_prob[y_test == 1], bins=30, alpha=0.6, label='True samples', color='red')
axes[0].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
axes[0].set_xlabel('Predicted Probability (True)')
axes[0].set_ylabel('Count')
axes[0].set_title('Classification Probability Distribution')
axes[0].legend()

# Feature importance (first 16 most important)
importances = rf.feature_importances_
indices = np.argsort(importances)[-16:]
axes[1].barh(range(16), importances[indices], color='green', alpha=0.7)
axes[1].set_yticks(range(16))
axes[1].set_yticklabels([f'Latent {i}' for i in indices])
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 16 Feature Importances')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'classifier_results.png', dpi=150)

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
Files saved to {OUTPUT_DIR}:
  - encoder.keras
  - decoder.keras
  - random_forest.joblib
  - training_history.png
  - latent_space.png
  - classifier_results.png

Performance Metrics:
  - Final VAE Loss: {history['loss'][-1]:.2f}
  - RF Accuracy: {accuracy:.4f}
  - RF AUC-ROC: {auc:.4f}

The model is ready for SNR sensitivity testing!
""")

# Cleanup
del true_train, false_train, true_eval, false_eval
gc.collect()
