"""
Large-Scale Training Pipeline (Paper Parameters).

This script replicates the original paper training approach:
- 20 training batches with fresh data each batch
- 150-100 epochs per batch
- 5000 samples per batch
- Uses model.fit() for native multi-GPU support

Estimated runtime: 6-10 hours on 2x RTX 4090
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

# ============================================
# GPU CONFIGURATION
# ============================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU config error: {e}")
else:
    print("⚠ No GPU found - training will be slow!")

print("=" * 70)
print("SETI Signal Detector - Large-Scale Training (Paper Parameters)")
print("=" * 70)

# ============================================
# CONFIGURATION (Memory-optimized: 40 batches × 2500 samples)
# ============================================
NUM_BATCHES = 20                  # Doubled to maintain total samples
NUM_SAMPLES_TRAIN = 2500          # Half of original (memory-friendly)
NUM_SAMPLES_VAL = 500             # Reduced proportionally
BATCH_SIZE = 1000                 # Unchanged (15 updates/epoch)
VALIDATION_BATCH_SIZE = 3000      # 500 × 6 = 3000

# Epochs per batch (extended pattern for 40 batches, with early stopping)
"""EPOCHS_PER_BATCH = [150, 150, 150, 150, 150, 150, 150, 100, 100, 100,
                   100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                   100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                   100, 100, 100, 100, 100, 100, 100, 100, 100, 100]"""

EPOCHS_PER_BATCH = [150, 150, 150, 150, 150, 150, 150, 100, 100, 100,
                   100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

# SNR settings (paper: snr_base=10, snr_range=40)
SNR_BASE = 10
SNR_RANGE = 40

# Model hyperparameters
LEARNING_RATE = 0.001
LATENT_DIM = 8
DENSE_UNITS = 512
ALPHA = 10
BETA = 1.5  # Paper: beta=1.5

# Early Stopping configuration
EARLY_STOPPING_PATIENCE = 15      # Stop if no improvement for 15 epochs
EARLY_STOPPING_MIN_DELTA = 10.0   # Minimum change to qualify as improvement

OUTPUT_DIR = Path("checkpoints/large_scale_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration (Paper Parameters):")
print(f"  Training batches: {NUM_BATCHES}")
print(f"  Samples per batch: {NUM_SAMPLES_TRAIN}")
print(f"  Epochs pattern: {EPOCHS_PER_BATCH[0]}-{EPOCHS_PER_BATCH[-1]}")
print(f"  Total epochs: {sum(EPOCHS_PER_BATCH)}")
print(f"  SNR range: {SNR_BASE} to {SNR_BASE + SNR_RANGE}")
print(f"  beta: {BETA}")
print(f"  Output: {OUTPUT_DIR}")

# ============================================
# IMPORTS
# ============================================
from src.data.cadence_generator import CadenceGenerator, CadenceParams
from src.models.vae import build_vae
from src.utils.preprocessing import preprocess, downscale, combine_cadences, recombine_latents

# ============================================
# DISTRIBUTION STRATEGY
# ============================================
print("\n" + "=" * 70)
print("SETTING UP MULTI-GPU TRAINING")
print("=" * 70)

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"✓ Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
else:
    strategy = tf.distribute.get_strategy()
    print("Using single GPU/CPU")

# Effective batch size with multi-GPU
effective_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
print(f"  Per-GPU batch size: {BATCH_SIZE}")
print(f"  Effective batch size: {effective_batch_size}")

# ============================================
# BUILD MODEL (inside strategy scope)
# ============================================
print("\nBuilding model...")

with strategy.scope():
    model = build_vae(
        input_shape=(16, 512, 1),
        latent_dim=LATENT_DIM,
        dense_units=DENSE_UNITS,
        alpha=ALPHA,
        beta=BETA,
        learning_rate=LEARNING_RATE
    )

# ============================================
# HELPER FUNCTIONS
# ============================================
from multiprocessing import Pool, cpu_count
import os

# Number of parallel workers (reduced to avoid memory issues)
NUM_WORKERS = 46

def _generate_true_sample(args):
    """Worker function for generating a true sample."""
    seed, snr_base, snr_range = args
    params = CadenceParams(fchans=4096, tchans=16, snr_base=snr_base, snr_range=snr_range)
    cadence_gen = CadenceGenerator(params, seed=seed)
    return cadence_gen.create_true_sample_fast()

def _generate_false_sample(args):
    """Worker function for generating a false sample."""
    seed, snr_base, snr_range = args
    params = CadenceParams(fchans=4096, tchans=16, snr_base=snr_base, snr_range=snr_range)
    cadence_gen = CadenceGenerator(params, seed=seed)
    return cadence_gen.create_false_sample()

def generate_training_data(num_samples, seed=None):
    """Generate training data with multiprocessing.
    
    Original code format:
    - data (vae): (N*6, 16, 512, 1) - flattened from N cadences
    - true_data: (N*6, 6, 16, 512, 1) - N*6 complete cadences (for clustering loss)
    - false_data: (N*6, 6, 16, 512, 1) - N*6 complete cadences (for clustering loss)
    
    This ensures all have same first dimension for model.fit().
    """
    base_seed = seed if seed is not None else 0
    
    print(f"  Using {NUM_WORKERS} CPU workers for parallel generation...")
    
    # Prepare arguments with unique seeds
    vae_args = [(base_seed + i, SNR_BASE, SNR_RANGE) for i in range(num_samples)]
    true_args = [(base_seed + 10000 + i, SNR_BASE, SNR_RANGE) for i in range(num_samples * 6)]
    false_args = [(base_seed + 100000 + i, SNR_BASE, SNR_RANGE) for i in range(num_samples * 6)]
    
    # Generate in parallel
    with Pool(NUM_WORKERS) as pool:
        print(f"  Generating {num_samples} VAE samples...")
        vae_cadences = list(tqdm(pool.imap(_generate_true_sample, vae_args), 
                                  total=num_samples, desc="VAE", leave=False))
        
        print(f"  Generating {num_samples * 6} True samples...")
        true_cadences = list(tqdm(pool.imap(_generate_true_sample, true_args), 
                                   total=num_samples * 6, desc="True", leave=False))
        
        print(f"  Generating {num_samples * 6} False samples...")
        false_cadences = list(tqdm(pool.imap(_generate_false_sample, false_args), 
                                    total=num_samples * 6, desc="False", leave=False))
    
    vae_cadences = np.array(vae_cadences)
    true_cadences = np.array(true_cadences)
    false_cadences = np.array(false_cadences)
    
    # Preprocess all
    vae_cadences = downscale(vae_cadences, factor=8)
    true_cadences = downscale(true_cadences, factor=8)
    false_cadences = downscale(false_cadences, factor=8)
    
    vae_cadences = preprocess(vae_cadences, add_channel=True)
    true_cadences = preprocess(true_cadences, add_channel=True)
    false_cadences = preprocess(false_cadences, add_channel=True)
    
    # vae_data: flatten (N, 6, H, W, C) -> (N*6, H, W, C)
    vae_data = combine_cadences(vae_cadences)  # (N*6, 16, 512, 1)
    
    # true_data, false_data: keep as cadences (N*6, 6, 16, 512, 1)
    # for clustering loss which needs the 6 observations structure
    true_data = true_cadences   # (N*6, 6, 16, 512, 1)
    false_data = false_cadences # (N*6, 6, 16, 512, 1)
    
    print(f"  Data shapes: vae={vae_data.shape}, true={true_data.shape}, false={false_data.shape}")
    
    return vae_data, true_data, false_data


def plot_batch_history(history, batch_idx, output_dir):
    """Plot training history for a single batch."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Training History - Batch {batch_idx + 1}', fontsize=14)
    
    # Total loss
    axes[0, 0].plot(history.history['loss'], 'b-', label='Train')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], 'b--', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    if 'reconstruction_loss' in history.history:
        axes[0, 1].plot(history.history['reconstruction_loss'], 'g-', label='Train')
        if 'val_reconstruction_loss' in history.history:
            axes[0, 1].plot(history.history['val_reconstruction_loss'], 'g--', label='Val')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Clustering losses
    if 'true_loss' in history.history:
        axes[1, 0].plot(history.history['true_loss'], 'c-', label='True')
        axes[1, 0].plot(history.history['false_loss'], 'm-', label='False')
        axes[1, 0].set_title('Clustering Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # KL loss
    if 'kl_loss' in history.history:
        axes[1, 1].plot(history.history['kl_loss'], 'r-', label='Train')
        if 'val_kl_loss' in history.history:
            axes[1, 1].plot(history.history['val_kl_loss'], 'r--', label='Val')
        axes[1, 1].set_title('KL Divergence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_history_batch_{batch_idx + 1}.png', dpi=150)
    plt.close()


# ============================================
# TRAINING LOOP (Paper approach)
# ============================================
print("\n" + "=" * 70)
print("STARTING TRAINING (Paper Approach)")
print("=" * 70)

all_histories = []
total_start_time = time.time()

for batch_idx in range(NUM_BATCHES):
    print(f"\n{'='*70}")
    print(f"BATCH {batch_idx + 1}/{NUM_BATCHES}")
    print(f"{'='*70}")
    
    batch_start_time = time.time()
    epochs_this_batch = EPOCHS_PER_BATCH[batch_idx]
    
    # Generate fresh training data
    print(f"\nGenerating training data (seed={batch_idx * 1000})...")
    vae_train, true_train, false_train = generate_training_data(
        NUM_SAMPLES_TRAIN, 
        seed=batch_idx * 1000
    )
    print(f"  VAE data: {vae_train.shape}")
    print(f"  True data: {true_train.shape}")
    print(f"  False data: {false_train.shape}")
    
    # Generate fresh validation data
    print(f"Generating validation data...")
    vae_val, true_val, false_val = generate_training_data(
        NUM_SAMPLES_VAL, 
        seed=batch_idx * 1000 + 500
    )
    
    # Prepare data in format expected by model.fit()
    x_train = [vae_train, true_train, false_train]
    y_train = vae_train
    
    x_val = [vae_val, true_val, false_val]
    y_val = vae_val
    
    # Train with Early Stopping
    print(f"\nTraining for up to {epochs_this_batch} epochs (early stopping enabled)...")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs_this_batch,
        batch_size=effective_batch_size,
        validation_data=(x_val, y_val),
        validation_batch_size=VALIDATION_BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Report actual epochs trained
    actual_epochs = len(history.history['loss'])
    if actual_epochs < epochs_this_batch:
        print(f"  Early stopped at epoch {actual_epochs}/{epochs_this_batch}")
    
    all_histories.append(history)
    
    # Save checkpoint
    model.encoder.save(str(OUTPUT_DIR / f'encoder_batch_{batch_idx + 1}.keras'))
    model.decoder.save(str(OUTPUT_DIR / f'decoder_batch_{batch_idx + 1}.keras'))
    
    # Plot batch history
    plot_batch_history(history, batch_idx, OUTPUT_DIR)
    
    batch_time = time.time() - batch_start_time
    print(f"\nBatch {batch_idx + 1} completed in {batch_time/60:.1f} minutes")
    print(f"  Final loss: {history.history['loss'][-1]:.2f}")
    
    # Cleanup
    del vae_train, true_train, false_train
    del vae_val, true_val, false_val
    del x_train, y_train, x_val, y_val
    gc.collect()

# Save final model
model.encoder.save(str(OUTPUT_DIR / 'encoder_final.keras'))
model.decoder.save(str(OUTPUT_DIR / 'decoder_final.keras'))

total_time = time.time() - total_start_time
print(f"\n{'='*70}")
print(f"VAE TRAINING COMPLETE")
print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
print(f"{'='*70}")

# ============================================
# EVALUATION ON FINAL DATA
# ============================================
print("\n" + "=" * 70)
print("PHASE 2: FINAL EVALUATION")
print("=" * 70)

# Generate evaluation data
print("\nGenerating evaluation data...")
NUM_EVAL_SAMPLES = 2000
vae_eval, true_eval, false_eval = generate_training_data(NUM_EVAL_SAMPLES, seed=99999)

# Extract latents
print("Extracting latent representations...")
true_flat = combine_cadences(true_eval).astype(np.float32)
false_flat = combine_cadences(false_eval).astype(np.float32)

true_latents = model.encoder.predict(true_flat, batch_size=500)[2]
false_latents = model.encoder.predict(false_flat, batch_size=500)[2]

# Recombine for cadence-level
true_latents_cadence = recombine_latents(true_latents)
false_latents_cadence = recombine_latents(false_latents)

print(f"  True cadence latents: {true_latents_cadence.shape}")
print(f"  False cadence latents: {false_latents_cadence.shape}")

# Plot latent space
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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

plt.suptitle('Latent Space Visualization (Final Model)', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'latent_space_final.png', dpi=150)
print("Latent space plot saved")

# ============================================
# RANDOM FOREST CLASSIFIER
# ============================================
print("\n" + "=" * 70)
print("PHASE 3: RANDOM FOREST CLASSIFIER")
print("=" * 70)

X = np.vstack([true_latents_cadence, false_latents_cadence])
y = np.concatenate([np.ones(len(true_latents_cadence)), np.zeros(len(false_latents_cadence))])

if not np.isfinite(X).all():
    print("Warning: Found NaN/Inf, replacing...")
    X = np.nan_to_num(X, nan=0, posinf=100, neginf=-100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nRandom Forest training...")
print(f"  Train: {X_train.shape} - Test: {X_test.shape}")

rf = RandomForestClassifier(
    n_estimators=1000,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n--- Random Forest Results ---")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  AUC-ROC:  {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['False (RFI)', 'True (ETI)']))

joblib.dump(rf, OUTPUT_DIR / 'random_forest.joblib')
print(f"Random Forest saved")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("LARGE-SCALE TRAINING COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
Files saved to {OUTPUT_DIR}:
  - encoder_final.keras
  - decoder_final.keras
  - random_forest.joblib
  - encoder_batch_*.keras (checkpoints)
  - training_history_batch_*.png (per-batch plots)
  - latent_space_final.png

Training Parameters (Paper-aligned):
  - Batches: {NUM_BATCHES}
  - Samples per batch: {NUM_SAMPLES_TRAIN}
  - Total epochs: {sum(EPOCHS_PER_BATCH)}
  - Total samples seen: {NUM_BATCHES * NUM_SAMPLES_TRAIN}
  - beta: {BETA}
  - alpha: {ALPHA}
  - SNR range: {SNR_BASE}-{SNR_BASE + SNR_RANGE}
  - GPUs: {strategy.num_replicas_in_sync}

Performance Metrics:
  - RF Accuracy: {accuracy:.4f}
  - RF AUC-ROC: {auc:.4f}

Total Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)
""")

gc.collect()
