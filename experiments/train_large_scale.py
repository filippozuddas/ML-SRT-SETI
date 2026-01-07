"""
Large-Scale Training Pipeline.

This script replicates the original paper training approach:
- Multiple training batches with fresh data each batch
- Epochs per batch with early stopping
- Uses model.fit() for native multi-GPU support
- Supports real SRT backgrounds or synthetic noise

Usage:
    python scripts/train_large_scale.py --plate data/srt_training/srt_backgrounds.npz
    python scripts/train_large_scale.py --no-plate  # Use synthetic noise
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-Scale Training Pipeline for SETI Signal Detection"
    )
    
    # Data configuration
    parser.add_argument('--plate', '-p', type=str, default=None,
                        help='Path to SRT plate file (.npz). If not provided, uses synthetic noise.')
    parser.add_argument('--no-plate', action='store_true',
                        help='Use synthetic Gaussian noise instead of real backgrounds')
    
    # Training configuration
    parser.add_argument('--batches', '-b', type=int, default=40,
                        help='Number of training batches (default: 40)')
    parser.add_argument('--samples', '-n', type=int, default=2500,
                        help='Samples per batch (default: 2500)')
    parser.add_argument('--val-samples', type=int, default=500,
                        help='Validation samples per batch (default: 500)')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='Max epochs per batch (default: 100)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Training batch size (default: 1000)')
    
    # Model hyperparameters
    parser.add_argument('--snr-base', type=int, default=10,
                        help='Base SNR for signal injection (default: 10)')
    parser.add_argument('--snr-range', type=int, default=40,
                        help='SNR range (default: 40)')
    parser.add_argument('--beta', type=float, default=1.5,
                        help='Beta parameter for VAE (default: 1.5)')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Alpha parameter for clustering loss (default: 10)')
    parser.add_argument('--latent-dim', type=int, default=8,
                        help='Latent dimension (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    
    # Output
    parser.add_argument('--output', '-o', type=str, default='/content/filippo/ML-SRT-SETI/results/real_obs_training',
                        help='Output directory for checkpoints')
    
    # Resume training
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='Path to encoder checkpoint to resume from (e.g., encoder_batch_7.keras)')
    parser.add_argument('--start-batch', type=int, default=0,
                        help='Batch number to start from (0-indexed, default: 0)')
    
    # Parallelization
    parser.add_argument('--workers', '-w', type=int, default=46,
                        help='Number of CPU workers for data generation (default: 46)')
    
    return parser.parse_args()


# Parse arguments
args = parse_args()

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
print("SETI Signal Detector - Large-Scale Training")
print("=" * 70)

# ============================================
# CONFIGURATION FROM ARGS
# ============================================
NUM_BATCHES = args.batches
NUM_SAMPLES_TRAIN = args.samples
NUM_SAMPLES_VAL = args.val_samples
BATCH_SIZE = args.batch_size
VALIDATION_BATCH_SIZE = args.val_samples * 6

# Epochs per batch
EPOCHS_PER_BATCH = [min(150, args.epochs)] * 7 + [args.epochs] * (NUM_BATCHES - 7)
if len(EPOCHS_PER_BATCH) < NUM_BATCHES:
    EPOCHS_PER_BATCH.extend([args.epochs] * (NUM_BATCHES - len(EPOCHS_PER_BATCH)))

# SNR settings
SNR_BASE = args.snr_base
SNR_RANGE = args.snr_range

# Model hyperparameters
LEARNING_RATE = args.lr
LATENT_DIM = args.latent_dim
DENSE_UNITS = 512
ALPHA = args.alpha
BETA = args.beta

# Early Stopping
EARLY_STOPPING_PATIENCE = args.patience
EARLY_STOPPING_MIN_DELTA = 3

# Plate configuration
USE_SRT_PLATE = args.plate is not None and not args.no_plate
SRT_PLATE_PATH = args.plate

# Output directory
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Workers
NUM_WORKERS = args.workers

print(f"\nConfiguration:")
print(f"  Training batches: {NUM_BATCHES}")
print(f"  Samples per batch: {NUM_SAMPLES_TRAIN}")
print(f"  Epochs per batch: {EPOCHS_PER_BATCH[0]}-{EPOCHS_PER_BATCH[-1]}")
print(f"  Total epochs: {sum(EPOCHS_PER_BATCH)}")
print(f"  SNR range: {SNR_BASE} to {SNR_BASE + SNR_RANGE}")
print(f"  beta: {BETA}, alpha: {ALPHA}")
print(f"  Use SRT Plate: {USE_SRT_PLATE}")
if USE_SRT_PLATE:
    print(f"  SRT Plate Path: {SRT_PLATE_PATH}")
print(f"  Workers: {NUM_WORKERS}")
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
# RESUME FROM CHECKPOINT (if provided)
# ============================================
if args.resume:
    print(f"\n{'='*70}")
    print("RESUMING FROM CHECKPOINT")
    print(f"{'='*70}")
    resume_path = Path(args.resume)
    if resume_path.exists():
        print(f"  Loading encoder from: {resume_path}")
        model.encoder.load_weights(str(resume_path))
        
        # Try to load matching decoder
        decoder_path = str(resume_path).replace('encoder_', 'decoder_')
        if Path(decoder_path).exists():
            print(f"  Loading decoder from: {decoder_path}")
            model.decoder.load_weights(decoder_path)
        else:
            print(f"  ⚠ No matching decoder found at {decoder_path}")
        
        print(f"  ✓ Checkpoint loaded successfully!")
        print(f"  Will start from batch {args.start_batch + 1}")
    else:
        print(f"  ❌ ERROR: Checkpoint not found: {resume_path}")
        print(f"  Starting fresh training instead...")

# Global best tracking
GLOBAL_BEST_VAL_LOSS = float('inf')
GLOBAL_BEST_BATCH = -1

# ============================================
# HELPER FUNCTIONS
# ============================================
from multiprocessing import Pool, cpu_count
import os

# Load SRT plate if configured
SRT_PLATE = None
SRT_PLATE_PATH_FOR_WORKERS = None  # Path for workers to load

if USE_SRT_PLATE:
    print(f"\nLoading SRT plate from {SRT_PLATE_PATH}...")
    try:
        plate_data = np.load(SRT_PLATE_PATH)
        SRT_PLATE = plate_data['backgrounds']
        SRT_PLATE_PATH_FOR_WORKERS = SRT_PLATE_PATH
        print(f"  Loaded plate shape: {SRT_PLATE.shape}")
        print(f"  Using {len(SRT_PLATE)} real SRT backgrounds for signal injection")
    except FileNotFoundError:
        print(f"  WARNING: SRT plate not found at {SRT_PLATE_PATH}")
        print(f"  Falling back to synthetic Gaussian noise")
        SRT_PLATE = None

# Global variable for workers (set by initializer)
_WORKER_PLATE = None
_WORKER_PLATE_PATH = None

def _init_worker(plate_path):
    """Initialize worker with plate loaded from file."""
    global _WORKER_PLATE, _WORKER_PLATE_PATH
    if plate_path is not None and plate_path != _WORKER_PLATE_PATH:
        _WORKER_PLATE = np.load(plate_path)['backgrounds']
        _WORKER_PLATE_PATH = plate_path

def _generate_true_sample(args):
    """Worker function for generating a true sample."""
    seed, snr_base, snr_range = args
    params = CadenceParams(fchans=4096, tchans=16, snr_base=snr_base, snr_range=snr_range)
    cadence_gen = CadenceGenerator(params, plate=_WORKER_PLATE, seed=seed)
    return cadence_gen.create_true_sample_fast()

def _generate_false_sample(args):
    """Worker function for generating a false sample."""
    seed, snr_base, snr_range = args
    params = CadenceParams(fchans=4096, tchans=16, snr_base=snr_base, snr_range=snr_range)
    cadence_gen = CadenceGenerator(params, plate=_WORKER_PLATE, seed=seed)
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
    
    # Prepare arguments with unique seeds (NO plate - loaded by workers)
    vae_args = [(base_seed + i, SNR_BASE, SNR_RANGE) for i in range(num_samples)]
    true_args = [(base_seed + 10000 + i, SNR_BASE, SNR_RANGE) for i in range(num_samples * 6)]
    false_args = [(base_seed + 100000 + i, SNR_BASE, SNR_RANGE) for i in range(num_samples * 6)]
    
    # Generate in parallel with initializer
    with Pool(NUM_WORKERS, initializer=_init_worker, initargs=(SRT_PLATE_PATH_FOR_WORKERS,)) as pool:
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

# Start batch (for resume)
START_BATCH = args.start_batch

for batch_idx in range(START_BATCH, NUM_BATCHES):
    print(f"\n{'='*70}")
    print(f"BATCH {batch_idx + 1}/{NUM_BATCHES}")
    if batch_idx > START_BATCH or args.resume:
        print(f"  (Global Best: batch {GLOBAL_BEST_BATCH + 1}, val_loss={GLOBAL_BEST_VAL_LOSS:.2f})")
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
    
    # Get best validation loss from this batch
    batch_best_val_loss = min(history.history['val_loss'])
    
    # Check for catastrophic degradation (>5% increase from global best)
    CATASTROPHIC_THRESHOLD = 1.05  # 5% tolerance
    if GLOBAL_BEST_VAL_LOSS < float('inf'):
        if batch_best_val_loss > GLOBAL_BEST_VAL_LOSS * CATASTROPHIC_THRESHOLD:
            print(f"\n  ⚠️  CATASTROPHIC DEGRADATION DETECTED!")
            print(f"      Batch best: {batch_best_val_loss:.2f} vs Global best: {GLOBAL_BEST_VAL_LOSS:.2f}")
            print(f"      Rolling back to global best (batch {GLOBAL_BEST_BATCH + 1})...")
            
            # Reload global best weights
            global_best_encoder = OUTPUT_DIR / 'encoder_global_best.keras'
            global_best_decoder = OUTPUT_DIR / 'decoder_global_best.keras'
            if global_best_encoder.exists():
                model.encoder.load_weights(str(global_best_encoder))
                model.decoder.load_weights(str(global_best_decoder))
                print(f"      ✓ Rolled back to global best checkpoint")
            else:
                print(f"      ❌ No global best checkpoint found - continuing with degraded weights")
        else:
            print(f"  ✓ Batch acceptable (val_loss={batch_best_val_loss:.2f})")
    
    # Update global best if this batch is better
    if batch_best_val_loss < GLOBAL_BEST_VAL_LOSS:
        GLOBAL_BEST_VAL_LOSS = batch_best_val_loss
        GLOBAL_BEST_BATCH = batch_idx
        print(f"  🏆 NEW GLOBAL BEST! val_loss={batch_best_val_loss:.2f}")
        
        # Save global best checkpoint
        model.encoder.save(str(OUTPUT_DIR / 'encoder_global_best.keras'))
        model.decoder.save(str(OUTPUT_DIR / 'decoder_global_best.keras'))
    
    all_histories.append(history)
    
    # Save per-batch checkpoint
    model.encoder.save(str(OUTPUT_DIR / f'encoder_batch_{batch_idx + 1}.keras'))
    model.decoder.save(str(OUTPUT_DIR / f'decoder_batch_{batch_idx + 1}.keras'))
    
    # Plot batch history
    plot_batch_history(history, batch_idx, OUTPUT_DIR)
    
    batch_time = time.time() - batch_start_time
    print(f"\nBatch {batch_idx + 1} completed in {batch_time/60:.1f} minutes")
    print(f"  Final loss: {history.history['loss'][-1]:.2f}")
    
    # Cleanup - aggressive memory management to prevent RAM saturation
    del vae_train, true_train, false_train
    del vae_val, true_val, false_val
    del x_train, y_train, x_val, y_val
    del history
    #gc.collect()
    
    # Clear TensorFlow internal caches
    """tf.keras.backend.clear_session()
    
    # Rebuild model after clear_session (keeps weights via global best checkpoint)
    if (OUTPUT_DIR / 'encoder_global_best.keras').exists():
        with strategy.scope():
            model = build_vae(
                input_shape=(16, 512, 1),
                latent_dim=LATENT_DIM,
                dense_units=DENSE_UNITS,
                alpha=ALPHA,
                beta=BETA,
                learning_rate=LEARNING_RATE
            )
            model.encoder.load_weights(str(OUTPUT_DIR / 'encoder_global_best.keras'))
            model.decoder.load_weights(str(OUTPUT_DIR / 'decoder_global_best.keras'))
    """
    gc.collect()

# Save final model (use global best if available)
global_best_encoder = OUTPUT_DIR / 'encoder_global_best.keras'
if global_best_encoder.exists():
    print(f"\nUsing global best checkpoint (batch {GLOBAL_BEST_BATCH + 1}) as final model...")
    model.encoder.load_weights(str(global_best_encoder))
    model.decoder.load_weights(str(OUTPUT_DIR / 'decoder_global_best.keras'))

model.encoder.save(str(OUTPUT_DIR / 'encoder_final.keras'))
model.decoder.save(str(OUTPUT_DIR / 'decoder_final.keras'))

total_time = time.time() - total_start_time
print(f"\n{'='*70}")
print(f"VAE TRAINING COMPLETE")
print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
print(f"Global best: batch {GLOBAL_BEST_BATCH + 1} with val_loss={GLOBAL_BEST_VAL_LOSS:.2f}")
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
