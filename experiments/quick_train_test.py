"""
Quick VAE Training Test.

Simplified training script to verify the VAE works correctly.
Uses a manual training loop instead of model.fit() for better control.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from tqdm import tqdm

print("=" * 60)
print("SETI VAE Quick Training Test")
print("=" * 60)

# Import modules
from src.utils.config import load_config
from src.data.cadence_generator import CadenceGenerator, CadenceParams
from src.models.vae import build_vae
from src.utils.preprocessing import preprocess, downscale, combine_cadences

# Configuration
NUM_SAMPLES = 50           # Small for quick test
BATCH_SIZE = 10
EPOCHS = 3
LEARNING_RATE = 0.001

print(f"\nConfiguration:")
print(f"  Samples: {NUM_SAMPLES}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")

# Initialize generators
print("\n[1/4] Generating training data...")
params = CadenceParams(fchans=4096, tchans=16, snr_base=30, snr_range=30)
cadence_gen = CadenceGenerator(params, seed=42)

# Generate samples
true_samples = []
false_samples = []

for i in tqdm(range(NUM_SAMPLES), desc="Generating samples"):
    true_samples.append(cadence_gen.create_true_sample_fast())
    false_samples.append(cadence_gen.create_false_sample())

true_data = np.array(true_samples)
false_data = np.array(false_samples)

print(f"  True data shape: {true_data.shape}")
print(f"  False data shape: {false_data.shape}")

# Preprocess: downscale and normalize
print("\n[2/4] Preprocessing data...")
true_data = downscale(true_data, factor=8)
false_data = downscale(false_data, factor=8)

true_data = preprocess(true_data, add_channel=True)
false_data = preprocess(false_data, add_channel=True)

print(f"  True data (preprocessed): {true_data.shape}")
print(f"  False data (preprocessed): {false_data.shape}")

# Flatten for VAE input
vae_data = combine_cadences(true_data)
print(f"  VAE input data: {vae_data.shape}")

# Build model
print("\n[3/4] Building VAE model...")
model = build_vae(
    input_shape=(16, 512, 1),
    latent_dim=8,
    dense_units=512,
    alpha=10,
    beta=0.5,
    learning_rate=LEARNING_RATE
)

# Manual training loop
print("\n[4/4] Training...")
print("-" * 60)

num_batches = NUM_SAMPLES // BATCH_SIZE
histories = {'loss': [], 'reconstruction_loss': [], 'kl_loss': [], 'true_loss': [], 'false_loss': []}

for epoch in range(EPOCHS):
    epoch_losses = []
    
    # Shuffle data each epoch
    indices = np.random.permutation(NUM_SAMPLES)
    
    for batch_idx in range(num_batches):
        # Get batch indices
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch_indices = indices[start_idx:end_idx]
        
        # Prepare batch - need to properly index the flattened vae_data
        # vae_data is (NUM_SAMPLES * 6, 16, 512, 1)
        vae_indices = []
        for bi in batch_indices:
            vae_indices.extend(range(bi * 6, (bi + 1) * 6))
        
        vae_batch = vae_data[vae_indices].astype(np.float32)
        true_batch = true_data[batch_indices].astype(np.float32)
        false_batch = false_data[batch_indices].astype(np.float32)
        
        # Run training step
        loss_dict = model.train_step(((vae_batch, true_batch, false_batch), vae_batch))
        
        epoch_losses.append(float(loss_dict['loss']))
    
    # Compute epoch means
    mean_loss = np.mean(epoch_losses)
    histories['loss'].append(mean_loss)
    
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {mean_loss:.4f} - "
          f"recon: {float(loss_dict['reconstruction_loss']):.4f} - "
          f"kl: {float(loss_dict['kl_loss']):.4f} - "
          f"true: {float(loss_dict['true_loss']):.4f} - "
          f"false: {float(loss_dict['false_loss']):.4f}")

print("-" * 60)

# Test forward pass
print("\n[5/5] Testing forward pass...")
test_input = vae_data[:6].astype(np.float32)
z_mean, z_log_var, z, reconstruction = model(test_input, training=False)
print(f"  Input shape: {test_input.shape}")
print(f"  Latent z shape: {z.shape}")
print(f"  Reconstruction shape: {reconstruction.shape}")

# Save model
print("\nSaving test model...")
model.encoder.save('checkpoints/test_encoder.h5')
model.decoder.save('checkpoints/test_decoder.h5')

print("\n" + "=" * 60)
print("✓ Training test completed successfully!")
print("=" * 60)

# Plot loss curve
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    plt.plot(histories['loss'], 'b-o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Test Run)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('checkpoints/test_loss.png', dpi=100)
    print(f"Loss plot saved to checkpoints/test_loss.png")
except Exception as e:
    print(f"Could not save plot: {e}")
