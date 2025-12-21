"""
Quick test script to verify the SETI detector implementation.

This script runs a minimal training loop to verify all components work together.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf

print("=" * 50)
print("SETI Detector Implementation Test")
print("=" * 50)

# Test 1: Configuration loading
print("\n[1/6] Testing configuration loading...")
from src.utils.config import load_config
config = load_config()
print(f"  ✓ Config loaded: latent_dim={config.model.latent_dim}, alpha={config.model.alpha}")

# Test 2: Noise generation
print("\n[2/6] Testing noise generation...")
from src.data.noise_generator import NoiseGenerator
noise_gen = NoiseGenerator()
noise_frame = noise_gen.generate_frame(fchans=512, tchans=16)
print(f"  ✓ Generated noise frame: shape={noise_frame.shape}")

# Test 3: Signal injection
print("\n[3/6] Testing signal injection...")
from src.data.signal_generator import SignalGenerator
signal_gen = SignalGenerator(seed=42)
injected, info = signal_gen.inject_signal(noise_frame, snr=30)
print(f"  ✓ Injected signal: drift_rate={info['drift_rate']:.2f} Hz/s, snr={info['snr']:.1f}")

# Test 4: Cadence generation
print("\n[4/6] Testing cadence generation...")
from src.data.cadence_generator import CadenceGenerator, CadenceParams
params = CadenceParams(fchans=512, tchans=16)
cadence_gen = CadenceGenerator(params, seed=42)

true_sample = cadence_gen.create_true_sample_fast()
print(f"  ✓ True sample: shape={true_sample.shape}")

false_sample = cadence_gen.create_false_sample()
print(f"  ✓ False sample: shape={false_sample.shape}")

# Test 5: Model building
print("\n[5/6] Testing model building...")
from src.models.vae import build_vae

# Build model with standard dimensions (decoder outputs 512 width)
model = build_vae(
    input_shape=(16, 512, 1),  # Standard width matching decoder
    latent_dim=8,
    dense_units=64,  # Smaller for quick test
    alpha=1.0,
    beta=0.1
)
print(f"  ✓ VAE model built")

# Test 6: Forward pass
print("\n[6/6] Testing forward pass...")
# Create dummy data
batch_size = 2
dummy_input = np.random.rand(batch_size, 16, 512, 1).astype(np.float32)
dummy_true = np.random.rand(batch_size, 6, 16, 512, 1).astype(np.float32)
dummy_false = np.random.rand(batch_size, 6, 16, 512, 1).astype(np.float32)

# Forward pass
z_mean, z_log_var, z, reconstruction = model([dummy_input], training=False)
print(f"  ✓ Encoder output: z_mean={z_mean.shape}, z={z.shape}")
print(f"  ✓ Decoder output: reconstruction={reconstruction.shape}")

# Test training step
print("\n  Testing training step...")
loss = model.train_step(((dummy_input, dummy_true, dummy_false), dummy_input))
print(f"  ✓ Training step complete: loss={loss['loss']:.4f}")

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)

print("\nQuick summary of created files:")
print("  - src/data/noise_generator.py")
print("  - src/data/signal_generator.py") 
print("  - src/data/cadence_generator.py")
print("  - src/data/dataset.py")
print("  - src/models/sampling.py")
print("  - src/models/encoder.py")
print("  - src/models/decoder.py")
print("  - src/models/vae.py")
print("  - src/utils/config.py")
print("  - src/utils/preprocessing.py")
print("  - src/utils/visualization.py")
print("  - scripts/train_vae.py")
print("  - scripts/train_classifier.py")
print("  - configs/default.yaml")

print("\nTo start training:")
print("  python scripts/train_vae.py --config configs/default.yaml")
