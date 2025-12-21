"""
Random Forest Classifier Training Script.

Trains the Random Forest classifier on VAE latent representations.
"""

import argparse
import os
import sys
from pathlib import Path
import time
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import load_model

from src.utils.config import load_config, setup_gpu
from src.data.dataset import SETIDataset
from src.models.sampling import Sampling
from src.utils.preprocessing import combine_cadences, recombine_latents


def extract_latents(encoder, data, batch_size=5000):
    """
    Extract latent representations from data using the encoder.
    
    Args:
        encoder: Trained encoder model
        data: Cadence data of shape (batch, 6, height, width, 1)
        batch_size: Batch size for inference
        
    Returns:
        Concatenated latent vectors of shape (batch, latent_dim * 6)
    """
    # Flatten cadences for encoder: (batch, 6, H, W, 1) -> (batch*6, H, W, 1)
    flat_data = combine_cadences(data)
    
    # Get latent representations
    latents = encoder.predict(flat_data, batch_size=batch_size)[2]  # [2] is z
    
    # Recombine to (batch, latent_dim * 6)
    return recombine_latents(latents)


def main():
    parser = argparse.ArgumentParser(description='Train Random Forest Classifier')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--encoder', type=str, required=True,
                        help='Path to trained encoder model')
    parser.add_argument('--output', type=str, default='checkpoints/random_forest.joblib',
                        help='Output path for classifier')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples per class (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Setup GPU (minimal for inference)
    print("Setting up GPU...")
    setup_gpu(config)
    
    # Load encoder model
    print(f"Loading encoder: {args.encoder}")
    encoder = load_model(args.encoder, custom_objects={'Sampling': Sampling})
    encoder.summary()
    
    # Number of samples
    num_samples = args.num_samples or config.classifier.num_samples
    print(f"Generating {num_samples} samples per class...")
    
    # Initialize data generator
    dataset_gen = SETIDataset(plate=None, seed=42)
    
    # Generate TRUE samples
    print("Generating TRUE samples...")
    start = time.time()
    true_data = dataset_gen.generate_training_data(
        num_samples=num_samples,
        snr_base=config.signal.snr_base,
        snr_range=config.signal.snr_range
    )
    true_cadences = true_data[1]  # Get cadence data (not flattened)
    print(f"Time: {time.time() - start:.1f}s, Shape: {true_cadences.shape}")
    
    # Generate FALSE samples
    print("Generating FALSE samples...")
    start = time.time()
    false_data = dataset_gen.generate_training_data(
        num_samples=num_samples,
        snr_base=config.signal.snr_base,
        snr_range=config.signal.snr_range
    )
    false_cadences = false_data[2]  # Get false cadences
    print(f"Time: {time.time() - start:.1f}s, Shape: {false_cadences.shape}")
    
    del true_data, false_data
    gc.collect()
    
    # Extract latent representations
    print("\nExtracting TRUE latents...")
    true_latents = extract_latents(encoder, true_cadences)
    print(f"True latents shape: {true_latents.shape}")
    
    print("Extracting FALSE latents...")
    false_latents = extract_latents(encoder, false_cadences)
    print(f"False latents shape: {false_latents.shape}")
    
    del true_cadences, false_cadences
    gc.collect()
    
    # Prepare training data
    print("\nPreparing training data...")
    X_train = np.concatenate([true_latents, false_latents], axis=0)
    y_train = np.concatenate([
        np.ones(true_latents.shape[0]),   # 1 = True (ETI signal)
        np.zeros(false_latents.shape[0])  # 0 = False (RFI/noise)
    ])
    
    # Shuffle
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    print(f"Training data shape: {X_train.shape}")
    print(f"Class distribution: True={np.sum(y_train == 1)}, False={np.sum(y_train == 0)}")
    
    # Check for NaN/Inf
    if not np.isfinite(X_train).all():
        print("WARNING: Found NaN/Inf in latents, replacing...")
        X_train = np.nan_to_num(X_train, nan=0, posinf=100, neginf=-100)
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    print(f"  n_estimators: {config.classifier.n_estimators}")
    print(f"  max_features: {config.classifier.max_features}")
    print(f"  n_jobs: {config.classifier.n_jobs}")
    
    start = time.time()
    clf = RandomForestClassifier(
        n_estimators=config.classifier.n_estimators,
        max_features=config.classifier.max_features,
        bootstrap=config.classifier.bootstrap,
        n_jobs=config.classifier.n_jobs,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"Training time: {time.time() - start:.1f}s")
    
    # Evaluate on training data
    print("\nTraining set performance:")
    y_pred = clf.predict(X_train)
    print(f"Accuracy: {accuracy_score(y_train, y_pred):.4f}")
    print(classification_report(y_train, y_pred, target_names=['False (RFI)', 'True (ETI)']))
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, str(output_path))
    print(f"\nSaved classifier: {output_path}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
