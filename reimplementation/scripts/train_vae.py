"""
VAE Training Script.

Main entry point for training the Contrastive β-VAE model.
"""

import argparse
import os
import sys
import gc
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils.config import load_config, setup_gpu, get_strategy
from src.data.dataset import SETIDataset, create_tf_dataset
from src.models.vae import build_vae
from src.utils.visualization import plot_training_history


def train_cycle(model, dataset, epochs, batch_size, validation_data=None, validation_batch_size=None):
    """
    Run a single training cycle.
    
    Args:
        model: VAE model to train
        dataset: Training dataset tuple (vae_data, true_data, false_data)
        epochs: Number of epochs
        batch_size: Batch size
        validation_data: Optional validation dataset tuple
        validation_batch_size: Validation batch size
        
    Returns:
        Training history
    """
    vae_data, true_data, false_data = dataset
    
    # Prepare data as model expects
    x_train = [vae_data, true_data, false_data]
    y_train = vae_data
    
    if validation_data is not None:
        val_vae, val_true, val_false = validation_data
        x_val = [val_vae, val_true, val_false]
        y_val = val_vae
        validation = (x_val, y_val)
        val_batch_size = validation_batch_size or batch_size
    else:
        validation = None
    
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation,
        validation_batch_size=validation_batch_size
    )
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train SETI VAE Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--name', type=str, default=None,
                        help='Model name (default: auto-generated)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Setup GPU
    print("Setting up GPU...")
    setup_gpu(config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate model name
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"vae_latent{config.model.latent_dim}_{timestamp}"
    else:
        model_name = args.name
    print(f"Model name: {model_name}")
    
    # Get distribution strategy
    strategy = get_strategy(config)
    
    # Build model within strategy scope
    print("Building model...")
    with strategy.scope():
        if args.resume:
            from src.models.sampling import Sampling
            from tensorflow.keras.models import load_model
            encoder = load_model(args.resume, custom_objects={'Sampling': Sampling})
            # Need to rebuild full VAE from encoder
            from src.models.vae import ContrastiveVAE
            from src.models.decoder import build_decoder
            decoder = build_decoder(
                latent_dim=config.model.latent_dim,
                dense_units=config.model.dense_units
            )
            model = ContrastiveVAE(
                encoder, decoder,
                alpha=config.model.alpha,
                beta=config.model.beta,
                gamma=config.model.gamma
            )
            model.compile(optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.training.learning_rate
            ))
            print(f"Resumed from: {args.resume}")
        else:
            model = build_vae(
                input_shape=(config.data.frame_height, config.data.frame_width, 1),
                latent_dim=config.model.latent_dim,
                dense_units=config.model.dense_units,
                kernel_size=tuple(config.model.kernel_size),
                alpha=config.model.alpha,
                beta=config.model.beta,
                gamma=config.model.gamma,
                learning_rate=config.training.learning_rate,
                l1_weight=config.model.l1_weight,
                l2_weight=config.model.l2_weight
            )
    
    # Initialize dataset generator
    print("Initializing data generator...")
    dataset_gen = SETIDataset(
        plate=None,  # Use synthetic noise for Phase 1
        seed=42
    )
    
    # Training loop
    all_history = {
        'loss': [], 'reconstruction_loss': [], 'kl_loss': [],
        'true_loss': [], 'false_loss': [],
        'val_loss': [], 'val_reconstruction_loss': [], 'val_kl_loss': [],
        'val_true_loss': [], 'val_false_loss': []
    }
    
    print(f"\nStarting training: {config.training.num_cycles} cycles")
    print(f"Samples per cycle: {config.training.num_samples_train}")
    print(f"Epochs per cycle: {config.training.epochs_per_cycle}")
    print("=" * 50)
    
    for cycle in range(config.training.num_cycles):
        print(f"\n=== Cycle {cycle + 1}/{config.training.num_cycles} ===")
        start_time = time.time()
        
        # Generate new training data each cycle
        print("Generating training data...")
        train_data = dataset_gen.generate_training_data(
            num_samples=config.training.num_samples_train,
            snr_base=config.signal.snr_base,
            snr_range=config.signal.snr_range
        )
        
        # Generate validation data
        print("Generating validation data...")
        val_data = dataset_gen.generate_training_data(
            num_samples=config.training.num_samples_val,
            snr_base=config.signal.snr_base,
            snr_range=config.signal.snr_range
        )
        
        print(f"Data generation time: {time.time() - start_time:.1f}s")
        
        # Train
        history = train_cycle(
            model,
            train_data,
            epochs=config.training.epochs_per_cycle,
            batch_size=config.training.batch_size,
            validation_data=val_data,
            validation_batch_size=config.training.validation_batch_size
        )
        
        # Append history
        for key in history.history:
            if key in all_history:
                all_history[key].extend(history.history[key])
        
        # Save checkpoint
        if (cycle + 1) % config.training.save_frequency == 0:
            checkpoint_path = output_dir / f"{model_name}_cycle{cycle + 1}.h5"
            model.encoder.save(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Cleanup
        del train_data, val_data
        gc.collect()
        
        print(f"Cycle time: {time.time() - start_time:.1f}s")
    
    # Save final model
    final_path = output_dir / f"{model_name}_final.h5"
    model.encoder.save(str(final_path))
    print(f"\nSaved final model: {final_path}")
    
    # Save decoder separately
    decoder_path = output_dir / f"{model_name}_decoder.h5"
    model.decoder.save(str(decoder_path))
    print(f"Saved decoder: {decoder_path}")
    
    # Plot training history
    print("Generating training plots...")
    fig = plot_training_history(all_history, save_path=str(output_dir / f"{model_name}_history.png"))
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
