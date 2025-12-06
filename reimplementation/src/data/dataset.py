"""
TensorFlow Dataset pipeline for SETI training.

Provides efficient data loading and augmentation using tf.data
for optimal GPU utilization.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Generator, Union
from dataclasses import dataclass
from .cadence_generator import CadenceGenerator, CadenceParams
from ..utils.preprocessing import preprocess, downscale, combine_cadences


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    batch_size: int = 1000
    num_samples: int = 6000
    snr_base: float = 20
    snr_range: float = 40
    downscale_factor: int = 8
    prefetch_buffer: int = 4
    shuffle_buffer: int = 1000


class SETIDataset:
    """
    Dataset manager for SETI training data.
    
    Handles generation of training/validation data with proper
    preprocessing and batching.
    """
    
    def __init__(self,
                 plate: Optional[np.ndarray] = None,
                 config: Optional[DatasetConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            plate: Optional real observation plate for backgrounds
            config: Dataset configuration
            seed: Random seed
        """
        self.plate = plate
        self.config = config or DatasetConfig()
        self.seed = seed
        
        # Initialize cadence generator
        cadence_params = CadenceParams(
            snr_base=self.config.snr_base,
            snr_range=self.config.snr_range
        )
        self.cadence_gen = CadenceGenerator(cadence_params, plate, seed)
    
    def generate_training_data(self,
                               num_samples: Optional[int] = None,
                               snr_base: Optional[float] = None,
                               snr_range: Optional[float] = None
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a full training dataset.
        
        Creates:
        - VAE training data (individual observations)
        - True cadence data for contrastive loss
        - False cadence data for contrastive loss
        
        Args:
            num_samples: Number of cadence samples
            snr_base: Override SNR base
            snr_range: Override SNR range
            
        Returns:
            Tuple of (vae_data, true_data, false_data)
        """
        num_samples = num_samples or self.config.num_samples
        snr_base = snr_base or self.config.snr_base
        snr_range = snr_range or self.config.snr_range
        
        print(f"Generating {num_samples} training samples...")
        
        # Generate TRUE samples for VAE training
        print("Creating True samples...")
        true_cadences = self.cadence_gen.generate_batch(
            'true_fast', num_samples, snr_base, snr_range
        )
        
        # Downscale and preprocess for VAE
        true_cadences = downscale(true_cadences, self.config.downscale_factor)
        vae_data = preprocess(true_cadences, add_channel=True)
        
        # Flatten for VAE: (N, 6, H, W, 1) -> (N*6, H, W, 1)
        vae_data_flat = combine_cadences(vae_data)
        
        # Generate FALSE samples for contrastive learning
        print("Creating False samples...")
        false_cadences = self.cadence_gen.generate_batch(
            'false', num_samples * 6, snr_base, snr_range
        )
        false_cadences = downscale(false_cadences, self.config.downscale_factor)
        false_data = preprocess(false_cadences, add_channel=True)
        
        # Generate additional TRUE samples for contrastive learning
        print("Creating True contrastive samples...")
        true_contrast_1 = self.cadence_gen.generate_batch(
            'true_fast', num_samples * 3, snr_base, snr_range
        )
        true_contrast_1 = downscale(true_contrast_1, self.config.downscale_factor)
        true_contrast_1 = preprocess(true_contrast_1, add_channel=True)
        
        true_contrast_2 = self.cadence_gen.generate_batch(
            'single_shot', num_samples * 3, snr_base, snr_range
        )
        true_contrast_2 = downscale(true_contrast_2, self.config.downscale_factor)
        true_contrast_2 = preprocess(true_contrast_2, add_channel=True)
        
        true_data = np.concatenate([true_contrast_1, true_contrast_2], axis=0)
        
        print(f"Generated - VAE: {vae_data_flat.shape}, True: {true_data.shape}, False: {false_data.shape}")
        
        return vae_data_flat, true_data, false_data


def create_tf_dataset(vae_data: np.ndarray,
                      true_data: np.ndarray,
                      false_data: np.ndarray,
                      batch_size: int = 1000,
                      shuffle: bool = True,
                      prefetch: int = 4) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset for VAE training.
    
    The dataset yields tuples of ((vae_input, true_cadences, false_cadences), vae_target)
    for the custom training loop.
    
    Args:
        vae_data: VAE input/target data
        true_data: True cadences for contrastive loss
        false_data: False cadences for contrastive loss
        batch_size: Batch size
        shuffle: Whether to shuffle data
        prefetch: Prefetch buffer size
        
    Returns:
        tf.data.Dataset
    """
    # Calculate batch indices
    num_vae_samples = vae_data.shape[0]
    num_contrast_samples = true_data.shape[0]
    
    # Ratio of contrast samples to VAE samples per batch
    vae_per_batch = batch_size
    contrast_per_batch = batch_size // 6  # 6 VAE samples per cadence
    
    def data_generator():
        """Generate batches of data."""
        indices = np.arange(num_vae_samples)
        contrast_indices = np.arange(num_contrast_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            np.random.shuffle(contrast_indices)
        
        for i in range(0, num_vae_samples - vae_per_batch + 1, vae_per_batch):
            batch_indices = indices[i:i + vae_per_batch]
            
            # Sample contrast indices
            c_start = (i // vae_per_batch * contrast_per_batch) % (num_contrast_samples - contrast_per_batch)
            c_indices = contrast_indices[c_start:c_start + contrast_per_batch]
            
            vae_batch = vae_data[batch_indices].astype(np.float32)
            true_batch = true_data[c_indices].astype(np.float32)
            false_batch = false_data[c_indices].astype(np.float32)
            
            yield (vae_batch, true_batch, false_batch), vae_batch
    
    # Define output signature
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 16, 512, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 6, 16, 512, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 6, 16, 512, 1), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 16, 512, 1), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )
    
    if prefetch > 0:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_simple_dataset(data: np.ndarray,
                          batch_size: int = 1000,
                          shuffle: bool = True) -> tf.data.Dataset:
    """
    Create a simple TensorFlow dataset for the classifier.
    
    Args:
        data: Input data array
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices(data.astype(np.float32))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(data)))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
