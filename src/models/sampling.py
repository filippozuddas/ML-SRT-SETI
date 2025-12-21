"""
Sampling layer for VAE.

Custom Keras layer implementing the reparameterization trick.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """
    Sampling layer using the reparameterization trick.
    
    Samples from the latent distribution N(z_mean, z_log_var) using:
    z = z_mean + exp(0.5 * z_log_var) * epsilon
    where epsilon ~ N(0, 1)
    
    This allows gradients to flow through the sampling operation.
    """
    
    def call(self, inputs):
        """
        Sample from latent distribution.
        
        Args:
            inputs: Tuple of (z_mean, z_log_var)
            
        Returns:
            Sampled latent vector z
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Sample epsilon from standard normal
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        # Reparameterization trick
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        """Return layer configuration."""
        return super().get_config()


def sample_from_latent(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
    """
    Functional implementation of sampling.
    
    Args:
        z_mean: Mean of latent distribution
        z_log_var: Log variance of latent distribution
        
    Returns:
        Sampled latent vector
    """
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
