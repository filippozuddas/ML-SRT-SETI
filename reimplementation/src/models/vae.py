"""
Contrastive β-VAE Model.

Custom Keras model implementing the SETI signal detection VAE
with contrastive clustering loss.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Dict, Optional

from .encoder import build_encoder
from .decoder import build_decoder
from .sampling import Sampling


class ContrastiveVAE(keras.Model):
    """
    Contrastive Variational Autoencoder for SETI signal detection.
    
    This model combines a standard VAE with contrastive learning losses:
    - Reconstruction loss: Binary crossentropy between input and output
    - KL divergence: Regularize latent space to unit Gaussian
    - Clustering loss: Push same-type observations together, different apart
    
    The training balances these losses with weights alpha, beta, gamma.
    """
    
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 alpha: float = 10,
                 beta: float = 1.5,  # Paper: beta=1.5 for balance with clustering loss
                 gamma: float = 0,
                 **kwargs):
        """
        Initialize Contrastive VAE.
        
        Args:
            encoder: Encoder model
            decoder: Decoder model
            alpha: Weight for clustering loss
            beta: Weight for KL divergence
            gamma: Weight for score loss (disabled by default)
        """
        super().__init__(**kwargs)
        
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Metrics trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.true_loss_tracker = keras.metrics.Mean(name="true_loss")
        self.false_loss_tracker = keras.metrics.Mean(name="false_loss")
        
        # Validation metrics
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
        self.val_true_loss_tracker = keras.metrics.Mean(name="val_true_loss")
        self.val_false_loss_tracker = keras.metrics.Mean(name="val_false_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.true_loss_tracker,
            self.false_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
            self.val_true_loss_tracker,
            self.val_false_loss_tracker,
        ]
    
    def call(self, inputs, training=False):
        """Forward pass through encoder and decoder.
        
        Handles both single input and list of inputs (for model.fit with multiple inputs).
        """
        # Handle list of inputs from model.fit([vae_data, true_data, false_data], ...)
        if isinstance(inputs, (list, tuple)):
            vae_input = inputs[0]  # Only use first input for VAE forward pass
        else:
            vae_input = inputs
            
        z_mean, z_log_var, z = self.encoder(vae_input, training=training)
        reconstruction = self.decoder(z, training=training)
        return z_mean, z_log_var, z, reconstruction
    
    @tf.function
    def loss_same(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Calculate mean Euclidean distance between batches.
        
        Used to measure similarity - lower is more similar.
        """
        return tf.reduce_mean(tf.norm(a - b, axis=1))
    
    @tf.function
    def loss_diff(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Calculate inverse of mean Euclidean distance.
        
        Used to maximize distance between dissimilar items.
        """
        return 1.0 / (self.loss_same(a, b) + 1e-8)
    
    @tf.function
    def true_clustering(self, true_data: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Calculate clustering loss for TRUE samples.
        
        For true samples (ETI pattern):
        - A1, A2, A3 (ON observations) should cluster together
        - B, C, D (OFF observations) should cluster together  
        - ON and OFF clusters should be far apart
        
        Args:
            true_data: Cadence data of shape (batch, 6, height, width, channels)
            training: Whether in training mode
            
        Returns:
            Clustering loss value
        """
        # Get latent representations for each observation
        a1 = self.encoder(true_data[:, 0, :, :, :], training=training)[2]
        b = self.encoder(true_data[:, 1, :, :, :], training=training)[2]
        a2 = self.encoder(true_data[:, 2, :, :, :], training=training)[2]
        c = self.encoder(true_data[:, 3, :, :, :], training=training)[2]
        a3 = self.encoder(true_data[:, 4, :, :, :], training=training)[2]
        d = self.encoder(true_data[:, 5, :, :, :], training=training)[2]
        
        # Same-class loss: minimize distance within ON and OFF groups
        same = tf.constant(0.0)
        same += self.loss_same(a1, a2) + self.loss_same(a1, a3) + self.loss_same(a2, a3)
        same += self.loss_same(b, c) + self.loss_same(c, d) + self.loss_same(b, d)
        
        # Different-class loss: maximize distance between ON and OFF
        diff = tf.constant(0.0)
        diff += self.loss_diff(a1, b) + self.loss_diff(a1, c) + self.loss_diff(a1, d)
        diff += self.loss_diff(a2, b) + self.loss_diff(a2, c) + self.loss_diff(a2, d)
        diff += self.loss_diff(a3, b) + self.loss_diff(a3, c) + self.loss_diff(a3, d)
        
        return same + diff
    
    @tf.function
    def false_clustering(self, false_data: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Calculate clustering loss for FALSE samples.
        
        For false samples (RFI/noise pattern):
        - All 6 observations should be similar (same cluster)
        
        Args:
            false_data: Cadence data of shape (batch, 6, height, width, channels)
            training: Whether in training mode
            
        Returns:
            Clustering loss value
        """
        # Get latent representations
        a1 = self.encoder(false_data[:, 0, :, :, :], training=training)[2]
        b = self.encoder(false_data[:, 1, :, :, :], training=training)[2]
        a2 = self.encoder(false_data[:, 2, :, :, :], training=training)[2]
        c = self.encoder(false_data[:, 3, :, :, :], training=training)[2]
        a3 = self.encoder(false_data[:, 4, :, :, :], training=training)[2]
        d = self.encoder(false_data[:, 5, :, :, :], training=training)[2]
        
        # All observations should cluster together
        same = tf.constant(0.0)
        same += self.loss_same(a1, a2) + self.loss_same(a1, a3)
        same += self.loss_same(a2, a3)
        same += self.loss_same(b, c) + self.loss_same(c, d) + self.loss_same(b, d)
        
        # Cross-group distances should also be small for false samples
        diff = tf.constant(0.0)
        diff += self.loss_same(a1, b) + self.loss_same(a1, c) + self.loss_same(a1, d)
        diff += self.loss_same(a2, b) + self.loss_same(a2, c) + self.loss_same(a2, d)
        diff += self.loss_same(a3, b) + self.loss_same(a3, c) + self.loss_same(a3, d)
        
        return same + diff
    
    def train_step(self, data):
        """
        Custom training step.
        
        Args:
            data: Tuple of ((vae_input, true_data, false_data), target)
                  All inputs should have same first dimension (N*6).
        """
        x, y = data
        vae_input = x[0]
        true_data = x[1]
        false_data = x[2]
        
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(vae_input, training=True)
            reconstruction = self.decoder(z, training=True)
            
            # Reconstruction loss (no normalization, like original)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(y, reconstruction),
                    axis=(1, 2)
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Contrastive clustering losses
            true_loss = self.true_clustering(true_data, training=True)
            false_loss = self.false_clustering(false_data, training=True)
            
            # Total loss
            total_loss = (
                reconstruction_loss + 
                self.beta * kl_loss + 
                self.alpha * (true_loss + false_loss)
            )
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.true_loss_tracker.update_state(true_loss)
        self.false_loss_tracker.update_state(false_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "true_loss": self.true_loss_tracker.result(),
            "false_loss": self.false_loss_tracker.result(),
        }
    
    def test_step(self, data):
        """Custom validation step."""
        x, y = data
        vae_input = x[0]
        true_data = x[1]
        false_data = x[2]
        
        # Forward pass (no training)
        z_mean, z_log_var, z = self.encoder(vae_input, training=False)
        reconstruction = self.decoder(z, training=False)
        
        # Calculate losses (no normalization, like original)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(y, reconstruction),
                axis=(1, 2)
            )
        )
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        true_loss = self.true_clustering(true_data, training=False)
        false_loss = self.false_clustering(false_data, training=False)
        
        total_loss = (
            reconstruction_loss + 
            self.beta * kl_loss + 
            self.alpha * (true_loss + false_loss)
        )
        
        # Update validation metrics
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        self.val_true_loss_tracker.update_state(true_loss)
        self.val_false_loss_tracker.update_state(false_loss)
        
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
            "true_loss": self.val_true_loss_tracker.result(),
            "false_loss": self.val_false_loss_tracker.result(),
        }
    
    def get_config(self):
        """Return model configuration."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }


def build_vae(
    input_shape: Tuple[int, int, int] = (16, 512, 1),
    latent_dim: int = 8,
    dense_units: int = 512,
    kernel_size: Tuple[int, int] = (3, 3),
    alpha: float = 10,
    beta: float = 1.5,  # Paper: beta=1.5 for balance with clustering loss
    gamma: float = 0,
    learning_rate: float = 0.001,
    l1_weight: float = 0.001,
    l2_weight: float = 0.01
) -> ContrastiveVAE:
    """
    Build and compile a Contrastive VAE model.
    
    Args:
        input_shape: Input spectrogram shape
        latent_dim: Latent space dimension
        dense_units: Dense layer units
        kernel_size: Convolution kernel size
        alpha: Clustering loss weight
        beta: KL divergence weight
        gamma: Score loss weight
        learning_rate: Optimizer learning rate
        l1_weight: L1 regularization weight
        l2_weight: L2 regularization weight
        
    Returns:
        Compiled ContrastiveVAE model
    """
    # Build encoder and decoder
    encoder = build_encoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        dense_units=dense_units,
        kernel_size=kernel_size,
        l1_weight=l1_weight,
        l2_weight=l2_weight
    )
    
    decoder = build_decoder(
        latent_dim=latent_dim,
        output_shape=input_shape,
        dense_units=dense_units,
        kernel_size=kernel_size,
        l1_weight=l1_weight,
        l2_weight=l2_weight
    )
    
    # Build VAE
    vae = ContrastiveVAE(encoder, decoder, alpha, beta, gamma)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
    # Print summaries
    encoder.summary()
    decoder.summary()
    
    return vae
