import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Import REQUIRED per i regolarizzatori L1/L2
from keras.regularizers import l1, l2
import numpy as np
from . import config 


class Sampling(layers.Layer):
    """Layer di campionamento VAE"""
    def call(self, inputs):
        z_mean, z_log_var = inputs        
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Definiamo i regolarizzatori di Ma (usati su quasi tutti i layer)
REG_PARAMS = {
    "activity_regularizer": l1(0.001),
    "kernel_regularizer": l2(0.01),
    "bias_regularizer": l2(0.01)
}
DEFAULT_DENSE_LAYERS = 512
DEFAULT_KERNEL = (3, 3)

def build_encoder(input_shape, latent_dim):
    # Input: (16, 512, 1)
    encoder_inputs = keras.Input(shape=input_shape)
    
    # Livello 1: 16x512 -> 8x256
    x = layers.Conv2D(16, DEFAULT_KERNEL, activation="relu", strides=2, padding="same", **REG_PARAMS)(encoder_inputs)
    x = layers.Conv2D(16, DEFAULT_KERNEL, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    
    # Livello 2: 8x256 -> 4x128
    x = layers.Conv2D(32, DEFAULT_KERNEL, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(32, DEFAULT_KERNEL, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(32, DEFAULT_KERNEL, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    
    # Livello 3: 4x128 -> 2x64
    x = layers.Conv2D(64, DEFAULT_KERNEL, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(64, DEFAULT_KERNEL, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    
    # Livello 4: 2x64 -> 1x32 (Notare che in Ma c'erano più filtri, replichiamo il numero di layers)
    x = layers.Conv2D(128, DEFAULT_KERNEL, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(256, DEFAULT_KERNEL, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(DEFAULT_DENSE_LAYERS, activation="relu", **REG_PARAMS)(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean", **REG_PARAMS)(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", **REG_PARAMS)(x)
    z = Sampling()([z_mean, z_log_var])
    
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def build_decoder(latent_dim, target_shape):
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    # Ricostruzione inversa degli strati Dense
    x = layers.Dense(DEFAULT_DENSE_LAYERS, activation="relu", **REG_PARAMS)(latent_inputs)
    x = layers.Dense(1 * 32 * 256, activation="relu", **REG_PARAMS)(x)
    x = layers.Reshape((1, 32, 256))(x)
    
    # 4 Livelli De-Convoluzionali
    # 1x32 -> 2x64
    x = layers.Conv2DTranspose(256, DEFAULT_KERNEL, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    # 2x64 -> 4x128
    x = layers.Conv2DTranspose(128, DEFAULT_KERNEL, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    # 4x128 -> 8x256
    x = layers.Conv2DTranspose(64, DEFAULT_KERNEL, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    # 8x256 -> 16x512
    x = layers.Conv2DTranspose(32, DEFAULT_KERNEL, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    
    # Output Sigmoid per dati [0, 1]
    decoder_outputs = layers.Conv2DTranspose(1, DEFAULT_KERNEL, activation="sigmoid", padding="same")(x)
    
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Il training di Ma è un training multi-loss molto complesso (Cella [15]) che non possiamo implementare in un semplice CVAE.
# Qui implementiamo solo il VAE standard con la loss SOMMATA (quella che esplode, ma che Ma gestisce con L1/L2)
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.5, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def _calculate_loss(self, data, training=False):
        if isinstance(data, tuple): data = data[0]
            
        # L'uso di training=training è necessario per L1/L2
        z_mean, z_log_var, z = self.encoder(data, training=training)
        reconstruction = self.decoder(z, training=training)
        
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        total_loss = reconstruction_loss + (self.beta * kl_loss)
        
        # Aggiungiamo le L2 penalties che Keras non include automaticamente nella loss custom
        total_loss += tf.add_n(self.losses)
        
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple): x, y = data
        else: x, y = data, data

        with tf.GradientTape() as tape:
            # Passiamo y (dati target) per la loss di ricostruzione
            total_loss, recon_loss, kl_loss = self._calculate_loss(y, training=True)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {"total_loss": self.total_loss_tracker.result(), "reconstruction_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}

    def test_step(self, data):
        if isinstance(data, tuple): x, y = data
        else: x, y = data, data
            
        total_loss, recon_loss, kl_loss = self._calculate_loss(y, training=False)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {"total_loss": self.total_loss_tracker.result(), "reconstruction_loss": self.reconstruction_loss_tracker.result(), "kl_loss": self.kl_loss_tracker.result()}

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs, training=False)
        return self.decoder(z, training=False)