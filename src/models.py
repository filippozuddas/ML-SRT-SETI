import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from . import config

class Sampling(layers.Layer):
    """
        Usato per il campionamento nel VAE
    """
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
def build_encoder(input_shape, latent_dim):
    """
        Costruisce la parte Encoder della rete
    """
    
    encoder_inputs = keras.Input(shape=input_shape)
    
    # Blocco Convoluzionale 1
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
    x = layers.BatchNormalization()(x)
    
    # Blocco Convoluzionale 2
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Blocco Convoluzionale 3
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Appiattimento e Vettore Latente
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    
    # Output media e log-varianza (per la distribuzione gaussiana)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    # Layer di campionamento (Reparameterization trick)
    z = Sampling()([z_mean, z_log_var])
    
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def build_decoder(latent_dim, target_shape):
    """
        Costruisce la parte Decoder della rete (Speculare all'Encoder)
    """
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    # Calcoliamo le dimensioni da cui partire dopo il Dense per fare il Reshape
    # Input originale: 16 x 512. Dopo 3 stride da 2: 
    # 16 / 8 = 2
    # 512 / 8 = 64
    # Canali finali encoder = 128
    shape_before_flattening = (2, 64, 128) 
    
    x = layers.Dense(np.prod(shape_before_flattening), activation="relu")(latent_inputs)
    x = layers.Reshape(shape_before_flattening)(x)
    
    # Deconvoluzione 1
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Deconvoluzione 2
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Deconvoluzione 3
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Output finale (Ricostruzione)
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same", activation="linear")(x)
    
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

class CVAE(keras.Model):
    """
        Modello Completo Convolutional Variational Autoencoder con parametro Beta
    """
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta # Beta factor per disentanglement (Ma et al. usano valori specifici)
        
        # Metriche per tracciare il training
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Calcolo Loss Ricostruzione (MSE o Binary Crossentropy)
            # Qui usiamo MSE perché i dati sono normalizzati
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data, reconstruction), axis=(1, 2)
                )
            )
            
            # Calcolo Loss KL Divergence
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Loss Totale
            total_loss = reconstruction_loss + (self.beta * kl_loss)
            
        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Aggiornamento metriche
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mse(data, reconstruction), axis=(1, 2)
            )
        )
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        total_loss = reconstruction_loss + (self.beta * kl_loss)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def call(self, inputs):
        # Per inferenza standard
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)