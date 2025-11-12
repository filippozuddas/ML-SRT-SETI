import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from . import config

class Sampling(layers.Layer):
    """Usato per il campionamento nel VAE."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

    