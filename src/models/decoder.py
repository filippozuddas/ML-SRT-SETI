"""
VAE Decoder network.

Transposed convolutional decoder for the SETI β-VAE model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1, l2
from typing import Tuple


def build_decoder(
    latent_dim: int = 8,
    output_shape: Tuple[int, int, int] = (16, 512, 1),
    dense_units: int = 512,
    kernel_size: Tuple[int, int] = (3, 3),
    l1_weight: float = 0.001,
    l2_weight: float = 0.01
) -> keras.Model:
    """
    Build the VAE decoder network.
    
    Architecture mirrors the encoder with transposed convolutions
    to reconstruct the input from the latent representation.
    
    Args:
        latent_dim: Dimension of latent space
        output_shape: Target output shape (height, width, channels)
        dense_units: Units in dense layer
        kernel_size: Convolution kernel size
        l1_weight: L1 regularization weight
        l2_weight: L2 regularization weight
        
    Returns:
        Keras decoder model
    """
    latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    
    # Dense layers to reshape
    x = layers.Dense(
        dense_units, 
        activation="relu",
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight),
        name='dense1'
    )(latent_inputs)
    
    x = layers.Dense(
        1 * 32 * 256,  # Shape before reshape
        activation="relu",
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight),
        name='dense2'
    )(x)
    
    # Reshape to start transposed convolutions
    x = layers.Reshape((1, 32, 256), name='reshape')(x)
    
    # Transposed convolution layers (mirror of encoder)
    # Block 1: 1 -> 2 (stride 2)
    x = layers.Conv2DTranspose(256, kernel_size, activation="relu", strides=2, 
                               padding="same", name='deconv1')(x)
    x = layers.Conv2DTranspose(128, kernel_size, activation="relu", strides=1, 
                               padding="same", name='deconv2')(x)
    
    # Block 2: 2 -> 4 (stride 2)
    x = layers.Conv2DTranspose(64, kernel_size, activation="relu", strides=1, 
                               padding="same", name='deconv3')(x)
    x = layers.Conv2DTranspose(64, kernel_size, activation="relu", strides=2, 
                               padding="same", name='deconv4')(x)
    
    # Block 3: 4 -> 8 (stride 2)
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=1, 
                               padding="same", name='deconv5')(x)
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=1, 
                               padding="same", name='deconv6')(x)
    x = layers.Conv2DTranspose(32, kernel_size, activation="relu", strides=2, 
                               padding="same", name='deconv7')(x)
    
    # Block 4: 8 -> 16 (stride 2)
    x = layers.Conv2DTranspose(16, kernel_size, activation="relu", strides=1, 
                               padding="same", name='deconv8')(x)
    x = layers.Conv2DTranspose(16, kernel_size, activation="relu", strides=2, 
                               padding="same", name='deconv9')(x)
    
    # Output layer with sigmoid activation
    decoder_outputs = layers.Conv2DTranspose(
        output_shape[2], kernel_size, 
        activation="sigmoid", 
        padding="same",
        name='decoder_output'
    )(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    return decoder


def build_decoder_flexible(
    latent_dim: int = 8,
    output_shape: Tuple[int, int, int] = (16, 512, 1),
    dense_units: int = 512,
    kernel_size: Tuple[int, int] = (3, 3),
    l1_weight: float = 0.001,
    l2_weight: float = 0.01,
    dropout_rate: float = 0.0
) -> keras.Model:
    """
    Build a flexible decoder with configurable architecture.
    
    Args:
        latent_dim: Latent space dimension
        output_shape: Target output shape
        dense_units: Dense layer units
        kernel_size: Convolution kernel size
        l1_weight: L1 regularization weight
        l2_weight: L2 regularization weight
        dropout_rate: Dropout rate (0 to disable)
        
    Returns:
        Keras decoder model
    """
    latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    
    x = layers.Dense(
        dense_units,
        activation="relu",
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight),
        name='dense1'
    )(latent_inputs)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(
        1 * 32 * 256,
        activation="relu",
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight),
        name='dense2'
    )(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Reshape((1, 32, 256), name='reshape')(x)
    
    # Transposed convolutions
    conv_filters = [256, 128, 64, 64, 32, 32, 32, 16, 16]
    stride_layers = {0, 3, 6, 8}  # Indices with stride 2
    
    for i, filters in enumerate(conv_filters):
        stride = 2 if i in stride_layers else 1
        x = layers.Conv2DTranspose(
            filters, kernel_size,
            activation="relu",
            strides=stride,
            padding="same",
            name=f'deconv{i+1}'
        )(x)
        
        if dropout_rate > 0 and i < len(conv_filters) - 1:
            x = layers.Dropout(dropout_rate)(x)
    
    decoder_outputs = layers.Conv2DTranspose(
        output_shape[2], kernel_size,
        activation="sigmoid",
        padding="same",
        name='decoder_output'
    )(x)
    
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")
