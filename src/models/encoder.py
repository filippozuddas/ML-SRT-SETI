"""
VAE Encoder network.

Convolutional encoder for the SETI β-VAE model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1, l2
from typing import Tuple, List, Optional
from .sampling import Sampling


def build_encoder(
    input_shape: Tuple[int, int, int] = (16, 512, 1),
    latent_dim: int = 8,
    dense_units: int = 512,
    kernel_size: Tuple[int, int] = (3, 3),
    l1_weight: float = 0.001,
    l2_weight: float = 0.01
) -> keras.Model:
    """
    Build the VAE encoder network.
    
    Architecture:
    - 9 convolutional layers with increasing filters
    - Dense layer with regularization
    - Two output heads: z_mean and z_log_var
    - Sampling layer for reparameterization
    
    Args:
        input_shape: Input spectrogram shape (height, width, channels)
        latent_dim: Dimension of latent space
        dense_units: Units in dense layer
        kernel_size: Convolution kernel size
        l1_weight: L1 regularization weight
        l2_weight: L2 regularization weight
        
    Returns:
        Keras model with outputs [z_mean, z_log_var, z]
    """
    encoder_inputs = keras.Input(shape=input_shape, name='encoder_input')
    
    # Convolutional layers
    # Block 1: 16 -> 8 (stride 2)
    x = layers.Conv2D(16, kernel_size, activation="relu", strides=2, padding="same",
                      name='conv1')(encoder_inputs)
    x = layers.Conv2D(16, kernel_size, activation="relu", strides=1, padding="same",
                      name='conv2')(x)
    
    # Block 2: 8 -> 4 (stride 2)
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=2, padding="same",
                      name='conv3')(x)
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=1, padding="same",
                      name='conv4')(x)
    x = layers.Conv2D(32, kernel_size, activation="relu", strides=1, padding="same",
                      name='conv5')(x)
    
    # Block 3: 4 -> 2 (stride 2)
    x = layers.Conv2D(64, kernel_size, activation="relu", strides=2, padding="same",
                      name='conv6')(x)
    x = layers.Conv2D(64, kernel_size, activation="relu", strides=1, padding="same",
                      name='conv7')(x)
    x = layers.Conv2D(128, kernel_size, activation="relu", strides=1, padding="same",
                      name='conv8')(x)
    
    # Block 4: 2 -> 1 (stride 2)
    x = layers.Conv2D(256, kernel_size, activation="relu", strides=2, padding="same",
                      name='conv9')(x)
    
    # Flatten and dense
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(
        dense_units, 
        activation="relu",
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight),
        name='dense'
    )(x)
    
    # Latent space parameters
    z_mean = layers.Dense(
        latent_dim, 
        name="z_mean",
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight)
    )(x)
    
    z_log_var = layers.Dense(
        latent_dim, 
        name="z_log_var",
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight)
    )(x)
    
    # Sampling layer
    z = Sampling(name='sampling')([z_mean, z_log_var])
    
    encoder = keras.Model(
        encoder_inputs, 
        [z_mean, z_log_var, z], 
        name="encoder"
    )
    
    return encoder


def build_encoder_flexible(
    input_shape: Tuple[int, int, int] = (16, 512, 1),
    latent_dim: int = 8,
    dense_units: int = 512,
    conv_filters: Optional[List[int]] = None,
    kernel_size: Tuple[int, int] = (3, 3),
    l1_weight: float = 0.001,
    l2_weight: float = 0.01,
    dropout_rate: float = 0.0
) -> keras.Model:
    """
    Build a flexible encoder with configurable architecture.
    
    Args:
        input_shape: Input spectrogram shape
        latent_dim: Latent space dimension
        dense_units: Dense layer units  
        conv_filters: List of filter counts for each conv layer
        kernel_size: Convolution kernel size
        l1_weight: L1 regularization weight
        l2_weight: L2 regularization weight
        dropout_rate: Dropout rate (0 to disable)
        
    Returns:
        Keras encoder model
    """
    if conv_filters is None:
        conv_filters = [16, 16, 32, 32, 32, 64, 64, 128, 256]
    
    # Define which layers have stride 2
    stride_layers = {0, 2, 5, 8}  # Indices with stride 2
    
    encoder_inputs = keras.Input(shape=input_shape, name='encoder_input')
    x = encoder_inputs
    
    for i, filters in enumerate(conv_filters):
        stride = 2 if i in stride_layers else 1
        x = layers.Conv2D(
            filters, kernel_size, 
            activation="relu", 
            strides=stride, 
            padding="same",
            name=f'conv{i+1}'
        )(x)
        
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight),
        name='dense'
    )(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    z_mean = layers.Dense(
        latent_dim,
        name="z_mean",
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight)
    )(x)
    
    z_log_var = layers.Dense(
        latent_dim,
        name="z_log_var", 
        activity_regularizer=l1(l1_weight),
        kernel_regularizer=l2(l2_weight),
        bias_regularizer=l2(l2_weight)
    )(x)
    
    z = Sampling(name='sampling')([z_mean, z_log_var])
    
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
