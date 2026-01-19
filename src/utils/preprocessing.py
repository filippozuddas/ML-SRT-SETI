"""
Preprocessing utilities for SETI data.

Implements log normalization and data reshaping operations
optimized with numba for performance.
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional
from skimage.transform import downscale_local_mean


@jit(nopython=True)
def normalize_log(data: np.ndarray) -> np.ndarray:
    """
    Apply log normalization to data.
    
    Transforms data as: log(data) -> shift to 0 -> scale to [0, 1]
    
    NOTE: As per the paper, this should be applied to the ENTIRE snippet
    (6×16×512) together, not per-observation. This ensures ON and OFF 
    observations are on the same scale.
    
    Args:
        data: Input array with positive values.
        
    Returns:
        Log-normalized array in range [0, 1].
    """
    # Apply log transform
    log_data = np.log(data)
    
    # Shift minimum to 0
    log_data = log_data - log_data.min()
    
    # Scale to [0, 1]
    max_val = log_data.max()
    if max_val > 0:
        log_data = log_data / max_val
    
    return log_data


@jit(parallel=True)
def preprocess_batch(data: np.ndarray) -> np.ndarray:
    """
    Preprocess a batch of cadence data.
    
    As per the paper: normalizes the ENTIRE snippet (6×16×512) together,
    not per-observation. This ensures ON and OFF are on the same scale.
    
    Args:
        data: Array of shape (batch, 6, height, width)
        
    Returns:
        Preprocessed array: (batch, 6, height, width)
    """
    batch_size = data.shape[0]
    height = data.shape[2]
    width = data.shape[3]
    
    result = np.zeros((batch_size, 6, height, width), dtype=np.float32)
    
    for i in prange(batch_size):
        # Normalize ENTIRE snippet together (all 6 observations)
        # This is the key difference from per-observation normalization
        result[i, :, :, :] = normalize_log(data[i, :, :, :])
    
    return result


def preprocess(data: np.ndarray, add_channel: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline for SETI data.
    
    Args:
        data: Input array of shape (batch, 6, height, width) or (6, height, width)
        add_channel: Whether to add channel dimension at the end.
        
    Returns:
        Preprocessed array.
    """
    # Handle single sample
    single_sample = data.ndim == 3
    if single_sample:
        data = data[np.newaxis, ...]
    
    # Apply batch preprocessing
    result = preprocess_batch(data)
    
    # Add channel dimension if requested
    if add_channel:
        result = result[..., np.newaxis]
    
    # Remove batch dimension if input was single sample
    if single_sample:
        result = result[0]
    
    return result


def downscale(data: np.ndarray, factor: int = 8) -> np.ndarray:
    """
    Downscale frequency dimension by averaging.
    
    Args:
        data: Input array of shape (..., height, width)
        factor: Downscaling factor for width dimension.
        
    Returns:
        Downscaled array with width reduced by factor.
    """
    if data.ndim == 4:
        # Shape: (batch, 6, height, width)
        result = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3] // factor))
        for i in range(6):
            result[:, i, :, :] = downscale_local_mean(data[:, i, :, :], (1, 1, factor))
        return result
    elif data.ndim == 3:
        # Shape: (6, height, width)
        result = np.zeros((data.shape[0], data.shape[1], data.shape[2] // factor))
        for i in range(6):
            result[i, :, :] = downscale_local_mean(data[i, :, :], (1, factor))
        return result
    else:
        return downscale_local_mean(data, (1, factor))


@jit(parallel=True)
def combine_cadences(data: np.ndarray) -> np.ndarray:
    """
    Flatten cadence dimension for model input.
    
    Transforms (batch, 6, height, width, 1) -> (batch*6, height, width, 1)
    
    Args:
        data: Array of shape (batch, 6, height, width, 1)
        
    Returns:
        Flattened array of shape (batch*6, height, width, 1)
    """
    batch_size = data.shape[0]
    cadence_len = data.shape[1]
    height = data.shape[2]
    width = data.shape[3]
    channels = data.shape[4]
    
    result = np.zeros((batch_size * cadence_len, height, width, channels), dtype=np.float32)
    
    for i in prange(batch_size):
        result[i * cadence_len:(i + 1) * cadence_len, :, :, :] = data[i, :, :, :, :]
    
    return result


def recombine_latents(latents: np.ndarray, cadence_len: int = 6) -> np.ndarray:
    """
    Recombine latent vectors for classifier input.
    
    Takes individual observation latents and concatenates them
    for each cadence sample.
    
    Args:
        latents: Array of shape (batch*6, latent_dim)
        cadence_len: Number of observations per cadence (default: 6)
        
    Returns:
        Array of shape (batch, latent_dim*6) for classifier input.
    """
    num_samples = latents.shape[0] // cadence_len
    latent_dim = latents.shape[1]
    
    result = np.zeros((num_samples, latent_dim * cadence_len), dtype=np.float32)
    
    for i in range(num_samples):
        result[i, :] = latents[i * cadence_len:(i + 1) * cadence_len, :].ravel()
    
    return result


def calculate_snr(data: np.ndarray) -> float:
    """
    Calculate approximate SNR of a data sample.
    
    Args:
        data: Input array.
        
    Returns:
        SNR estimate as max/mean ratio.
    """
    return float(data.max() / np.mean(data))
