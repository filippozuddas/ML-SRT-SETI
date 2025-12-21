"""
Visualization utilities for SETI data and model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path


def plot_cadence(
    cadence: np.ndarray,
    title: str = "SETI Cadence",
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 8),
    cmap: str = 'hot',
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot a 6-observation cadence as a grid.
    
    Args:
        cadence: Array of shape (6, height, width) or (6, height, width, 1)
        title: Plot title
        labels: Labels for each observation (default: A1, B, A2, C, A3, D)
        figsize: Figure size
        cmap: Colormap to use
        save_path: If provided, save figure to this path
        
    Returns:
        Matplotlib figure object
    """
    if labels is None:
        labels = ['A1 (ON)', 'B (OFF)', 'A2 (ON)', 'C (OFF)', 'A3 (ON)', 'D (OFF)']
    
    # Remove channel dimension if present
    if cadence.ndim == 4:
        cadence = cadence[..., 0]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    
    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        im = ax.imshow(cadence[i], aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_title(label)
        ax.set_xlabel('Frequency Channel')
        ax.set_ylabel('Time Bin')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_signal_injection(
    original: np.ndarray,
    injected: np.ndarray,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> Figure:
    """
    Compare original and signal-injected spectrograms.
    
    Args:
        original: Original spectrogram (height, width)
        injected: Spectrogram with injected signal
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    im0 = axes[0].imshow(original, aspect='auto', cmap='hot')
    axes[0].set_title('Original')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(injected, aspect='auto', cmap='hot')
    axes[1].set_title('With Signal')
    plt.colorbar(im1, ax=axes[1])
    
    diff = injected - original
    im2 = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r')
    axes[2].set_title('Difference')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_latent_space(
    latents_true: np.ndarray,
    latents_false: np.ndarray,
    dims: Tuple[int, int] = (0, 1),
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot 2D projection of latent space.
    
    Args:
        latents_true: Latent vectors for true samples (batch, latent_dim)
        latents_false: Latent vectors for false samples
        dims: Which dimensions to plot
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(
        latents_false[:, dims[0]], latents_false[:, dims[1]], 
        alpha=0.5, label='False (RFI/Noise)', c='blue', s=20
    )
    ax.scatter(
        latents_true[:, dims[0]], latents_true[:, dims[1]], 
        alpha=0.5, label='True (Signal)', c='red', s=20
    )
    
    ax.set_xlabel(f'Latent Dim {dims[0]}')
    ax.set_ylabel(f'Latent Dim {dims[1]}')
    ax.set_title('Latent Space Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with metric names as keys and lists of values
        metrics: Specific metrics to plot. If None, plot all.
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        Matplotlib figure object
    """
    if metrics is None:
        # Default metrics to plot
        metrics = [k for k in history.keys() if not k.startswith('val_')]
    
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flat if n_metrics > 1 else [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}')
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric}', linestyle='--')
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_snr_sensitivity(
    snr_values: List[float],
    tpr_values: List[float],
    fpr_values: List[float],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot TPR and FPR vs SNR.
    
    Args:
        snr_values: List of SNR values
        tpr_values: True positive rates at each SNR
        fpr_values: False positive rates at each SNR
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(snr_values, tpr_values, 'g-o', label='True Positive Rate', linewidth=2)
    ax.plot(snr_values, fpr_values, 'r-s', label='False Positive Rate', linewidth=2)
    
    ax.set_xlabel('Signal-to-Noise Ratio (SNR)')
    ax.set_ylabel('Rate')
    ax.set_title('Detection Performance vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_samples: int = 4,
    figsize: Tuple[int, int] = (15, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot original vs reconstructed spectrograms.
    
    Args:
        original: Original data (batch, height, width, 1)
        reconstructed: Reconstructed data
        n_samples: Number of samples to show
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    for i in range(n_samples):
        axes[0, i].imshow(original[i, :, :, 0], aspect='auto', cmap='hot')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed[i, :, :, 0], aspect='auto', cmap='hot')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
