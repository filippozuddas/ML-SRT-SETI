#!/usr/bin/env python3
"""
Plot ETI candidates from pipeline results.

Usage:
    python -m src.inference.plot_candidates --cadence data/*.h5 --candidates results_candidates.csv
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from typing import List, Tuple

from ..data.loader import SRTDataLoader


def load_candidates(csv_path: str) -> List[dict]:
    """Load candidates from CSV file."""
    candidates = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append({
                'freq_mhz': float(row['freq_mhz']),
                'probability': float(row['probability']),
                'center_channel': int(row['center_channel'])
            })
    return candidates

def plot_candidate_processed(cadence_data: np.ndarray, 
                              center_channel: int,
                              freq_mhz: float,
                              probability: float,
                              output_path: str = None):
    """
    Plot the 16x512 processed snippet that the model sees.
    """
    SNIPPET_WIDTH = 4096
    DOWNSAMPLE_FACTOR = 8
    FINAL_FREQ_BINS = 512
    
    # Extract snippet
    start = max(0, center_channel - SNIPPET_WIDTH // 2)
    end = min(cadence_data.shape[2], center_channel + SNIPPET_WIDTH // 2)
    snippet = cadence_data[:, :, start:end]
    
    # Downsample frequency by 8x
    if snippet.shape[2] == SNIPPET_WIDTH:
        snippet_ds = snippet.reshape(6, snippet.shape[1], FINAL_FREQ_BINS, DOWNSAMPLE_FACTOR).mean(axis=3)
    else:
        # Fallback if snippet is smaller
        snippet_ds = snippet
    
    # Log normalize per observation
    processed = np.zeros_like(snippet_ds)
    for i in range(6):
        obs_data = snippet_ds[i].astype(np.float64)
        obs_data = np.log(np.abs(obs_data) + 1e-10)
        obs_data = obs_data - obs_data.min()
        max_val = obs_data.max()
        if max_val > 0:
            obs_data = obs_data / max_val
        processed[i] = obs_data
    
    # Create figure
    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
    fig.subplots_adjust(hspace=0)
    
    for i in range(6):
        ax = axes[i]
        obs_type = "ON" if i % 2 == 0 else "OFF"
        
        data_plot = processed[i]
        
        img = ax.imshow(data_plot, aspect='auto', cmap='hot', origin='upper',
                        extent=[0, FINAL_FREQ_BINS, data_plot.shape[0], 0],
                        vmin=0, vmax=1)
        
        ax.text(0.02, 0.85, f'Obs {i+1} ({obs_type})', transform=ax.transAxes, 
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='square', facecolor='black', alpha=0.5))
        
        ax.set_ylabel('Time bin')
        
        if i < 5:
            ax.tick_params(labelbottom=False)
    
    axes[5].set_xlabel('Frequency bins (512 - After 8x downsampling)')
    
    cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Log-Normalized Power [0-1]')
    
    fig.suptitle(f'ETI Candidate (Model View): {freq_mhz:.4f} MHz - P(ETI)={probability:.3f}', 
                 fontsize=14, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {Path(output_path).name}")
    
    plt.close()
    return fig


def plot_all_candidates(file_paths: List[str],
                       candidates: List[dict],
                       output_dir: str,
                       max_plots: int = 300):
    """
    Plot all candidates from a cadence.
    """
    print(f"\n📊 Plotting {min(len(candidates), max_plots)} candidates...")
    
    # Load cadence
    loader = SRTDataLoader()
    cadence = loader.load_cadence(file_paths)
    cadence_data = cadence.to_array()  # (6, time, freq)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by probability (highest first)
    sorted_candidates = sorted(candidates, key=lambda x: x['probability'], reverse=True)
    
    # Plot each candidate
    for i, cand in enumerate(sorted_candidates[:max_plots]):
        # Processed 512-channel plot (what model sees)
        filename_proc = f"candidate_{i+1}_{cand['freq_mhz']:.2f}MHz_P{cand['probability']:.3f}_processed.png"
        plot_candidate_processed(
            cadence_data=cadence_data,
            center_channel=cand['center_channel'],
            freq_mhz=cand['freq_mhz'],
            probability=cand['probability'],
            output_path=str(output_dir / filename_proc)
        )
    
    # Create summary plot
    plot_summary(candidates, output_dir / "candidates_summary.png")
    
    print(f"\n✅ All plots saved to {output_dir}")


def plot_summary(candidates: List[dict], output_path: str):
    """Create a summary bar plot of all candidates."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by probability
    sorted_cands = sorted(candidates, key=lambda x: x['probability'], reverse=True)
    
    freqs = [c['freq_mhz'] for c in sorted_cands]
    probs = [c['probability'] for c in sorted_cands]
    
    colors = ['#2ecc71' if p >= 0.95 else '#f39c12' if p >= 0.9 else '#e74c3c' for p in probs]
    
    bars = ax.bar(range(len(sorted_cands)), probs, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Candidate (sorted by probability)', fontsize=12)
    ax.set_ylabel('ETI Probability', fontsize=12)
    ax.set_title('ETI Candidates Summary', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.9)')
    
    # Add value labels on bars
    for bar, prob, freq in zip(bars, probs, freqs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Add frequency labels
    ax.set_xticks(range(len(sorted_cands)))
    ax.set_xticklabels([f'{f:.2f}' for f in freqs], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='P ≥ 0.95'),
        Patch(facecolor='#f39c12', label='0.9 ≤ P < 0.95'),
        Patch(facecolor='#e74c3c', label='P < 0.9')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {Path(output_path).name}")


def main():
    parser = argparse.ArgumentParser(description='Plot ETI candidates')
    parser.add_argument('--cadence', '-c', nargs=6, required=True,
                        help='6 observation files in order [ON,OFF,ON,OFF,ON,OFF]')
    parser.add_argument('--candidates', '-i', required=True,
                        help='CSV file with candidates')
    parser.add_argument('--output-dir', '-o', default='results/plots/candidates',
                        help='Output directory for plots')
    parser.add_argument('--max-plots', '-n', type=int, default=10,
                        help='Maximum number of candidates to plot')
    
    args = parser.parse_args()
    
    # Validate files
    for f in args.cadence:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            return
    
    if not Path(args.candidates).exists():
        print(f"Error: Candidates file not found: {args.candidates}")
        return
    
    # Load candidates
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates from {args.candidates}")
    
    # Plot
    plot_all_candidates(
        file_paths=args.cadence,
        candidates=candidates,
        output_dir=args.output_dir,
        max_plots=args.max_plots
    )


if __name__ == '__main__':
    main()
