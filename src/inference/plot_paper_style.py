#!/usr/bin/env python3
"""
Paper-style candidate visualization.

Reproduces the plotting style from the Ma et al. 2023 paper:
- Header with target metadata
- Stacked waterfall plots for all 6 observations
- 8x frequency downsampling (same as model input)
- Relative frequency axis in Hz from center
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import csv
import json
from datetime import datetime
from typing import List, Optional
import h5py

# Constants
SNIPPET_WIDTH = 4096
DOWNSAMPLE_FACTOR = 8
FINAL_FREQ_BINS = 512


def load_snippet_from_files(file_paths: List[str], center_channel: int, width: int = 4096) -> tuple:
    """Load snippet and metadata from HDF5/filterbank files."""
    snippets = []
    metadata = {}
    
    start = max(0, center_channel - width // 2)
    end = start + width
    
    for i, path in enumerate(file_paths):
        with h5py.File(path, 'r') as f:
            data = f['data'][:, 0, start:end]
            snippets.append(data)
            
            # Get metadata from first file
            if i == 0:
                metadata['fch1'] = f['data'].attrs.get('fch1', 0)
                metadata['foff'] = f['data'].attrs.get('foff', 0)
                metadata['source_name'] = f['data'].attrs.get('source_name', b'Unknown')
                if isinstance(metadata['source_name'], bytes):
                    metadata['source_name'] = metadata['source_name'].decode('utf-8', errors='ignore')
                metadata['tstart'] = f['data'].attrs.get('tstart', 0)
                metadata['tsamp'] = f['data'].attrs.get('tsamp', 1)
                metadata['nchans'] = f['data'].attrs.get('nchans', 0)
    
    return np.array(snippets), metadata


def plot_candidate_paper_style(file_paths: List[str],
                                center_channel: int,
                                freq_mhz: float,
                                probability: float,
                                target_name: str,
                                output_path: str = None,
                                candidate_id: str = None,
                                total_candidates: int = None):
    """
    Create paper-style candidate visualization.
    
    Features:
    - Header with target/observation info
    - 6 stacked waterfall panels (8x downsampled)
    - Relative frequency axis (Hz from center)
    - Hot colormap
    """
    # Load data
    snippet_data, metadata = load_snippet_from_files(file_paths, center_channel, SNIPPET_WIDTH)
    
    # Downscale 8x in frequency
    if snippet_data.shape[2] >= SNIPPET_WIDTH:
        n_time = snippet_data.shape[1]
        snippet_ds = snippet_data.reshape(6, n_time, FINAL_FREQ_BINS, DOWNSAMPLE_FACTOR).mean(axis=3)
    else:
        snippet_ds = snippet_data
    
    # Calculate relative frequency axis (Hz from center)
    foff = metadata.get('foff', -2.79e-6)  # MHz per channel
    channel_width_hz = abs(foff) * 1e6 * DOWNSAMPLE_FACTOR  # Hz per downsampled bin
    half_bins = FINAL_FREQ_BINS // 2
    freq_axis_hz = np.arange(-half_bins, half_bins) * channel_width_hz
    
    # Time axis
    tsamp = metadata.get('tsamp', 1.0)
    n_time = snippet_ds.shape[1]
    time_axis = np.arange(n_time) * tsamp
    
    # Create figure
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(8, 1, height_ratios=[0.8, 0.05, 1, 1, 1, 1, 1, 1], hspace=0.02)
    
    # === Header Panel ===
    ax_header = fig.add_subplot(gs[0])
    ax_header.axis('off')
    
    # Header text
    header_left = f"""Target name: {target_name}
Source: {metadata.get('source_name', 'Unknown')}
Telescope: SRT
Observation start (MJD): {metadata.get('tstart', 'N/A'):.6f}
Cadence type: ABACAD"""
    
    header_right = f"""Center frequency: {freq_mhz:.6f} MHz
Event bandwidth: {channel_width_hz * FINAL_FREQ_BINS:.2f} Hz
Confidence: {probability*100:.1f}%
Candidate ID: {candidate_id or 'N/A'}"""
    
    ax_header.text(0.02, 0.95, "Candidate Information", fontsize=14, fontweight='bold',
                   transform=ax_header.transAxes, va='top')
    ax_header.text(0.02, 0.75, header_left, fontsize=10, transform=ax_header.transAxes,
                   va='top', family='monospace')
    ax_header.text(0.55, 0.75, header_right, fontsize=10, transform=ax_header.transAxes,
                   va='top', family='monospace')
    
    # === Waterfall Panels ===
    obs_labels = ['A (ON)', 'B (OFF)', 'A (ON)', 'C (OFF)', 'A (ON)', 'D (OFF)']
    
    axes = []
    for i in range(6):
        ax = fig.add_subplot(gs[i + 2])
        axes.append(ax)
        
        # Log scale and normalize
        data = snippet_ds[i].astype(np.float64)
        data = np.log10(np.abs(data) + 1e-10)
        
        # Plot
        img = ax.imshow(data, aspect='auto', cmap='hot', origin='upper',
                       extent=[freq_axis_hz[0], freq_axis_hz[-1], time_axis[-1], time_axis[0]])
        
        # Labels
        ax.set_ylabel('Time [s]', fontsize=9)
        ax.text(0.02, 0.85, f'Obs {i+1} - {obs_labels[i]}', transform=ax.transAxes,
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='square', facecolor='black', alpha=0.7))
        
        if i < 5:
            ax.tick_params(labelbottom=False)
    
    # X-axis label on bottom panel
    axes[5].set_xlabel(f'Relative frequency in Hz from {freq_mhz:.6f} MHz', fontsize=10)
    
    # Title
    fig.suptitle(f'Waterfall plot', fontsize=12, fontweight='bold', y=0.62)
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {Path(output_path).name}")
    
    plt.close()
    return fig


def plot_all_paper_style(file_paths: List[str],
                         candidates: List[dict],
                         target_name: str,
                         output_dir: str,
                         max_plots: int = None):
    """Plot all candidates in paper style."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_to_plot = min(len(candidates), max_plots) if max_plots else len(candidates)
    print(f"\n📊 Creating {n_to_plot} paper-style plots...")
    
    # Sort by probability
    sorted_cands = sorted(candidates, key=lambda x: float(x['probability']), reverse=True)
    
    for i, cand in enumerate(sorted_cands[:n_to_plot], 1):
        freq = float(cand['freq_mhz'])
        prob = float(cand['probability'])
        center = int(cand['center_channel'])
        
        output_path = output_dir / f"candidate_{i:03d}_{freq:.2f}MHz_P{prob:.3f}_paper.png"
        
        plot_candidate_paper_style(
            file_paths=file_paths,
            center_channel=center,
            freq_mhz=freq,
            probability=prob,
            target_name=target_name,
            output_path=str(output_path),
            candidate_id=f"{i:04d}",
            total_candidates=len(candidates)
        )
    
    print(f"\n✅ {n_to_plot} paper-style plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create paper-style candidate visualizations')
    parser.add_argument('--target-dir', '-d', required=True,
                        help='Target result directory with metadata.json and candidates.csv')
    parser.add_argument('--max-plots', '-n', type=int, default=10,
                        help='Maximum number of candidates to plot')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory (default: target_dir/plots_paper)')
    
    args = parser.parse_args()
    
    target_dir = Path(args.target_dir)
    metadata_path = target_dir / "metadata.json"
    candidates_path = target_dir / "candidates.csv"
    
    if not metadata_path.exists():
        print(f"Error: metadata.json not found in {target_dir}")
        return
    if not candidates_path.exists():
        print(f"Error: candidates.csv not found in {target_dir}")
        return
    
    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    target_name = metadata['target']
    file_paths = metadata['files']
    
    # Load candidates
    candidates = []
    with open(candidates_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(row)
    
    print(f"Target: {target_name}")
    print(f"Loaded {len(candidates)} candidates")
    
    # Validate files
    for f in file_paths:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            return
    
    output_dir = args.output_dir or str(target_dir / "plots_paper")
    
    plot_all_paper_style(
        file_paths=file_paths,
        candidates=candidates,
        target_name=target_name,
        output_dir=output_dir,
        max_plots=args.max_plots
    )


if __name__ == '__main__':
    main()
