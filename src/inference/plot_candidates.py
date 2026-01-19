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
from ..utils.preprocessing import preprocess, downscale

# Constants (must match training/inference pipeline)
SNIPPET_WIDTH = 4096
DOWNSAMPLE_FACTOR = 8
FINAL_FREQ_BINS = 512


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


def load_snippet_from_files(file_paths: List[str], center_channel: int, width: int = 4096) -> np.ndarray:
    """Load only the required frequency slice from HDF5 files.
    
    This is MUCH faster than loading the entire file when we only need a small slice.
    """
    import h5py
    
    snippets = []
    start = max(0, center_channel - width // 2)
    end = start + width
    
    for path in file_paths:
        with h5py.File(path, 'r') as f:
            # Slice DIRECTLY on the HDF5 dataset - only loads the slice, not the whole file
            data = f['data'][:, 0, start:end]  # (time, freq_slice)
            snippets.append(data)
    
    return np.array(snippets)  # (6, time, freq_slice)


def plot_all_candidates(file_paths: List[str],
                       candidates: List[dict],
                       output_dir: str,
                       max_plots: int = 10,
                       raw_mode: bool = False):
    """
    Plot all candidates from a cadence using LAZY LOADING.
    Only loads the frequency slice needed for each candidate instead of the entire file.
    
    Args:
        raw_mode: If True, show log-scaled data without per-observation normalization
    """
    n_to_plot = min(len(candidates), max_plots) if max_plots else len(candidates)
    mode_str = "raw" if raw_mode else "preprocessed"
    print(f"\n📊 Plotting {n_to_plot} candidates ({mode_str}, lazy loading)...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by probability (highest first)
    sorted_candidates = sorted(candidates, key=lambda x: x['probability'], reverse=True)
    
    # Plot each candidate - load only the slice needed
    for i, cand in enumerate(sorted_candidates[:max_plots]):
        # Load only the 4096 channels around the candidate
        snippet_data = load_snippet_from_files(file_paths, cand['center_channel'], width=4096)
        
        # Downscale using the SAME function as training/inference
        if snippet_data.shape[2] >= 4096:
            snippet_ds = downscale(snippet_data, factor=8)  # (6, 16, 512)
        else:
            snippet_ds = snippet_data
        
        if raw_mode:
            # Raw mode: just log scale, global normalization for visibility
            all_data = np.log(np.abs(snippet_ds) + 1e-10)
            vmin, vmax = np.percentile(all_data, [1, 99])  # Clip outliers
            processed = all_data
            cbar_label = 'Log Power (raw)'
        else:
            # Preprocessed mode: use EXACT same preprocessing as training/inference
            # Add batch dim, preprocess, remove batch dim
            processed = preprocess(snippet_ds[np.newaxis, ...], add_channel=False)[0]
            vmin, vmax = 0, 1
            cbar_label = 'Log-Normalized Power [0-1]'
        
        # Create figure
        fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
        fig.subplots_adjust(hspace=0)
        
        for j in range(6):
            ax = axes[j]
            obs_type = "ON" if j % 2 == 0 else "OFF"
            
            img = ax.imshow(processed[j], aspect='auto', cmap='hot', origin='upper',
                            extent=[0, FINAL_FREQ_BINS, processed[j].shape[0], 0],
                            vmin=vmin, vmax=vmax)
            
            ax.text(0.02, 0.85, f'Obs {j+1} ({obs_type})', transform=ax.transAxes, 
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='square', facecolor='black', alpha=0.5))
            
            ax.set_ylabel('Time bin')
            
            if j < 5:
                ax.tick_params(labelbottom=False)
        
        axes[5].set_xlabel('Frequency bins (512 - After 8x downsampling)')
        
        cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label(cbar_label)
        
        mode_suffix = "_raw" if raw_mode else ""
        
        fig.suptitle(f'ETI Candidate: {cand["freq_mhz"]:.4f} MHz - P(ETI)={cand["probability"]:.3f}', 
                     fontsize=14, fontweight='bold')
        
        filename_proc = f"candidate_{i+1}_{cand['freq_mhz']:.2f}MHz_P{cand['probability']:.3f}{mode_suffix}.png"
        plt.savefig(output_dir / filename_proc, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename_proc}")
        plt.close()
    
    # Create summary plot
    plot_summary(candidates, output_dir / "candidates_summary.png")
    
    print(f"\n✅ {n_to_plot} plots saved to {output_dir}")


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
    parser.add_argument('--cadence', '-c', nargs=6, default=None,
                        help='6 observation files in order [ON,OFF,ON,OFF,ON,OFF]')
    parser.add_argument('--candidates', '-i', default=None,
                        help='CSV file with candidates')
    parser.add_argument('--output-dir', '-o', default='results/plots/candidates',
                        help='Output directory for plots')
    parser.add_argument('--max-plots', '-n', type=int, default=10,
                        help='Maximum number of candidates to plot')
    parser.add_argument('--target-dir', '-d', type=str, default=None,
                        help='Target result directory (reads metadata.json for file paths)')
    parser.add_argument('--inference-dir', '-I', type=str, default=None,
                        help='Inference results directory (plots ALL targets with candidates)')
    parser.add_argument('--raw', '-r', action='store_true',
                        help='Plot raw log-scaled data without per-observation normalization')
    
    args = parser.parse_args()
    
    # Batch mode: process all targets in inference directory
    if args.inference_dir:
        inference_dir = Path(args.inference_dir)
        if not inference_dir.exists():
            print(f"Error: Directory not found: {inference_dir}")
            return
        
        import json
        
        # Find all target directories with candidates
        target_dirs = []
        for subdir in inference_dir.iterdir():
            if subdir.is_dir():
                metadata_path = subdir / "metadata.json"
                candidates_path = subdir / "candidates.csv"
                if metadata_path.exists() and candidates_path.exists():
                    target_dirs.append(subdir)
        
        print(f"Found {len(target_dirs)} targets with candidates in {inference_dir}")
        
        if not target_dirs:
            print("No targets with candidates found.")
            return
        
        # Process each target
        plotted = 0
        skipped = 0
        for i, target_dir in enumerate(sorted(target_dirs), 1):
            target_name = target_dir.name
            print(f"\n[{i}/{len(target_dirs)}] {target_name}...")
            
            metadata_path = target_dir / "metadata.json"
            candidates_path = target_dir / "candidates.csv"
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            file_paths = metadata['files']
            
            # Check if files exist
            missing = [p for p in file_paths if not Path(p).exists()]
            if missing:
                print(f"  ⚠️ Skipping: {len(missing)} files missing")
                skipped += 1
                continue
            
            candidates = load_candidates(str(candidates_path))
            if not candidates:
                print(f"  ⚠️ Skipping: no candidates")
                skipped += 1
                continue
            
            output_dir = target_dir / "plots"
            
            print(f"  Plotting {len(candidates)} candidates...")
            plot_all_candidates(
                file_paths=file_paths,
                candidates=candidates,
                output_dir=str(output_dir),
                max_plots=args.max_plots,
                raw_mode=args.raw
            )
            plotted += 1
        
        print(f"\n{'='*50}")
        print(f"BATCH PLOTTING COMPLETE")
        print(f"  Targets plotted: {plotted}")
        print(f"  Targets skipped: {skipped}")
        return
    
    # If target-dir is specified, load metadata
    if args.target_dir:
        target_dir = Path(args.target_dir)
        metadata_path = target_dir / "metadata.json"
        candidates_path = target_dir / "candidates.csv"
        
        if not metadata_path.exists():
            print(f"Error: metadata.json not found in {target_dir}")
            return
        
        if not candidates_path.exists():
            print(f"Error: candidates.csv not found in {target_dir}")
            return
        
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        file_paths = metadata['files']
        candidates = load_candidates(str(candidates_path))
        output_dir = args.output_dir or str(target_dir / "plots")
        
        print(f"Target: {metadata['target']}")
        print(f"Loaded {len(candidates)} candidates")
        
        # Validate files
        for f in file_paths:
            if not Path(f).exists():
                print(f"Error: File not found: {f}")
                return
        
        plot_all_candidates(
            file_paths=file_paths,
            candidates=candidates,
            output_dir=output_dir,
            max_plots=args.max_plots,
            raw_mode=args.raw
        )
        return
    
    # Original behavior with --cadence and --candidates
    if not args.cadence or not args.candidates:
        print("Error: Either --inference-dir, --target-dir, OR (--cadence AND --candidates) required")
        return
    
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
        max_plots=args.max_plots,
        raw_mode=args.raw
    )


if __name__ == '__main__':
    main()
