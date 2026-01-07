#!/usr/bin/env python3
"""
Analyze all cadences in a directory and report frequency bands.
Creates a summary plot showing the frequency coverage.
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict
import warnings
import argparse

import numpy as np
import matplotlib.pyplot as plt
from blimpy import Waterfall

# Suppress blimpy warnings
warnings.filterwarnings('ignore')


def extract_timestamp(filename: str) -> int:
    """
    Extract timestamp from GUPPI filename for proper sorting.
    
    Format: guppi_MJD_TIMESTAMP_ID_SOURCE_ON/OFF_0001.0000.h5
    or:     blc01_guppi_MJD_TIMESTAMP_ID_SOURCE_ON/OFF_0001.0000.h5
    
    Returns the TIMESTAMP field as integer for sorting.
    """
    # Match patterns like guppi_60703_20682_... or blc01_guppi_59368_25667_...
    match = re.search(r'guppi_(\d+)_(\d+)_', filename)
    if match:
        mjd = int(match.group(1))
        timestamp = int(match.group(2))
        # Combine MJD and timestamp for unique sorting
        return mjd * 1000000 + timestamp
    return 0


def sort_files_natural(files: list) -> list:
    """Sort files by their timestamp for proper cadence ordering."""
    return sorted(files, key=lambda f: extract_timestamp(f.name))


def get_file_info(filepath: str) -> dict:
    """Extract frequency info from a single file."""
    try:
        wf = Waterfall(filepath, load_data=False)
        header = wf.header
        
        fch1 = header['fch1']
        foff = header['foff']
        nchans = header['nchans']
        
        # Calculate frequency range
        if foff < 0:
            f_start = fch1 + foff * nchans
            f_end = fch1
        else:
            f_start = fch1
            f_end = fch1 + foff * nchans
        
        return {
            'source': header.get('source_name', 'Unknown'),
            'f_start': min(f_start, f_end),
            'f_end': max(f_start, f_end),
            'nchans': nchans,
            'bandwidth_mhz': abs(f_end - f_start),
            'freq_resolution_hz': abs(foff) * 1e6,
            'tsamp': header.get('tsamp', 0),
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def find_cadences(directory: Path) -> list:
    """
    Find all cadences (groups of 6 files) in directory.
    Groups files by source name and observation time.
    """
    h5_files = list(directory.rglob("*.h5"))  # Recursive search
    
    if not h5_files:
        return []
    
    # Group by base pattern (remove observation number)
    cadence_groups = defaultdict(list)
    
    for f in sort_files_natural(h5_files):
        # Try to extract source name from file
        info = get_file_info(str(f))
        if info['success']:
            source = info['source']
            # Group by source
            cadence_groups[source].append(f)
    
    # Split into cadences of 6
    cadences = []
    for source, files in cadence_groups.items():
        # Sort files
        files = sort_files_natural(files)
        # Take groups of 6
        for i in range(0, len(files) - 5, 6):
            cadences.append({
                'source': source,
                'files': files[i:i+6]
            })
    
    return cadences


def analyze_directory(directory: Path, output_dir: Path = None) -> dict:
    """Analyze all cadences in directory - only reads FIRST file of each cadence."""
    
    print("=" * 70)
    print("CADENCE FREQUENCY ANALYSIS")
    print("=" * 70)
    print(f"\nDirectory: {directory}")
    
    # Find all h5 files recursively
    h5_files = sort_files_natural(list(directory.rglob("*.h5")))
    n_total_files = len(h5_files)
    print(f"Found {n_total_files} .h5 files (searching recursively)")
    
    if not h5_files:
        print("No .h5 files found!")
        return {}
    
    # Group files by parent directory
    dir_groups = defaultdict(list)
    for f in h5_files:
        dir_groups[f.parent].append(f)
    
    print(f"Found files in {len(dir_groups)} directories")
    
    # Identify cadences: take every 6 files as a cadence, analyze only the FIRST
    cadences_info = []
    skipped_count = 0
    
    print("\n" + "-" * 70)
    print("ANALYZING CADENCES (first file only)")
    print("-" * 70)
    
    invalid_count = 0
    
    for dir_path, files in sorted(dir_groups.items()):
        files = sort_files_natural(files)  # Sort by timestamp
        
        print(f"\n📁 Directory: {dir_path}")
        print(f"   Files: {len(files)}")
        
        # Group files by source (without ON/OFF suffix)
        source_groups = defaultdict(list)
        for f in files:
            # Extract source name without ON/OFF
            name = f.name
            if '_ON_' in name:
                source = name.split('_ON_')[0].split('_')[-1]
                obs_type = 'ON'
            elif '_OFF_' in name:
                source = name.split('_OFF_')[0].split('_')[-1]
                obs_type = 'OFF'
            else:
                source = 'UNKNOWN'
                obs_type = 'UNKNOWN'
            
            source_groups[source].append((f, obs_type, extract_timestamp(f.name)))
        
        # Process each source
        for source, obs_list in source_groups.items():
            # Sort by timestamp
            obs_list.sort(key=lambda x: x[2])
            
            # Check if we have exactly 6 files with ON/OFF/ON/OFF/ON/OFF pattern
            if len(obs_list) != 6:
                invalid_count += 1
                print(f"\n⚠️  INCOMPLETE: {source} has {len(obs_list)} files (need 6)")
                continue
            
            # Check pattern: should be ON, OFF, ON, OFF, ON, OFF
            expected = ['ON', 'OFF', 'ON', 'OFF', 'ON', 'OFF']
            actual = [obs[1] for obs in obs_list]
            
            if actual != expected:
                invalid_count += 1
                print(f"\n⚠️  INVALID PATTERN: {source}")
                print(f"   Expected: {expected}")
                print(f"   Got:      {actual}")
                continue
            
            # Valid cadence! Analyze first file
            first_file = obs_list[0][0]
            info = get_file_info(str(first_file))
            
            if info['success']:
                info['filename'] = first_file.name
                info['directory'] = str(dir_path.relative_to(directory) if dir_path != directory else '.')
                info['source_name'] = source
                cadences_info.append(info)
                
                print(f"\n[Cadence {len(cadences_info)}] {source}:")
                print(f"  First file: {first_file.name}")
                print(f"  Freq range: {info['f_start']:.6f} - {info['f_end']:.6f} MHz")
                print(f"  Bandwidth: {info['bandwidth_mhz']:.3f} MHz")
            else:
                skipped_count += 1
                print(f"\n⚠️  CORRUPT: {first_file.name}")
                print(f"   Error: {info.get('error', 'Unknown error')}")
    
    n_valid = len(cadences_info)
    n_total = n_valid + skipped_count + invalid_count
    print(f"\n\nTotal cadences: {n_total}")
    print(f"  ✅ Valid: {n_valid}")
    print(f"  ⚠️  Invalid/Incomplete: {invalid_count}")
    print(f"  ❌ Corrupt: {skipped_count}")
    
    # Summary by source
    print("\n" + "=" * 70)
    print("SUMMARY BY SOURCE")
    print("=" * 70)
    
    source_summary = {}
    
    # Group by source + frequency band (different receivers = different entries)
    for c in cadences_info:
        # Create key using source + rounded frequency to group nearby bands
        source = c['source']
        freq_band = f"{c['f_start']:.0f}-{c['f_end']:.0f}"
        key = f"{source} @ {freq_band} MHz"
        
        if key not in source_summary:
            source_summary[key] = {
                'source': source,
                'n_files': 0,
                'f_min': c['f_start'],
                'f_max': c['f_end'],
                'bandwidth': c['bandwidth_mhz'],
                'cadences': []
            }
        
        source_summary[key]['n_files'] += 6
        source_summary[key]['cadences'].append(c)
    
    for key in sorted(source_summary.keys()):
        data = source_summary[key]
        print(f"\n📡 {key}:")
        print(f"   Bandwidth: {data['bandwidth']:.3f} MHz")
    
    # Create visualization
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        create_frequency_plot(source_summary, output_dir / "frequency_coverage.png")
        create_summary_table(source_summary, output_dir / "cadence_summary.csv")
    
    return source_summary


def create_frequency_plot(source_summary: dict, output_path: Path):
    """Create a professional summary plot showing frequency bands for observatory presentation."""
    
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)
    
    n_sources = len(source_summary)
    
    if n_sources == 0:
        print("No sources to plot!")
        return
    
    # Collect frequency band data
    bands = {}
    for source, data in source_summary.items():
        f_min = data['f_min']
        f_max = data['f_max']
        band_key = f"{f_min/1000:.2f}-{f_max/1000:.2f} GHz"
        
        if band_key not in bands:
            bands[band_key] = {
                'f_min': f_min,
                'f_max': f_max,
                'bandwidth_mhz': f_max - f_min,
                'n_sources': 0,
                'n_cadences': 0
            }
        bands[band_key]['n_sources'] += 1
        bands[band_key]['n_cadences'] += len(data.get('cadences', []))
    
    # Sort bands by frequency
    sorted_bands = sorted(bands.items(), key=lambda x: x[1]['f_min'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ========== LEFT: Frequency Band Overview ==========
    ax1 = axes[0]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_bands)))
    
    for i, (band_name, band_data) in enumerate(sorted_bands):
        f_start = band_data['f_min']
        f_width = band_data['bandwidth_mhz']
        n_cad = band_data['n_cadences']
        
        bar = ax1.barh(i, f_width, left=f_start, height=0.7, 
                       color=colors[i], edgecolor='black', linewidth=1.5)
        
        # Add label inside bar
        label = f"{band_data['f_min']:.0f} - {band_data['f_max']:.0f} MHz"
        ax1.text(f_start + f_width/2, i, label,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='black')
    
    ax1.set_yticks(range(len(sorted_bands)))
    ax1.set_yticklabels([f"Band {i+1}" for i in range(len(sorted_bands))], fontsize=11)
    ax1.set_xlabel('Frequency (MHz)', fontsize=12, fontweight='bold')
    ax1.set_title('Frequency Bands Coverage', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(min(b['f_min'] for b in bands.values()) - 500,
                 max(b['f_max'] for b in bands.values()) + 500)
    
    # ========== RIGHT: Summary Table ==========
    ax2 = axes[1]
    ax2.axis('off')
    
    # Total statistics
    total_sources = sum(b['n_sources'] for b in bands.values())
    total_cadences = sum(b['n_cadences'] for b in bands.values())
    f_global_min = min(b['f_min'] for b in bands.values())
    f_global_max = max(b['f_max'] for b in bands.values())
    
    # Create table data
    table_data = []
    for i, (band_name, band_data) in enumerate(sorted_bands):
        table_data.append([
            f"Band {i+1}",
            f"{band_data['f_min']:.1f} MHz",
            f"{band_data['f_max']:.1f} MHz",
            f"{band_data['bandwidth_mhz']:.1f} MHz",
            str(band_data['n_cadences'])
        ])
    
    # Add totals row
    table_data.append([
        "TOTAL",
        f"{f_global_min:.1f} MHz",
        f"{f_global_max:.1f} MHz",
        "-",
        str(total_cadences)
    ])
    
    table = ax2.table(
        cellText=table_data,
        colLabels=['Band', 'Start Freq', 'End Freq', 'Bandwidth', 'Cadences'],
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.22, 0.22, 0.22, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style totals row
    n_rows = len(table_data)
    for i in range(5):
        table[(n_rows, i)].set_facecolor('#E2EFDA')
        table[(n_rows, i)].set_text_props(fontweight='bold')
    
    # Color band rows to match bars
    for i, (band_name, _) in enumerate(sorted_bands):
        for j in range(5):
            table[(i+1, j)].set_facecolor(colors[i])
            table[(i+1, j)].set_alpha(0.5)
    
    ax2.set_title('Frequency Bands Summary', fontsize=14, fontweight='bold', pad=15)
    
    # Add footer with total info
    footer_text = f"Total: {total_cadences} valid cadences across {len(sorted_bands)} frequency bands"
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=12, style='italic')
    
    plt.suptitle('GBT Observation Frequency Coverage Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved plot to {output_path}")
    
    plt.close()


def create_summary_table(source_summary: dict, output_path: Path):
    """Save summary to CSV."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'n_files', 'n_cadences', 'f_min_mhz', 'f_max_mhz', 'bandwidth_mhz'])
        
        for source, data in sorted(source_summary.items()):
            bandwidth = data.get('bandwidth', data['f_max'] - data['f_min'])
            writer.writerow([
                source,
                data['n_files'],
                len(data.get('cadences', [])),
                f"{data['f_min']:.6f}",
                f"{data['f_max']:.6f}",
                f"{bandwidth:.6f}"
            ])
    
    print(f"✓ Saved summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze cadence frequency coverage")
    parser.add_argument('directory', type=str, help='Directory containing .h5 files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results (default: same as input)')
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else directory
    
    analyze_directory(directory, output_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
