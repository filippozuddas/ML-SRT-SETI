#!/usr/bin/env python3
"""
Azimuth/Elevation plot for RFI characterization.

Computes telescope pointing (Az/El) for each observed target using the 
RA/Dec and observation time from file headers, then generates a scatter 
plot to help determine if a detected signal is position-dependent RFI.

Since the GBT filterbank headers have az_start=0 and za_start=0 (not populated),
we compute Az/El from src_raj, src_dej, and tstart using astropy.

Usage:
    # With config JSON (same format as cross_target_may2024.json)
    python experiments/plot_azimuth_elevation.py \
        --config configs/all_targets.json \
        --output azimuth_elevation.png

    # With inference results directory (reads metadata.json for file paths)
    python experiments/plot_azimuth_elevation.py \
        --results-dir results/inference \
        --output azimuth_elevation.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from blimpy import Waterfall
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import warnings

warnings.filterwarnings('ignore')

# GBT (Green Bank Telescope) location
GBT_LOCATION = EarthLocation(
    lat=38.4331 * u.deg,
    lon=-79.8397 * u.deg,
    height=824.6 * u.m
)

# SRT (Sardinia Radio Telescope) location - if needed
SRT_LOCATION = EarthLocation(
    lat=39.4930 * u.deg,
    lon=9.2451 * u.deg,
    height=600.0 * u.m
)

# Map telescope_id to location
TELESCOPE_LOCATIONS = {
    10: ("GBT", GBT_LOCATION),
    # Add SRT telescope_id here if known
}


def extract_pointing_info(filepath: str) -> dict:
    """
    Extract pointing information from a single observation file header.
    
    Returns dict with: target_name, ra_deg, dec_deg, az, el, tstart, mjd
    """
    wf = Waterfall(str(filepath), load_data=False)
    header = wf.header
    
    # Extract RA/Dec
    src_raj = header.get('src_raj')
    src_dej = header.get('src_dej')
    tstart = header.get('tstart', 0)
    telescope_id = header.get('telescope_id', 10)
    source_name = header.get('source_name', 'Unknown')
    
    # Convert RA from hourangle to degrees
    if hasattr(src_raj, 'deg'):
        ra_deg = src_raj.deg  # Already an Angle object
    elif hasattr(src_raj, 'value'):
        ra_deg = src_raj.value * 15.0  # hourangle to degrees
    else:
        ra_deg = float(src_raj) * 15.0  # hourangle to degrees
        
    # Convert Dec to degrees
    if hasattr(src_dej, 'deg'):
        dec_deg = src_dej.deg
    elif hasattr(src_dej, 'value'):
        dec_deg = src_dej.value
    else:
        dec_deg = float(src_dej)
    
    # Get telescope location
    tel_name, tel_location = TELESCOPE_LOCATIONS.get(
        telescope_id, ("GBT", GBT_LOCATION)
    )
    
    # Compute Az/El using astropy
    obs_time = Time(tstart, format='mjd')
    target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    altaz_frame = AltAz(obstime=obs_time, location=tel_location)
    altaz = target_coord.transform_to(altaz_frame)
    
    return {
        'source_name': source_name.replace('_ON', '').replace('_OFF', ''),
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'azimuth': altaz.az.deg,
        'elevation': altaz.alt.deg,
        'mjd': tstart,
        'utc': obs_time.iso,
        'telescope': tel_name,
    }


def load_targets_from_config(config_path: str) -> dict:
    """Load target file paths from JSON config."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_targets_from_results(results_dir: str) -> dict:
    """Load target file paths from inference results directory."""
    results_path = Path(results_dir)
    targets = {}
    
    for target_dir in sorted(results_path.iterdir()):
        if not target_dir.is_dir():
            continue
        
        metadata_path = target_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            targets[metadata.get('target', target_dir.name)] = metadata['files']
        else:
            # Try to find h5 files directly
            h5_files = sorted(target_dir.glob("*.h5"))
            if h5_files:
                targets[target_dir.name] = [str(f) for f in h5_files[:6]]
    
    return targets


def plot_azimuth_elevation(target_data: list, output_path: str = None, 
                           title: str = None):
    """
    Create Azimuth vs Elevation scatter plot.
    
    Args:
        target_data: List of dicts from extract_pointing_info
        output_path: Optional path to save figure
        title: Optional custom title
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # --- Plot 1: Cartesian Az vs El ---
    ax1 = axes[0]
    
    azimuths = [d['azimuth'] for d in target_data]
    elevations = [d['elevation'] for d in target_data]
    names = [d['source_name'] for d in target_data]
    
    # Color by observation time (MJD)
    mjds = [d['mjd'] for d in target_data]
    
    scatter = ax1.scatter(azimuths, elevations, 
                          c=mjds, cmap='viridis',
                          s=150, edgecolors='black', linewidths=0.8,
                          zorder=5)
    
    # Annotate each point with target name
    for i, name in enumerate(names):
        # Shorter name for display
        short_name = name.replace('TIC', '').replace('HIP', '')
        ax1.annotate(short_name, 
                     (azimuths[i], elevations[i]),
                     textcoords="offset points",
                     xytext=(8, 8),
                     fontsize=8,
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', 
                              facecolor='white', alpha=0.7, edgecolor='gray'),
                     zorder=10)
    
    cbar = plt.colorbar(scatter, ax=ax1, label='Observation Time (MJD)')
    
    ax1.set_xlabel('Azimuth (°)', fontsize=12)
    ax1.set_ylabel('Elevation (°)', fontsize=12)
    ax1.set_title('Target Pointing Positions', fontsize=13, fontweight='bold')
    ax1.set_xlim(-5, 365)
    ax1.set_ylim(0, 95)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Horizon limit (5°)')
    ax1.legend(fontsize=9)
    
    # --- Plot 2: Polar plot (more intuitive) ---
    ax2 = axes[1]
    ax2.remove()
    ax2 = fig.add_subplot(122, projection='polar')
    
    # In polar: theta = azimuth (radians), r = zenith angle (90 - elevation)
    theta = np.radians(azimuths)
    r = [90 - el for el in elevations]  # zenith distance
    
    scatter2 = ax2.scatter(theta, r,
                            c=mjds, cmap='viridis',
                            s=150, edgecolors='black', linewidths=0.8,
                            zorder=5)
    
    # Annotate
    for i, name in enumerate(names):
        short_name = name.replace('TIC', '').replace('HIP', '')
        ax2.annotate(short_name,
                     (theta[i], r[i]),
                     textcoords="offset points",
                     xytext=(8, 8),
                     fontsize=8,
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.7, edgecolor='gray'),
                     zorder=10)
    
    ax2.set_theta_zero_location('N')  # North at top
    ax2.set_theta_direction(-1)       # Clockwise
    ax2.set_rlim(0, 90)
    ax2.set_rlabel_position(22.5)
    ax2.set_title('Sky View (from telescope)\nCenter = Zenith, Edge = Horizon',
                   fontsize=11, fontweight='bold', pad=20)
    
    # Set radial labels to show elevation instead of zenith distance
    ax2.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax2.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'])
    
    plt.colorbar(scatter2, ax=ax2, label='MJD', pad=0.1, shrink=0.8)
    
    # Overall title
    plot_title = title or 'Azimuth / Elevation of Observed Targets'
    fig.suptitle(plot_title, fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compute and plot Azimuth/Elevation for observed targets'
    )
    
    # Input sources (at least one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--config', '-c', type=str,
                              help='JSON config with target file paths')
    input_group.add_argument('--results-dir', '-r', type=str,
                              help='Inference results directory')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output image path (e.g., azimuth_elevation.png)')
    parser.add_argument('--title', '-t', type=str, default=None,
                        help='Custom plot title')
    parser.add_argument('--print-table', action='store_true',
                        help='Print a table of all target coordinates')
    
    args = parser.parse_args()
    
    # Load targets
    if args.config:
        targets = load_targets_from_config(args.config)
    else:
        targets = load_targets_from_results(args.results_dir)
    
    print(f"📡 Processing {len(targets)} targets...")
    print("=" * 70)
    
    # Extract pointing info for each target (use first ON file)
    all_data = []
    
    for target_name, files in targets.items():
        # Use the first file (ON observation) for pointing info
        first_file = files[0]
        
        if not Path(first_file).exists():
            print(f"  ⚠️  {target_name}: File not found ({first_file})")
            continue
        
        try:
            info = extract_pointing_info(first_file)
            info['source_name'] = target_name  # Override with config name
            all_data.append(info)
            
            print(f"  ✅ {target_name:25s} | "
                  f"Az={info['azimuth']:7.2f}° | "
                  f"El={info['elevation']:6.2f}° | "
                  f"RA={info['ra_deg']:8.4f}° | "
                  f"Dec={info['dec_deg']:+7.3f}° | "
                  f"{info['utc']}")
        except Exception as e:
            print(f"  ❌ {target_name}: {e}")
    
    print("=" * 70)
    print(f"📊 Successfully processed {len(all_data)}/{len(targets)} targets")
    
    if not all_data:
        print("❌ No data to plot!")
        return
    
    # Print summary table
    if args.print_table:
        print("\n" + "=" * 90)
        print(f"{'Target':25s} | {'Az (°)':>8s} | {'El (°)':>8s} | {'RA (°)':>9s} | {'Dec (°)':>9s} | {'MJD':>14s}")
        print("-" * 90)
        for d in sorted(all_data, key=lambda x: x['azimuth']):
            print(f"{d['source_name']:25s} | {d['azimuth']:8.2f} | {d['elevation']:8.2f} | "
                  f"{d['ra_deg']:9.4f} | {d['dec_deg']:+9.3f} | {d['mjd']:.6f}")
        print("=" * 90)
    
    # Generate plot
    output_path = args.output or "azimuth_elevation.png"
    plot_azimuth_elevation(all_data, output_path, title=args.title)


if __name__ == '__main__':
    main()
