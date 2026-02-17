#!/usr/bin/env python3
"""
Cross-target frequency search.

Check if a signal at a given frequency appears in other targets.
If a signal appears at the same frequency across multiple targets 
(different sky positions), it's likely RFI, not ETI.

Usage:
    python scripts/search_frequency_cross_target.py \
        --freq 4950.027 \
        --results results/inference \
        --tolerance 0.001
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict


def search_frequency(results_dir: Path, 
                     target_freq_mhz: float, 
                     tolerance_mhz: float = 0.001) -> List[Dict]:
    """
    Search for signals at a specific frequency across all targets.
    
    Args:
        results_dir: Directory containing inference results
        target_freq_mhz: Frequency to search for (MHz)
        tolerance_mhz: Tolerance window (MHz), default ±1 kHz
        
    Returns:
        List of matches with target, freq, probability info
    """
    matches = []
    
    for target_dir in sorted(results_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        
        csv_path = target_dir / "candidates.csv"
        if not csv_path.exists():
            continue
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
            continue
        
        if 'freq_mhz' not in df.columns:
            continue
            
        target_name = target_dir.name
        
        # Search for frequencies within tolerance
        mask = abs(df['freq_mhz'] - target_freq_mhz) < tolerance_mhz
        
        for _, row in df[mask].iterrows():
            matches.append({
                'target': target_name,
                'freq_mhz': row['freq_mhz'],
                'prob': row.get('eti_probability', row.get('probability', 0)),
                'offset_hz': (row['freq_mhz'] - target_freq_mhz) * 1e6
            })
    
    return matches


def main():
    parser = argparse.ArgumentParser(
        description='Search for signals at a specific frequency across all targets'
    )
    parser.add_argument('--freq', '-f', type=float, required=True,
                        help='Frequency to search for (MHz)')
    parser.add_argument('--results', '-r', type=str, required=True,
                        help='Inference results directory')
    parser.add_argument('--tolerance', '-t', type=float, default=0.001,
                        help='Tolerance in MHz (default: 0.001 = 1 kHz)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    target_freq = args.freq
    tolerance = args.tolerance
    
    print(f"Searching for signals at {target_freq:.6f} MHz (±{tolerance*1e3:.1f} kHz)")
    print("-" * 60)
    
    matches = search_frequency(results_dir, target_freq, tolerance)
    
    if not matches:
        print("❌ No matches found")
        return
    
    # Group by target and sort by probability
    matches_sorted = sorted(matches, key=lambda x: x['prob'], reverse=True)
    
    for m in matches_sorted:
        print(f"{m['target']:30s} | {m['freq_mhz']:.6f} MHz | P={m['prob']:.3f}")
    
    print("-" * 60)
    
    unique_targets = set(m['target'] for m in matches)
    n_targets = len(unique_targets)
    
    print(f"Found in {n_targets} different target(s)")
    
    if n_targets > 1:
        print("Signal appears in MULTIPLE targets → Likely RFI")
    else:
        print("Signal appears in only 1 target → Potentially interesting!")


if __name__ == '__main__':
    main()
