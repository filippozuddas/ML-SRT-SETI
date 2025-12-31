#!/usr/bin/env python3
"""
SRT Dataset Builder.

Extracts RAW background snippets from SRT HDF5 observation files
for use with CadenceGenerator in training.
"""

import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json
import h5py
from tqdm import tqdm
import warnings


@dataclass
class CadenceInfo:
    """Information about a cadence (6 ON/OFF files)."""
    target_name: str
    date: str
    files: List[Path] = field(default_factory=list)
    n_channels: int = 0
    freq_start: float = 0.0
    freq_end: float = 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if cadence has all 6 files."""
        return len(self.files) == 6
    
    @property
    def n_snippets(self) -> int:
        """Estimated number of 4096-channel snippets."""
        return self.n_channels // 4096 if self.n_channels > 0 else 0


class SRTDatasetBuilder:
    """
    Build training datasets from SRT observation files.
    
    Extracts RAW 4096-channel backgrounds for use with CadenceGenerator.
    Signal injection and preprocessing happen in the training pipeline.
    """
    
    SNIPPET_WIDTH = 4096  # Frequency channels per snippet
    
    def __init__(self, output_dir: str = "data/srt_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cadences: Dict[str, CadenceInfo] = {}
        
    def scan_directory(self, directory: str, recursive: bool = True) -> List[Path]:
        """Scan directory for HDF5 files."""
        directory = Path(directory)
        pattern = "**/*.h5" if recursive else "*.h5"
        files = list(directory.glob(pattern))
        print(f"Found {len(files)} HDF5 files in {directory}")
        return files
    
    def parse_filename(self, filepath: Path) -> Dict:
        """
        Parse SRT filename to extract metadata.
        
        Example filenames:
        - blc01_guppi_59368_33807_105548_TIC82452140_ON_0001.0000.h5
        - guppi_60705_85460_171014_TIC241225337_ON_0001.0000.h5
        """
        name = filepath.stem
        
        # Extract TIC name and ON/OFF status
        tic_match = re.search(r'(TIC\d+)_(ON|OFF)', name)
        if not tic_match:
            return None
        
        target = tic_match.group(1)
        obs_type = tic_match.group(2)
        
        # Extract timestamp (MJD and seconds)
        time_match = re.search(r'_(\d{5})_(\d+)_', name)
        mjd = time_match.group(1) if time_match else "unknown"
        
        # Extract date from parent directory if available
        parent = filepath.parent.name
        date_match = re.search(r'(\d{8})', parent)
        date = date_match.group(1) if date_match else mjd
        
        return {
            'target': target,
            'obs_type': obs_type,
            'mjd': mjd,
            'date': date,
            'filepath': filepath
        }
    
    def group_into_cadences(self, files: List[Path]) -> Dict[str, CadenceInfo]:
        """Group files into cadences by target name."""
        groups = defaultdict(list)
        
        for f in files:
            info = self.parse_filename(f)
            if info:
                key = f"{info['target']}_{info['date']}"
                groups[key].append(info)
        
        cadences = {}
        for key, file_infos in groups.items():
            file_infos.sort(key=lambda x: str(x['filepath']))
            
            on_files = [f for f in file_infos if f['obs_type'] == 'ON']
            off_files = [f for f in file_infos if f['obs_type'] == 'OFF']
            
            if len(on_files) >= 3 and len(off_files) >= 3:
                # Interleave: ON/OFF/ON/OFF/ON/OFF
                cadence_files = []
                for i in range(min(3, len(on_files), len(off_files))):
                    cadence_files.append(on_files[i]['filepath'])
                    cadence_files.append(off_files[i]['filepath'])
                
                cadence = CadenceInfo(
                    target_name=file_infos[0]['target'],
                    date=file_infos[0]['date'],
                    files=cadence_files[:6]
                )
                
                if cadence.is_complete:
                    try:
                        with h5py.File(cadence_files[0], 'r') as f:
                            header = dict(f['data'].attrs)
                            cadence.n_channels = header.get('nchans', 0)
                            cadence.freq_start = header.get('fch1', 0.0)
                    except:
                        pass
                
                cadences[key] = cadence
        
        self.cadences = cadences
        return cadences
    
    def print_cadence_summary(self):
        """Print summary of found cadences."""
        complete = [c for c in self.cadences.values() if c.is_complete]
        
        print(f"\n{'='*60}")
        print("CADENCE SUMMARY")
        print(f"{'='*60}")
        print(f"  Complete cadences: {len(complete)}")
        
        if complete:
            total_snippets = sum(c.n_snippets for c in complete)
            print(f"  Total potential snippets: {total_snippets:,}")
            
            print(f"\n  Cadences:")
            for c in complete[:10]:
                print(f"    - {c.target_name} ({c.date}): {c.n_snippets:,} snippets")
            if len(complete) > 10:
                print(f"    ... and {len(complete) - 10} more")
    
    def extract_backgrounds(self, 
                           cadence: CadenceInfo,
                           n_snippets: int = None,
                           random_sample: bool = True) -> np.ndarray:
        """
        Extract RAW background snippets from a cadence.
        
        Returns 4096-channel data for signal injection with CadenceGenerator.
        
        Args:
            cadence: CadenceInfo object
            n_snippets: Number of snippets to extract (None = all)
            random_sample: If True, randomly sample snippets
            
        Returns:
            Array of shape (n_snippets, 6, 16, 4096) - RAW, not normalized
        """
        from blimpy import Waterfall
        
        if not cadence.is_complete:
            raise ValueError(f"Cadence {cadence.target_name} is not complete")
        
        # Load all 6 observations
        cadence_data = []
        for filepath in cadence.files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf = Waterfall(str(filepath))
            data = wf.data.squeeze()
            cadence_data.append(data)
        
        # Stack: (6, time, freq)
        cadence_array = np.stack(cadence_data, axis=0)
        n_freq = cadence_array.shape[2]
        total_snippets = n_freq // self.SNIPPET_WIDTH
        
        if n_snippets is None:
            n_snippets = total_snippets
        
        if random_sample and n_snippets < total_snippets:
            indices = np.random.choice(total_snippets, n_snippets, replace=False)
            indices.sort()
        else:
            indices = range(min(n_snippets, total_snippets))
        
        # Extract RAW snippets
        snippets = []
        for idx in indices:
            start = idx * self.SNIPPET_WIDTH
            end = (idx + 1) * self.SNIPPET_WIDTH
            snippet = cadence_array[:, :, start:end]  # (6, 16, 4096)
            snippets.append(snippet)
        
        return np.array(snippets, dtype=np.float32)
    
    def build_training_dataset(self,
                               cadences: List[CadenceInfo] = None,
                               snippets_per_cadence: int = 500,
                               max_total_snippets: int = 20000,
                               output_name: str = "srt_backgrounds") -> str:
        """
        Build a training dataset from multiple cadences.
        
        Saves RAW 4096-channel backgrounds for use with CadenceGenerator.
        
        Args:
            cadences: List of cadences (None = all complete)
            snippets_per_cadence: Max snippets per cadence
            max_total_snippets: Maximum total snippets
            output_name: Name for output file
            
        Returns:
            Path to saved dataset
        """
        if cadences is None:
            cadences = [c for c in self.cadences.values() if c.is_complete]
        
        if not cadences:
            raise ValueError("No complete cadences found")
        
        print(f"\n{'='*60}")
        print("BUILDING RAW TRAINING DATASET")
        print(f"{'='*60}")
        print(f"  Cadences: {len(cadences)}")
        print(f"  Snippets per cadence: {snippets_per_cadence}")
        print(f"  Max total: {max_total_snippets}")
        print(f"  Output shape: (N, 6, 16, 4096)")
        
        all_snippets = []
        metadata = []
        
        for cadence in tqdm(cadences, desc="Processing"):
            try:
                n_to_extract = min(snippets_per_cadence, cadence.n_snippets)
                if n_to_extract == 0:
                    continue
                    
                snippets = self.extract_backgrounds(
                    cadence, 
                    n_snippets=n_to_extract,
                    random_sample=True
                )
                
                all_snippets.append(snippets)
                metadata.extend([{
                    'target': cadence.target_name,
                    'date': cadence.date
                }] * len(snippets))
                
                if sum(len(s) for s in all_snippets) >= max_total_snippets:
                    break
                    
            except Exception as e:
                print(f"  Error: {cadence.target_name}: {e}")
                continue
        
        if not all_snippets:
            raise ValueError("No snippets extracted")
        
        dataset = np.concatenate(all_snippets, axis=0)
        
        if len(dataset) > max_total_snippets:
            indices = np.random.choice(len(dataset), max_total_snippets, replace=False)
            dataset = dataset[indices]
        
        # Save
        output_path = self.output_dir / f"{output_name}.npz"
        np.savez_compressed(output_path, backgrounds=dataset, n_samples=len(dataset))
        
        meta_path = self.output_dir / f"{output_name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'n_samples': len(dataset),
                'n_cadences': len(cadences),
                'shape': list(dataset.shape),
                'fchans': self.SNIPPET_WIDTH,
                'targets': list(set(m['target'] for m in metadata))
            }, f, indent=2)
        
        print(f"\n✅ Dataset saved:")
        print(f"   {output_path} ({dataset.shape})")
        print(f"   {meta_path}")
        
        return str(output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract SRT backgrounds for training")
    parser.add_argument('--scan', '-s', nargs='+', required=True,
                        help='Directories to scan for HDF5 files')
    parser.add_argument('--output', '-o', default='data/srt_training',
                        help='Output directory')
    parser.add_argument('--snippets-per-cadence', '-n', type=int, default=500,
                        help='Max snippets per cadence')
    parser.add_argument('--max-snippets', '-m', type=int, default=20000,
                        help='Max total snippets')
    parser.add_argument('--name', default='srt_backgrounds',
                        help='Output dataset name')
    parser.add_argument('--list-only', action='store_true',
                        help='Only list cadences, do not extract')
    
    args = parser.parse_args()
    
    builder = SRTDatasetBuilder(output_dir=args.output)
    
    all_files = []
    for directory in args.scan:
        all_files.extend(builder.scan_directory(directory))
    
    print(f"\nTotal files: {len(all_files)}")
    
    builder.group_into_cadences(all_files)
    builder.print_cadence_summary()
    
    if not args.list_only:
        builder.build_training_dataset(
            snippets_per_cadence=args.snippets_per_cadence,
            max_total_snippets=args.max_snippets,
            output_name=args.name
        )


if __name__ == '__main__':
    main()
