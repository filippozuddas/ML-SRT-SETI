#!/usr/bin/env python3
"""
SRT/GBT Dataset Builder.

Extracts RAW background snippets from HDF5 observation files
for use with CadenceGenerator in training.

Supports frequency band separation (6 GHz vs 18 GHz bands)
and train/inference splitting.
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
# Register the bitshuffle/LZ4 HDF5 filters that Breakthrough Listen .h5 files
# are compressed with. Reading their 'data' with bare h5py otherwise fails
# ("can't open directory /usr/local/hdf5/lib/plugin"). blimpy used to pull this
# in for us; now that we read via h5py directly we must import it ourselves.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
from tqdm import tqdm


# Frequency band configuration
BAND_CONFIG = {
    '6GHz': {
        'name': 'C-band (~6.8 GHz)',
        'f_min': 6000,
        'f_max': 8000,
    },
    '18GHz': {
        'name': 'K-band (~18 GHz)', 
        'f_min': 17000,
        'f_max': 19000,
    },
    '1.4GHz': {
        'name': 'L-band (~1.4 GHz)',
        'f_min': 1000,
        'f_max': 2000,
    }
}


@dataclass
class CadenceInfo:
    """Information about a cadence (6 ON/OFF files)."""
    target_name: str
    date: str
    files: List[Path] = field(default_factory=list)
    n_channels: int = 0
    freq_start: float = 0.0
    freq_end: float = 0.0
    freq_band: str = 'unknown'  # '6GHz', '18GHz', or 'unknown'
    
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
        """Group files into cadences by target name with timestamp-based validation."""
        import re
        
        def extract_timestamp(filepath: Path) -> int:
            """Extract timestamp from GUPPI filename for sorting."""
            match = re.search(r'guppi_(\d+)_(\d+)_', filepath.name)
            if match:
                return int(match.group(1)) * 1000000 + int(match.group(2))
            return 0
        
        groups = defaultdict(list)
        
        for f in files:
            info = self.parse_filename(f)
            if info:
                # Key by target + date + directory (to separate different frequency observations)
                key = f"{info['target']}_{info['date']}_{f.parent}"
                info['timestamp'] = extract_timestamp(f)
                groups[key].append(info)
        
        cadences = {}
        invalid_count = 0
        
        for key, file_infos in groups.items():
            # Sort by TIMESTAMP (critical for correct ON/OFF order)
            file_infos.sort(key=lambda x: x['timestamp'])
            
            # Check if we have exactly 6 files
            if len(file_infos) != 6:
                invalid_count += 1
                continue
            
            # Validate ON/OFF pattern: should be ON, OFF, ON, OFF, ON, OFF
            expected_pattern = ['ON', 'OFF', 'ON', 'OFF', 'ON', 'OFF']
            actual_pattern = [f['obs_type'] for f in file_infos]
            
            if actual_pattern != expected_pattern:
                invalid_count += 1
                continue
            
            # Valid cadence!
            cadence_files = [f['filepath'] for f in file_infos]
            
            cadence = CadenceInfo(
                target_name=file_infos[0]['target'],
                date=file_infos[0]['date'],
                files=cadence_files
            )
            
            if cadence.is_complete:
                try:
                    with h5py.File(cadence_files[0], 'r') as f:
                        header = dict(f['data'].attrs)
                        cadence.n_channels = header.get('nchans', 0)
                        cadence.freq_start = header.get('fch1', 0.0)
                        
                        # Determine frequency band
                        for band_name, config in BAND_CONFIG.items():
                            if config['f_min'] <= cadence.freq_start <= config['f_max']:
                                cadence.freq_band = band_name
                                break
                except Exception as e:
                    # Skip corrupt files
                    continue
            
            cadences[key] = cadence

        if invalid_count > 0:
            print(f"  (Skipped {invalid_count} incomplete/invalid cadences)")

        # Deduplicate cadences that point to the SAME observation files under
        # different parent directories. turboSETI re-organizes each cadence into
        # SNR5/SNR10/SNR20 output folders (identical files); the parent-dir
        # component of the grouping key would otherwise count each copy as a
        # distinct cadence and over-represent it in training.
        deduped = {}
        seen_filesets = set()
        n_dups = 0
        for key, cad in cadences.items():
            fileset = tuple(sorted(f.name for f in cad.files))
            if fileset in seen_filesets:
                n_dups += 1
                continue
            seen_filesets.add(fileset)
            deduped[key] = cad
        if n_dups > 0:
            print(f"  (Deduplicated {n_dups} cadences with identical file sets)")
        cadences = deduped

        self.cadences = cadences
        return cadences
    
    def get_cadences_by_band(self, band: str = None) -> Dict[str, List[CadenceInfo]]:
        """Get cadences grouped by frequency band.
        
        Args:
            band: If specified, return only cadences for this band ('6GHz' or '18GHz')
            
        Returns:
            Dict with band names as keys and lists of CadenceInfo as values
        """
        complete = [c for c in self.cadences.values() if c.is_complete]
        
        by_band = defaultdict(list)
        for c in complete:
            by_band[c.freq_band].append(c)
        
        if band:
            return {band: by_band.get(band, [])}
        return dict(by_band)
    
    def print_cadence_summary(self):
        """Print summary of found cadences with band breakdown."""
        complete = [c for c in self.cadences.values() if c.is_complete]
        
        print(f"\n{'='*60}")
        print("CADENCE SUMMARY")
        print(f"{'='*60}")
        print(f"  Complete cadences: {len(complete)}")
        
        # Band breakdown
        by_band = self.get_cadences_by_band()
        print(f"\n  By frequency band:")
        for band_name, config in BAND_CONFIG.items():
            band_cadences = by_band.get(band_name, [])
            print(f"    {config['name']}: {len(band_cadences)} cadences")
        
        unknown = by_band.get('unknown', [])
        if unknown:
            print(f"    Unknown band: {len(unknown)} cadences")
        
        if complete:
            total_snippets = sum(c.n_snippets for c in complete)
            print(f"\n  Total potential snippets: {total_snippets:,}")
            
            print(f"\n  Sample cadences:")
            for c in complete[:5]:
                print(f"    - {c.target_name} ({c.freq_band}): {c.n_snippets:,} snippets")
            if len(complete) > 5:
                print(f"    ... and {len(complete) - 5} more")
    
    def extract_backgrounds(self,
                           cadence: CadenceInfo,
                           n_snippets: int = None,
                           random_sample: bool = True) -> np.ndarray:
        """
        Extract RAW background snippets from a cadence.

        Returns 4096-channel data for signal injection with CadenceGenerator.

        Memory-frugal: instead of loading the 6 multi-GB waterfalls into RAM
        (blimpy full-load stacks all 6 at once — at SNIPPET_WIDTH=4096 that's
        a ~250 GB transient peak on fine-resolution products, easily OOMing),
        this reads ONLY the chosen 4096-channel windows directly from each
        HDF5 file via h5py slicing. Peak memory ≈ the output array
        (n_snippets × 6 × 16 × 4096 float32), not the full files.

        Args:
            cadence: CadenceInfo object
            n_snippets: Number of snippets to extract (None = all)
            random_sample: If True, randomly sample snippets

        Returns:
            Array of shape (n_snippets, 6, 16, 4096) - RAW, not normalized
        """
        if not cadence.is_complete:
            raise ValueError(f"Cadence {cadence.target_name} is not complete")

        # Peek dataset shapes only — h5py does not read data here, just metadata.
        shapes = []
        for filepath in cadence.files:
            with h5py.File(str(filepath), 'r') as hf:
                shapes.append(hf['data'].shape)

        # Some observations carry 1-2 extra integration rows (17-18 time bins
        # instead of the canonical 16); we slice the first 16 rows from every
        # obs. Fewer than 16 bins is genuinely unusable.
        min_t = min(s[0] for s in shapes)
        if min_t < 16:
            raise ValueError(
                f"Cadence {cadence.target_name} has an observation with only "
                f"{min_t} time bins (<16); cannot use."
            )

        # Common channel count across the 6 files (last axis is frequency).
        n_freq = min(s[-1] for s in shapes)
        total_snippets = n_freq // self.SNIPPET_WIDTH
        if total_snippets == 0:
            raise ValueError(
                f"Cadence {cadence.target_name} has only {n_freq} channels "
                f"(<{self.SNIPPET_WIDTH}); cannot extract a snippet."
            )

        if n_snippets is None:
            n_snippets = total_snippets

        if random_sample and n_snippets < total_snippets:
            indices = np.random.choice(total_snippets, n_snippets, replace=False)
            indices.sort()
        else:
            indices = np.arange(min(n_snippets, total_snippets))

        # Preallocate the output; fill it window-by-window straight from disk.
        out = np.empty((len(indices), 6, 16, self.SNIPPET_WIDTH), dtype=np.float32)

        for fi, filepath in enumerate(cadence.files):
            print(f"\n    Reading file {fi+1}/6: {filepath.name}...", end=" ", flush=True)
            try:
                # Big raw-data chunk cache: indices are sorted, so a generous
                # cache lets adjacent windows reuse an already-decompressed
                # chunk instead of re-inflating it on every read.
                with h5py.File(str(filepath), 'r', rdcc_nbytes=256 * 1024 * 1024) as hf:
                    dset = hf['data']                  # (n_int, [n_if], n_chan)
                    three_d = dset.ndim == 3
                    for si, idx in enumerate(indices):
                        start = idx * self.SNIPPET_WIDTH
                        end = start + self.SNIPPET_WIDTH
                        if three_d:
                            out[si, fi] = dset[:16, 0, start:end]
                        else:
                            out[si, fi] = dset[:16, start:end]
                print("✓")
            except OSError as e:
                print(f"❌ FAILED")
                if "truncated file" in str(e).lower():
                    raise OSError(f"File is truncated or corrupt: {filepath}") from e
                raise OSError(f"Could not open HDF5 file {filepath}: {e}") from e

        return out
    
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

        # Accumulate into a disk-backed memmap, not an in-RAM array. At
        # SNIPPET_WIDTH=4096, a 200k-snippet plate is ~290 GiB — the OUTPUT
        # array alone doesn't fit in RAM regardless of how frugally each
        # cadence is read (see extract_backgrounds). Writing through a memmap
        # lets the OS page the buffer to disk instead of requiring it all
        # resident at once; peak RAM is one cadence's small temp array.
        per_cadence_n = [
            min(snippets_per_cadence, c.n_snippets)
            for c in cadences
        ]
        capacity = min(max_total_snippets, sum(per_cadence_n))
        if capacity == 0:
            raise ValueError("No snippets extracted")

        itemsize = 6 * 16 * self.SNIPPET_WIDTH * 4  # bytes per snippet (float32)
        building_path = self.output_dir / f"{output_name}.building.mmap"
        dataset = np.memmap(building_path, dtype=np.float32, mode='w+',
                            shape=(capacity, 6, 16, self.SNIPPET_WIDTH))
        metadata = []
        pos = 0

        for cadence, n_to_extract in zip(tqdm(cadences, desc="Processing"), per_cadence_n):
            if pos >= capacity:
                break
            if n_to_extract == 0:
                continue
            n_to_extract = min(n_to_extract, capacity - pos)
            try:
                snippets = self.extract_backgrounds(
                    cadence,
                    n_snippets=n_to_extract,
                    random_sample=True
                )
            except Exception as e:
                print(f"  Error: {cadence.target_name}: {e}")
                continue

            k = len(snippets)
            dataset[pos:pos + k] = snippets
            pos += k
            metadata.extend([{
                'target': cadence.target_name,
                'date': cadence.date
            }] * k)
            del snippets

        if pos == 0:
            building_path.unlink(missing_ok=True)
            raise ValueError("No snippets extracted")

        dataset.flush()
        del dataset  # release the mmap handle before truncating/renaming the file

        final_bytes = pos * itemsize
        saved_shape = (pos, 6, 16, self.SNIPPET_WIDTH)
        # Above this, skip packaging into a compressed .npz (the compressor
        # needs to walk the whole array — slow and disk-hungry for no benefit
        # on ~incompressible telescope noise) and keep the raw memmap instead.
        NPZ_THRESHOLD_BYTES = 20 * 1024 ** 3  # 20 GiB

        if final_bytes <= NPZ_THRESHOLD_BYTES:
            trimmed = np.memmap(building_path, dtype=np.float32, mode='r',
                                shape=(capacity, 6, 16, self.SNIPPET_WIDTH))[:pos]
            output_path = self.output_dir / f"{output_name}.npz"
            np.savez_compressed(output_path, backgrounds=trimmed, n_samples=pos)
            del trimmed
            building_path.unlink()
        else:
            # Too large to package — keep it as the raw memmap + meta.json
            # that train_large_scale.py already reads via --mmap (same
            # convention as experiments/convert_plate_to_mmap.py).
            output_path = self.output_dir / f"{output_name}.mmap"
            os.replace(building_path, output_path)  # instant, same filesystem
            os.truncate(output_path, final_bytes)     # drop unused tail capacity
            with open(self.output_dir / f"{output_name}.meta.json", 'w') as f:
                json.dump({
                    'shape': list(saved_shape),
                    'dtype': 'float32',
                    'nbytes': final_bytes,
                }, f, indent=2)

        meta_path = self.output_dir / f"{output_name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'n_samples': pos,
                'n_cadences': len(cadences),
                'shape': list(saved_shape),
                'fchans': self.SNIPPET_WIDTH,
                'targets': list(set(m['target'] for m in metadata))
            }, f, indent=2)

        print(f"\n✅ Dataset saved:")
        print(f"   {output_path} ({saved_shape})")
        print(f"   {meta_path}")
        if output_path.suffix == '.mmap':
            print(f"\n   Dataset too large for a portable .npz ({final_bytes / 1e9:.1f} GB) "
                  f"— saved as a raw memmap instead.")
            print(f"   Train with: --plate {output_path} --mmap")

        return str(output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract backgrounds for training (supports band separation)")
    parser.add_argument('--scan', '-s', nargs='+', required=True,
                        help='Directories to scan for HDF5 files')
    parser.add_argument('--output', '-o', default='data/srt_training',
                        help='Output directory')
    parser.add_argument('--snippets-per-cadence', '-n', type=int, default=500,
                        help='Max snippets per cadence')
    parser.add_argument('--max-snippets', '-m', type=int, default=15000,
                        help='Max total snippets per band')
    parser.add_argument('--name', default='backgrounds',
                        help='Output dataset name prefix')
    parser.add_argument('--band', '-b', choices=['6GHz', '18GHz', '1.4GHz', 'all', 'mixed'], default='all',
                        help='Frequency band to process (default: all)')
    parser.add_argument('--match-json', type=str, default=None,
                        help='Path to a metadata.json to extract targets (less precise than exclude-txt)')
    parser.add_argument('--exclude-txt', type=str, default=None,
                        help='Path to inference_cadences.txt to perfectly exclude inference cadences')
    parser.add_argument('--training-cadences', '-t', type=int, default=None,
                        help='Number of cadences to use for training (rest saved for inference)')
    parser.add_argument('--mix-bins', type=float, default=1000.0,
                        help='Bin size in MHz for balancing a mixed dataset (default: 1000)')
    parser.add_argument('--train-fraction', type=float, default=0.5,
                        help='Fraction of cadences per frequency bin used for training in '
                             '--band mixed; the rest are held out for inference (default: 0.5)')
    parser.add_argument('--exclude-targets', nargs='+', default=None,
                        help='Target names (e.g. TIC368536386) to keep OUT of training backgrounds; '
                             'routed to the inference pool instead')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible cadence selection in --band mixed')
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
    
    if args.list_only:
        return
        
    # Match specific targets if a JSON is provided
    target_filter = None
    if args.match_json:
        with open(args.match_json, 'r') as f:
            meta = json.load(f)
            target_filter = set(meta.get('targets', []))
            print(f"\nFiltering cadences by {len(target_filter)} targets from {args.match_json}")
            
    # Exclude specific cadences if a TXT is provided
    exclude_files = set()
    if args.exclude_txt:
        with open(args.exclude_txt, 'r') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|')
                    files_str = parts[-1]  # The file list is always the last part
                    first_file = Path(files_str.split(',')[0]).name
                    exclude_files.add(first_file)
        print(f"\nExcluding {len(exclude_files)} cadences listed in {args.exclude_txt}")
    
    if args.band == 'mixed':
        print(f"\n{'='*60}")
        print("PROCESSING: MIXED MULTI-BAND DATASET")
        print(f"{'='*60}")

        complete_cadences = [c for c in builder.cadences.values() if c.is_complete]

        if target_filter:
            complete_cadences = [c for c in complete_cadences if c.target_name in target_filter]
            print(f"  Cadences after JSON target filter: {len(complete_cadences)}")

        if exclude_files:
            complete_cadences = [c for c in complete_cadences if c.files[0].name not in exclude_files]
            print(f"  Cadences after TXT exclude filter: {len(complete_cadences)}")

        if not complete_cadences:
            print("No complete cadences found for mixed dataset.")
            return

        import random
        if args.seed is not None:
            random.seed(args.seed)

        # Hold out explicitly-excluded targets (e.g. the benchmark) from training:
        # they go straight to the inference pool and never feed the background set.
        exclude_targets = set(args.exclude_targets or [])
        excluded_cadences = [c for c in complete_cadences if c.target_name in exclude_targets]
        complete_cadences = [c for c in complete_cadences if c.target_name not in exclude_targets]
        if excluded_cadences:
            print(f"  Excluded {len(excluded_cadences)} cadence(s) from training "
                  f"(targets: {sorted(exclude_targets)}) -> inference pool")

        # Group by frequency bin (e.g. 4800 -> 5000, 6100 -> 6000 for mix_bins=1000)
        by_freq = defaultdict(list)
        for c in complete_cadences:
            bin_mhz = round(c.freq_start / args.mix_bins) * args.mix_bins
            by_freq[bin_mhz].append(c)

        selected_cadences = []
        inference_cadences = list(excluded_cadences)
        print(f"Splitting each of {len(by_freq)} frequency bins "
              f"{args.train_fraction:.0%} train / {1 - args.train_fraction:.0%} inference:")
        for bin_mhz in sorted(by_freq.keys()):
            cads = by_freq[bin_mhz]
            n_take = int(len(cads) * args.train_fraction)
            selected = random.sample(cads, n_take)

            selected_ids = {id(c) for c in selected}
            for c in cads:
                if id(c) not in selected_ids:
                    inference_cadences.append(c)

            selected_cadences.extend(selected)
            print(f"  - ~{bin_mhz/1000:.1f} GHz: {n_take}/{len(cads)} train, "
                  f"{len(cads) - n_take} inference")

        # Shuffle so the snippet-budget cap (if ever hit) doesn't systematically
        # starve whichever frequency bin is processed last.
        random.shuffle(selected_cadences)

        # Save the list of inference cadences (target|freq_start|files, matching
        # the format --exclude-txt already parses via the last '|'-separated field)
        if inference_cadences:
            inference_path = builder.output_dir / "inference_cadences_mixed.txt"
            with open(inference_path, 'w') as f:
                for c in inference_cadences:
                    files_str = ','.join(str(fp) for fp in c.files)
                    f.write(f"{c.target_name}|{c.freq_start}|{files_str}\n")
            print(f"\n  Saved {len(inference_cadences)} held-out cadences for inference testing to: {inference_path}")

        output_name = f"{args.name}_mixed"
        builder.build_training_dataset(
            cadences=selected_cadences,
            snippets_per_cadence=args.snippets_per_cadence,
            max_total_snippets=args.max_snippets,
            output_name=output_name
        )
        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        return

    # Get cadences by band
    bands_to_process = [args.band] if args.band != 'all' else list(BAND_CONFIG.keys())
    
    for band_name in bands_to_process:
        by_band = builder.get_cadences_by_band(band_name)
        band_cadences = by_band.get(band_name, [])
        
        if not band_cadences:
            print(f"\n⚠️  No cadences found for {band_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"PROCESSING: {BAND_CONFIG[band_name]['name']}")
        print(f"{'='*60}")
        print(f"  Cadences: {len(band_cadences)}")
        
        if target_filter:
            band_cadences = [c for c in band_cadences if c.target_name in target_filter]
            print(f"  Cadences after JSON target filter: {len(band_cadences)}")
            
        if exclude_files:
            band_cadences = [c for c in band_cadences if c.files[0].name not in exclude_files]
            print(f"  Cadences after TXT exclude filter: {len(band_cadences)}")
            
        if not band_cadences:
            print(f"  No matching cadences in this band after filters.")
            continue
            
        # Split into training vs inference if requested
        if args.training_cadences and args.training_cadences < len(band_cadences):
            training_cadences = band_cadences[:args.training_cadences]
            inference_cadences = band_cadences[args.training_cadences:]
            
            print(f"  Training: {len(training_cadences)} cadences")
            print(f"  Inference: {len(inference_cadences)} cadences")
            
            # Save inference cadence list
            inference_path = builder.output_dir / f"inference_cadences_{band_name}.txt"
            with open(inference_path, 'w') as f:
                for c in inference_cadences:
                    files_str = ','.join(str(fp) for fp in c.files)
                    f.write(f"{c.target_name}|{files_str}\n")
            print(f"  Saved: {inference_path}")
        else:
            training_cadences = band_cadences
        
        # Build dataset
        output_name = f"{args.name}_{band_name}"
        builder.build_training_dataset(
            cadences=training_cadences,
            snippets_per_cadence=args.snippets_per_cadence,
            max_total_snippets=args.max_snippets,
            output_name=output_name
        )
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
