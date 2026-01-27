#!/usr/bin/env python3
"""
Convert plate.npz to memory-mapped format for shared memory access.

This allows multiple worker processes to share the same memory region,
dramatically reducing RAM usage (from N*plate_size to 1*plate_size).

Usage:
    python scripts/convert_plate_to_mmap.py --input plate.npz --output plate.mmap
"""

import argparse
import numpy as np
from pathlib import Path
import json

def convert_npz_to_mmap(input_path: str, output_path: str):
    """Convert .npz plate file to memory-mapped format."""
    print(f"Loading plate from {input_path}...")
    plate_data = np.load(input_path)
    backgrounds = plate_data['backgrounds']
    
    print(f"  Shape: {backgrounds.shape}")
    print(f"  Dtype: {backgrounds.dtype}")
    print(f"  Size: {backgrounds.nbytes / 1e9:.2f} GB")
    
    # Save as memory-mapped file
    output_path = Path(output_path)
    mmap_path = output_path.with_suffix('.mmap')
    meta_path = output_path.with_suffix('.meta.json')
    
    print(f"\nCreating memory-mapped file: {mmap_path}")
    
    # Create mmap file with same shape and dtype
    mmap_array = np.memmap(
        str(mmap_path), 
        dtype=backgrounds.dtype, 
        mode='w+', 
        shape=backgrounds.shape
    )
    
    # Copy data
    print("  Copying data to mmap...")
    mmap_array[:] = backgrounds[:]
    mmap_array.flush()
    
    # Save metadata
    metadata = {
        'shape': list(backgrounds.shape),
        'dtype': str(backgrounds.dtype),
        'nbytes': int(backgrounds.nbytes)
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Created: {mmap_path}")
    print(f"  ✓ Metadata: {meta_path}")
    
    # Verify
    print("\nVerifying...")
    loaded = np.memmap(str(mmap_path), dtype=backgrounds.dtype, mode='r', shape=backgrounds.shape)
    if np.allclose(loaded[:10], backgrounds[:10]):
        print("  ✓ Verification passed!")
    else:
        print("  ✗ Verification FAILED!")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert plate.npz to memory-mapped format')
    parser.add_argument('--input', '-i', required=True, help='Input .npz file path')
    parser.add_argument('--output', '-o', required=True, help='Output base path (will create .mmap and .meta.json)')
    
    args = parser.parse_args()
    
    success = convert_npz_to_mmap(args.input, args.output)
    
    if success:
        print("\n" + "="*60)
        print("CONVERSION COMPLETE")
        print("="*60)
        print(f"\nTo use in training, run:")
        print(f"  python scripts/train_large_scale.py --plate {args.output}.mmap --mmap ...")
    else:
        print("\nConversion failed!")
        exit(1)

if __name__ == "__main__":
    main()
