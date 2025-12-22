#!/usr/bin/env python3
"""
SRT Pipeline Command-Line Interface.

Run ETI detection on SRT observation files from the command line.

Usage:
    python -m src.inference.cli process --files ON1.h5 OFF1.h5 ... --encoder model.keras --classifier rf.joblib
    python -m src.inference.cli info --file observation.h5
"""

import argparse
import sys
import json
import csv
from pathlib import Path
from typing import List, Optional


def cmd_process(args):
    """Process a cadence through the pipeline."""
    # Validate files
    if len(args.files) != 6:
        print(f"Error: Expected 6 files, got {len(args.files)}", file=sys.stderr)
        sys.exit(1)
    
    for f in args.files:
        if not Path(f).exists():
            print(f"Error: File not found: {f}", file=sys.stderr)
            sys.exit(1)
    
    # Choose pipeline based on --optimized flag
    if args.optimized:
        from .pipeline_optimized import SRTPipelineOptimized
        print("Using OPTIMIZED pipeline with chunked processing...")
        pipeline = SRTPipelineOptimized(
            encoder_path=args.encoder,
            classifier_path=args.classifier,
            threshold=args.threshold,
            n_chunks=args.chunks,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )
        result = pipeline.process_cadence(file_paths=args.files)
    else:
        from .pipeline import SRTPipeline
        pipeline = SRTPipeline(
            encoder_path=args.encoder,
            classifier_path=args.classifier,
            threshold=args.threshold,
            verbose=not args.quiet
        )
        result = pipeline.process_cadence(
            file_paths=args.files,
            f_start=args.freq_start,
            f_stop=args.freq_stop
        )
    
    # Output
    if args.output:
        output_path = Path(args.output)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to {output_path}")
            
        elif output_path.suffix == '.csv':
            # Save ALL snippets to *_all.csv
            all_path = output_path.with_name(output_path.stem + '_all.csv')
            with open(all_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['freq_mhz', 'probability', 'is_eti', 'center_channel'])
                for s in result.snippets:
                    writer.writerow([s.freq_mhz, s.eti_probability, s.is_eti, s.center_channel])
            
            # Save only ETI candidates to *_candidates.csv
            candidates_path = output_path.with_name(output_path.stem + '_candidates.csv')
            with open(candidates_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['freq_mhz', 'probability', 'is_eti', 'center_channel'])
                for s in result.snippets:
                    if s.is_eti:
                        writer.writerow([s.freq_mhz, s.eti_probability, s.is_eti, s.center_channel])
            
            print(f"\nResults saved:")
            print(f"  📄 All snippets: {all_path} ({len(result.snippets)} rows)")
            print(f"  🎯 Candidates:   {candidates_path} ({result.n_detections} rows)")
    
    # Print summary
    if not args.quiet:
        print(f"\n📊 SUMMARY")
        print(f"   Source: {result.source_name}")
        print(f"   Snippets analyzed: {len(result.snippets)}")
        print(f"   ETI candidates: {result.n_detections}")
        
        if result.n_detections > 0:
            print(f"\n🎯 TOP CANDIDATES:")
            for i, s in enumerate(result.top_candidates[:5], 1):
                status = "✅" if s.is_eti else "❌"
                print(f"   {i}. {s.freq_mhz:.6f} MHz - P(ETI)={s.eti_probability:.3f} {status}")


def cmd_info(args):
    """Show information about an observation file."""
    from .pipeline import SRTPipeline
    from ..data.loader import SRTDataLoader
    
    loader = SRTDataLoader()
    
    for filepath in args.files:
        if not Path(filepath).exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            continue
        
        info = loader.get_file_info(filepath)
        
        print(f"\n📁 {Path(filepath).name}")
        print(f"   Source: {info['source_name']}")
        print(f"   Channels: {info['nchans']:,}")
        print(f"   Start freq: {info['fch1']:.6f} MHz")
        print(f"   Channel width: {info['freq_resolution_hz']:.2f} Hz")
        print(f"   Integration time: {info['tsamp']:.2f} s")


def cmd_batch(args):
    """Process multiple cadences from a directory."""
    from .pipeline import SRTPipeline
    from ..data.loader import SRTDataLoader
    
    loader = SRTDataLoader()
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find cadences
    cadences = loader.find_cadence_files(input_dir)
    print(f"Found {len(cadences)} cadences in {input_dir}")
    
    if not cadences:
        print("No cadences found. Make sure files are organized in groups of 6.")
        sys.exit(0)
    
    # Create pipeline
    pipeline = SRTPipeline(
        encoder_path=args.encoder,
        classifier_path=args.classifier,
        threshold=args.threshold,
        verbose=False
    )
    
    # Process each cadence
    all_results = []
    
    for i, file_paths in enumerate(cadences, 1):
        print(f"\nProcessing cadence {i}/{len(cadences)}...")
        result = pipeline.process_cadence([str(p) for p in file_paths])
        all_results.append(result)
        print(f"  → {result.n_detections} ETI candidates")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'n_snippets', 'n_detections', 'top_freq_mhz', 'top_probability'])
            
            for r in all_results:
                top = r.top_candidates[0] if r.snippets else None
                writer.writerow([
                    r.source_name,
                    len(r.snippets),
                    r.n_detections,
                    top.freq_mhz if top else '',
                    top.eti_probability if top else ''
                ])
        
        print(f"\nBatch results saved to {output_path}")
    
    # Summary
    total_detections = sum(r.n_detections for r in all_results)
    print(f"\n{'='*50}")
    print(f"BATCH COMPLETE")
    print(f"  Cadences processed: {len(all_results)}")
    print(f"  Total ETI candidates: {total_detections}")


def main():
    parser = argparse.ArgumentParser(
        prog='srt-pipeline',
        description='SRT ETI Detection Pipeline'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a single cadence')
    process_parser.add_argument('--files', '-f', nargs=6, required=True,
                                help='6 observation files in order [ON,OFF,ON,OFF,ON,OFF]')
    process_parser.add_argument('--encoder', '-e', required=True,
                                help='Path to encoder model (.keras or .h5)')
    process_parser.add_argument('--classifier', '-c', required=True,
                                help='Path to Random Forest classifier (.joblib)')
    process_parser.add_argument('--threshold', '-t', type=float, default=0.9,
                                help='ETI detection threshold (default: 0.9)')
    process_parser.add_argument('--freq-start', type=float, default=None,
                                help='Start frequency in MHz (optional)')
    process_parser.add_argument('--freq-stop', type=float, default=None,
                                help='Stop frequency in MHz (optional)')
    process_parser.add_argument('--output', '-o', type=str, default=None,
                                help='Output file (.json or .csv)')
    process_parser.add_argument('--quiet', '-q', action='store_true',
                                help='Suppress progress output')
    # Optimized pipeline options
    process_parser.add_argument('--optimized', '-O', action='store_true',
                                help='Use optimized chunked pipeline for large files')
    process_parser.add_argument('--chunks', type=int, default=8,
                                help='Number of frequency chunks (default: 8, used with --optimized)')
    process_parser.add_argument('--batch-size', type=int, default=256,
                                help='Batch size for GPU encoding (default: 256, used with --optimized)')
    process_parser.set_defaults(func=cmd_process)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show file information')
    info_parser.add_argument('--files', '-f', nargs='+', required=True,
                             help='Observation file(s) to inspect')
    info_parser.set_defaults(func=cmd_info)
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple cadences')
    batch_parser.add_argument('--input-dir', '-i', required=True,
                              help='Directory containing observation files')
    batch_parser.add_argument('--encoder', '-e', required=True,
                              help='Path to encoder model')
    batch_parser.add_argument('--classifier', '-c', required=True,
                              help='Path to classifier')
    batch_parser.add_argument('--threshold', '-t', type=float, default=0.5,
                              help='ETI detection threshold')
    batch_parser.add_argument('--output', '-o', type=str, default=None,
                              help='Output CSV file')
    batch_parser.set_defaults(func=cmd_batch)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
