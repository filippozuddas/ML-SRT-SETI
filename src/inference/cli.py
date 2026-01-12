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


def cmd_listfile(args):
    """Process cadences from a list file.
    
    List file format (one per line):
        TARGET_NAME|path1.h5,path2.h5,path3.h5,path4.h5,path5.h5,path6.h5
    """
    from .pipeline import SRTPipeline
    
    list_path = Path(args.list_file)
    if not list_path.exists():
        print(f"Error: List file not found: {list_path}", file=sys.stderr)
        sys.exit(1)
    
    # Parse list file
    cadences = []
    with open(list_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) != 2:
                print(f"Warning: Line {line_num}: Expected TARGET|paths format, skipping")
                continue
            
            target_name = parts[0].strip()
            file_paths = [p.strip() for p in parts[1].split(',')]
            
            if len(file_paths) != 6:
                print(f"Warning: Line {line_num} ({target_name}): Expected 6 files, got {len(file_paths)}, skipping")
                continue
            
            # Check files exist
            missing = [p for p in file_paths if not Path(p).exists()]
            if missing:
                print(f"Warning: Line {line_num} ({target_name}): {len(missing)} files missing, skipping")
                continue
            
            cadences.append((target_name, file_paths))
    
    print(f"Found {len(cadences)} valid cadences in {list_path}")
    
    if not cadences:
        print("No valid cadences found. Exiting.")
        sys.exit(0)
    
    # Create pipeline
    overlap_enabled = hasattr(args, 'overlap') and args.overlap
    if hasattr(args, 'optimized') and args.optimized:
        from .pipeline_optimized import SRTPipelineOptimized
        overlap_str = "+overlap" if overlap_enabled else ""
        print(f"Using OPTIMIZED pipeline (chunks={args.chunks}, batch_size={args.batch_size}{overlap_str})")
        pipeline = SRTPipelineOptimized(
            encoder_path=args.encoder,
            classifier_path=args.classifier,
            threshold=args.threshold,
            n_chunks=args.chunks,
            batch_size=args.batch_size,
            overlap=overlap_enabled,
            verbose=False
        )
    else:
        pipeline = SRTPipeline(
            encoder_path=args.encoder,
            classifier_path=args.classifier,
            threshold=args.threshold,
            overlap=overlap_enabled,
            verbose=False
        )
    
    # Process each cadence
    all_results = []
    
    # Create output directory and summary file if output specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize summary CSV with header
        summary_path = output_dir / "inference_summary.csv"
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['target', 'n_snippets', 'n_detections', 'top_freq_mhz', 'top_probability', 'status'])
    
    for i, (target_name, file_paths) in enumerate(cadences, 1):
        print(f"[{i}/{len(cadences)}] {target_name}...", end=' ', flush=True)
        
        try:
            result = pipeline.process_cadence(file_paths)
            all_results.append((target_name, result))
            
            if result.n_detections > 0:
                top = result.top_candidates[0]
                print(f"🔴 {result.n_detections} detections (top: {top.eti_probability:.3f} @ {top.freq_mhz:.2f} MHz)")
            else:
                print(f"✓ No detections")
            
            # Save results IMMEDIATELY for this target
            if args.output:
                # Append to summary CSV
                with open(summary_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    top = result.top_candidates[0] if result.n_detections > 0 else None
                    writer.writerow([
                        target_name,
                        len(result.snippets),
                        result.n_detections,
                        top.freq_mhz if top else '',
                        top.eti_probability if top else '',
                        'success'
                    ])
                
                # Save per-target detailed results
                target_dir = output_dir / target_name
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Save metadata with file paths (for easy plotting later)
                metadata_path = target_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'target': target_name,
                        'files': file_paths,
                        'threshold': args.threshold,
                        'n_snippets': len(result.snippets),
                        'n_detections': result.n_detections
                    }, f, indent=2)
                
                # All snippets
                all_snippets_path = target_dir / "all_snippets.csv"
                with open(all_snippets_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['freq_mhz', 'probability', 'is_eti', 'center_channel'])
                    for s in result.snippets:
                        writer.writerow([s.freq_mhz, s.eti_probability, s.is_eti, s.center_channel])
                
                # Candidates only (those above threshold)
                if result.n_detections > 0:
                    candidates_path = target_dir / "candidates.csv"
                    with open(candidates_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['freq_mhz', 'probability', 'center_channel'])
                        # Save ALL candidates above threshold (sorted by probability)
                        eti_candidates = sorted(
                            [s for s in result.snippets if s.is_eti],
                            key=lambda x: x.eti_probability,
                            reverse=True
                        )
                        for s in eti_candidates:
                            writer.writerow([s.freq_mhz, s.eti_probability, s.center_channel])
                
        except Exception as e:
            print(f"❌ Error: {e}")
            all_results.append((target_name, None))
            
            # Log error to summary
            if args.output:
                with open(summary_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([target_name, 0, 0, '', '', f'error: {str(e)[:50]}'])
    
    # Final summary
    if args.output:
        print(f"\nResults saved to {output_dir}/")
        print(f"  📄 Summary: inference_summary.csv")
        print(f"  📁 Per-target folders: {len([r for _, r in all_results if r is not None])} targets")
    
    # Summary
    successful = [r for _, r in all_results if r is not None]
    total_detections = sum(r.n_detections for r in successful)
    with_detections = sum(1 for r in successful if r.n_detections > 0)
    
    print(f"\n{'='*50}")
    print(f"INFERENCE COMPLETE")
    print(f"  Cadences processed: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  With detections: {with_detections}")
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
    
    # Listfile command - process from TARGET|path1,...,path6 format
    listfile_parser = subparsers.add_parser('listfile', help='Process cadences from a list file')
    listfile_parser.add_argument('--list-file', '-l', required=True,
                                  help='Path to list file (TARGET|path1,path2,...,path6 format)')
    listfile_parser.add_argument('--encoder', '-e', required=True,
                                  help='Path to encoder model')
    listfile_parser.add_argument('--classifier', '-c', required=True,
                                  help='Path to classifier')
    listfile_parser.add_argument('--threshold', '-t', type=float, default=0.5,
                                  help='ETI detection threshold')
    listfile_parser.add_argument('--output', '-o', type=str, default=None,
                                  help='Output directory for results')
    # Optimized pipeline options
    listfile_parser.add_argument('--optimized', '-O', action='store_true',
                                  help='Use optimized chunked pipeline for large files')
    listfile_parser.add_argument('--chunks', type=int, default=8,
                                  help='Number of frequency chunks (default: 8, used with --optimized)')
    listfile_parser.add_argument('--batch-size', type=int, default=256,
                                  help='Batch size for GPU encoding (default: 256, used with --optimized)')
    listfile_parser.add_argument('--overlap', action='store_true',
                                  help='Use 50%% overlapping windows for better signal coverage (2x more snippets)')
    listfile_parser.set_defaults(func=cmd_listfile)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
