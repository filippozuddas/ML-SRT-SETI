#!/usr/bin/env python3
"""
Create a video slideshow from candidate plot images.

Usage:
    python scripts/create_candidates_video.py \
        --input results/inference/18GHz_2 \
        --output candidates_slideshow.mp4 \
        --fps 1
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
from typing import List, Tuple


def find_candidate_plots(inference_dir: Path, target_filter: str = None) -> List[Tuple[str, str, Path]]:
    """
    Find all candidate plot images in inference directory.
    
    Args:
        inference_dir: Root inference directory
        target_filter: Optional target name to filter (only include this target)
    
    Returns list of (target_name, candidate_info, image_path) tuples.
    """
    plots = []
    
    for target_dir in sorted(inference_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        
        # Apply target filter if specified
        if target_filter and target_dir.name != target_filter:
            continue
        
        plots_dir = target_dir / "plots"
        if not plots_dir.exists():
            continue
        
        target_name = target_dir.name
        
        # Find candidate images (sorted by number)
        for img_path in sorted(plots_dir.glob("candidate_*.png")):
            # Extract info from filename
            # Format: candidate_1_18167.37MHz_P0.971.png
            match = re.match(r'candidate_(\d+)_(.+)MHz_P([\d.]+)', img_path.stem)
            if match:
                cand_num = match.group(1)
                freq = match.group(2)
                prob = match.group(3)
                info = f"#{cand_num} - {freq} MHz - P={prob}"
            else:
                info = img_path.stem
            
            plots.append((target_name, info, img_path))
    
    return plots


def add_overlay_text(image: np.ndarray, target_name: str, candidate_info: str) -> np.ndarray:
    """Add target name and candidate info overlay to image."""
    # Convert to PIL for better text rendering
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Try to use a nice font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add semi-transparent background for text
    img_width, img_height = img_pil.size
    
    # Target name at top
    text_target = f"Target: {target_name}"
    bbox = draw.textbbox((0, 0), text_target, font=font_large)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw background rectangle
    draw.rectangle([10, 10, text_width + 30, text_height + 20], fill=(0, 0, 0, 180))
    draw.text((20, 12), text_target, font=font_large, fill=(255, 255, 255))
    
    # Candidate info below
    bbox2 = draw.textbbox((0, 0), candidate_info, font=font_small)
    text_width2 = bbox2[2] - bbox2[0]
    text_height2 = bbox2[3] - bbox2[1]
    
    draw.rectangle([10, text_height + 25, text_width2 + 30, text_height + text_height2 + 35], 
                   fill=(0, 0, 0, 180))
    draw.text((20, text_height + 27), candidate_info, font=font_small, fill=(255, 255, 0))
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def create_video(plots: List[Tuple[str, str, Path]], 
                 output_path: str,
                 fps: float = 1.0,
                 target_size: Tuple[int, int] = (1920, 1080)):
    """Create video from candidate plots."""
    if not plots:
        print("No plots found!")
        return
    
    print(f"Creating video with {len(plots)} frames at {fps} FPS...")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    for i, (target_name, info, img_path) in enumerate(plots):
        print(f"  [{i+1}/{len(plots)}] {target_name} - {info}", end='\r')
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n  Warning: Could not load {img_path}")
            continue
        
        # Resize to target size while maintaining aspect ratio
        h, w = img.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas with black background
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Center the image
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        # Add overlay text
        canvas = add_overlay_text(canvas, target_name, info)
        
        # Write frame
        out.write(canvas)
    
    out.release()
    print(f"\n✅ Video saved to: {output_path}")
    print(f"   Duration: {len(plots) / fps:.1f} seconds ({len(plots)} frames at {fps} FPS)")


def main():
    parser = argparse.ArgumentParser(description='Create video slideshow of candidate plots')
    parser.add_argument('--input', '-i', required=True,
                        help='Inference results directory')
    parser.add_argument('--output', '-o', default='candidates_slideshow.mp4',
                        help='Output video file (default: candidates_slideshow.mp4)')
    parser.add_argument('--fps', '-f', type=float, default=1.0,
                        help='Frames per second (default: 1.0 = 1 second per candidate)')
    parser.add_argument('--width', type=int, default=1920,
                        help='Video width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                        help='Video height (default: 1080)')
    parser.add_argument('--target', '-t', type=str, default=None,
                        help='Filter to specific target name (e.g., TIC241173474)')
    
    args = parser.parse_args()
    
    inference_dir = Path(args.input)
    if not inference_dir.exists():
        print(f"Error: Directory not found: {inference_dir}")
        return
    
    # Find all candidate plots
    plots = find_candidate_plots(inference_dir, args.target)
    filter_msg = f" (filtered to {args.target})" if args.target else ""
    print(f"Found {len(plots)} candidate plots in {inference_dir}{filter_msg}")
    
    if not plots:
        return
    
    # Show summary by target
    targets = {}
    for target, info, path in plots:
        targets[target] = targets.get(target, 0) + 1
    
    print("\nCandidates per target:")
    for target, count in sorted(targets.items()):
        print(f"  {target}: {count}")
    
    # Create video
    create_video(plots, args.output, args.fps, (args.width, args.height))


if __name__ == '__main__':
    main()
