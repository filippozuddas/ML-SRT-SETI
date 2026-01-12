"""
Optimized SRT Inference Pipeline with Chunked Processing.

Processes large HDF5 files in frequency chunks to manage memory.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import joblib
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import warnings
from blimpy import Waterfall

from ..utils.preprocessing import preprocess, downscale


@dataclass
class SnippetResult:
    """Result for a single snippet."""
    eti_probability: float
    is_eti: bool
    center_channel: int
    freq_mhz: float
    latent_features: np.ndarray
    
    @property
    def label(self) -> str:
        return "ETI" if self.is_eti else "RFI/Noise"


@dataclass
class CadenceResult:
    """Result for a full cadence analysis."""
    source_name: str
    snippets: List[SnippetResult] = field(default_factory=list)
    
    @property
    def n_detections(self) -> int:
        return sum(1 for s in self.snippets if s.is_eti)
    
    @property
    def top_candidates(self) -> List[SnippetResult]:
        """Get top 10 candidates sorted by probability."""
        return sorted(self.snippets, key=lambda x: x.eti_probability, reverse=True)[:10]
    
    def get_candidates(self, threshold: float = 0.5) -> List[SnippetResult]:
        """Get all candidates above threshold."""
        return [s for s in self.snippets if s.eti_probability >= threshold]


class SRTPipelineOptimized:
    """
    Optimized inference pipeline for SRT data.
    
    Processes large HDF5 files in frequency chunks to manage memory
    while maintaining full cadence analysis capability.
    """
    
    SNIPPET_WIDTH = 4096      # Channels per snippet
    FINAL_FREQ_BINS = 512     # After downsampling
    DOWNSAMPLE_FACTOR = 8     # 4096 / 512
    
    def __init__(self, 
                 encoder_path: Union[str, Path],
                 classifier_path: Union[str, Path] = None,
                 rf_path: Union[str, Path] = None,
                 threshold: float = 0.5,
                 n_chunks: int = 8,
                 batch_size: int = 256,
                 overlap: bool = False,
                 verbose: bool = True):
        """
        Initialize the optimized pipeline.
        
        Args:
            encoder_path: Path to VAE encoder model
            classifier_path: Path to Random Forest classifier (alias for rf_path)
            rf_path: Path to Random Forest classifier
            threshold: Classification threshold
            n_chunks: Number of chunks to split frequency band into
            batch_size: Batch size for encoding
            overlap: Use 50% overlapping windows for better signal coverage
            verbose: Print progress messages
        """
        # Support both parameter names
        rf_path = classifier_path or rf_path
        if rf_path is None:
            raise ValueError("Must provide classifier_path or rf_path")
        
        self.threshold = threshold
        self.n_chunks = n_chunks
        self.overlap = overlap
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Load models
        if self.verbose:
            print("Loading models...")
        self.encoder = tf.keras.models.load_model(encoder_path)
        self.rf = joblib.load(rf_path)
        if self.verbose:
            print(f"  Encoder: {encoder_path}")
            print(f"  Classifier: {rf_path}")
            print(f"  Chunks: {n_chunks}")
            print(f"  Batch size: {batch_size}")
    
    def process_cadence(self, 
                        cadence_files: List[Union[str, Path]] = None,
                        file_paths: List[Union[str, Path]] = None,
                        source_name: str = "Unknown") -> CadenceResult:
        """
        Process a full cadence with chunked loading.
        
        Args:
            cadence_files: List of 6 HDF5 file paths (ON, OFF, ON, OFF, ON, OFF)
            file_paths: Alias for cadence_files (for CLI compatibility)
            source_name: Name of the source
            
        Returns:
            CadenceResult with all snippet classifications
        """
        # Support both parameter names
        cadence_files = cadence_files or file_paths
        if cadence_files is None:
            raise ValueError("Must provide cadence_files or file_paths")
        
        if len(cadence_files) != 6:
            raise ValueError(f"Expected 6 files, got {len(cadence_files)}")
        
        # Get observation info from first file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = Waterfall(str(cadence_files[0]), load_data=False)
        
        total_channels = wf.header['nchans']
        fch1 = wf.header['fch1']
        foff = wf.header['foff']
        bandwidth = abs(foff) * total_channels
        
        print(f"\n{'='*60}")
        print(f"SRT OPTIMIZED PIPELINE - CHUNKED PROCESSING")
        print(f"{'='*60}")
        print(f"\nObservation info:")
        print(f"  Source: {source_name}")
        print(f"  Channels: {total_channels:,}")
        print(f"  Bandwidth: {bandwidth:.2f} MHz")
        print(f"  Frequency: {fch1:.2f} - {fch1 + total_channels * foff:.2f} MHz")
        
        # Calculate chunk boundaries
        chunk_size = total_channels // self.n_chunks
        print(f"\nProcessing in {self.n_chunks} chunks (~{abs(foff) * chunk_size:.1f} MHz each)")
        
        all_snippets = []
        
        for chunk_idx in range(self.n_chunks):
            # Calculate frequency range for this chunk
            f_start_chan = chunk_idx * chunk_size
            f_end_chan = (chunk_idx + 1) * chunk_size if chunk_idx < self.n_chunks - 1 else total_channels
            
            f_start = fch1 + f_start_chan * foff
            f_end = fch1 + f_end_chan * foff
            
            # Ensure f_start < f_end for loading
            if f_start > f_end:
                f_start, f_end = f_end, f_start
            
            print(f"\n--- Chunk {chunk_idx + 1}/{self.n_chunks}: {f_start:.2f} - {f_end:.2f} MHz ---")
            
            # Load chunk from all 6 files
            print("  Loading...")
            chunk_data = self._load_chunk(cadence_files, f_start, f_end)
            print(f"  Loaded shape: {chunk_data.shape}")
            
            # Extract and process snippets
            # Pass both f_start and f_end for correct frequency calculation
            snippets = self.extract_and_process_snippets(
                chunk_data, f_start, f_end, foff
            )
            print(f"  Snippets: {len(snippets)}")
            
            if len(snippets) == 0:
                continue
            
            # Encode in batches
            print("  Encoding...")
            all_processed = [s[2] for s in snippets]
            latents = self.batch_encode(all_processed)
            
            # Classify
            print("  Classifying...")
            probs = self.rf.predict_proba(latents)[:, 1]
            
            # Create results
            for i, (center_chan, freq_mhz, _) in enumerate(snippets):
                result = SnippetResult(
                    eti_probability=float(probs[i]),
                    is_eti=probs[i] >= self.threshold,
                    center_channel=f_start_chan + center_chan,
                    freq_mhz=freq_mhz,
                    latent_features=latents[i]
                )
                all_snippets.append(result)
            
            # Report progress
            n_candidates = sum(1 for s in all_snippets if s.is_eti)
            print(f"  ETI candidates so far: {n_candidates}")
            
            # Clear memory
            del chunk_data, snippets, all_processed, latents
            gc.collect()
        
        result = CadenceResult(source_name=source_name, snippets=all_snippets)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Total snippets: {len(all_snippets)}")
        print(f"  ETI detections: {result.n_detections}")
        
        if result.n_detections > 0:
            print(f"\n  Top candidates:")
            for i, cand in enumerate(result.top_candidates[:5]):
                print(f"    {i+1}. {cand.freq_mhz:.6f} MHz - P(ETI)={cand.eti_probability:.3f}")
        
        return result
    
    def _load_chunk(self, 
                    cadence_files: List[Union[str, Path]], 
                    f_start: float, 
                    f_end: float) -> np.ndarray:
        """Load a frequency chunk from all 6 observations."""
        cadence_data = []
        
        for filepath in cadence_files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf = Waterfall(str(filepath), f_start=f_start, f_stop=f_end)
            
            data = wf.data.squeeze()
            if data.ndim == 1:
                data = data[np.newaxis, :]
            
            cadence_data.append(data)
        
        return np.stack(cadence_data, axis=0)  # (6, time, freq)
    
    def extract_and_process_snippets(self, chunk_data: np.ndarray, 
                                      chunk_freq_min: float,
                                      chunk_freq_max: float,
                                      foff: float) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract 4096-channel snippets with NON-OVERLAPPING windows.
        
        Args:
            chunk_data: Data array (6, 16, n_freq)
            chunk_freq_min: Minimum frequency of chunk (MHz)
            chunk_freq_max: Maximum frequency of chunk (MHz)
            foff: Frequency offset (can be positive or negative)
            
        Returns: List of (local_channel, freq_mhz, processed_snippet)
        """
        n_freq = chunk_data.shape[2]
        
        snippets = []
        
        # Determine step size based on overlap setting
        if self.overlap:
            step = self.SNIPPET_WIDTH // 2  # 2048 = 50% overlap
        else:
            step = self.SNIPPET_WIDTH  # 4096 = no overlap
        
        start = 0
        while start + self.SNIPPET_WIDTH <= n_freq:
            end = start + self.SNIPPET_WIDTH
            
            # Extract
            snippet = chunk_data[:, :, start:end]  # (6, 16, 4096)
            
            # Use the SAME preprocessing as training:
            # 1. Downscale using shared function
            snippet_ds = downscale(snippet, factor=self.DOWNSAMPLE_FACTOR)  # (6, 16, 512)
            
            # 2. Preprocess using shared function (log normalize + add channel dim)
            processed = preprocess(snippet_ds, add_channel=True)  # (6, 16, 512, 1)
            
            # Calculate frequency (center of snippet)
            # When foff < 0: channel 0 is at MAX frequency, so we subtract
            # When foff > 0: channel 0 is at MIN frequency, so we add
            center_channel = start + self.SNIPPET_WIDTH // 2
            
            if foff < 0:
                # Frequencies decrease with channel
                freq_mhz = chunk_freq_max - center_channel * abs(foff)
            else:
                # Frequencies increase with channel
                freq_mhz = chunk_freq_min + center_channel * abs(foff)
            
            snippets.append((center_channel, freq_mhz, processed))
            
            start += step
        
        return snippets
    
    def batch_encode(self, snippets: List[np.ndarray]) -> np.ndarray:
        """Encode a batch of snippets through VAE."""
        # Stack all observations from all snippets
        all_obs = []
        for snippet in snippets:
            for obs_idx in range(6):
                all_obs.append(snippet[obs_idx])
        
        all_obs = np.array(all_obs)  # (n_snippets * 6, 16, 512, 1)
        
        # Encode in batches
        all_latents = []
        for i in range(0, len(all_obs), self.batch_size):
            batch = all_obs[i:i + self.batch_size]
            outputs = self.encoder.predict(batch, verbose=0)
            
            # Handle different output formats
            if isinstance(outputs, list):
                z = outputs[2]  # Sampling layer output
            else:
                z = outputs
            
            all_latents.append(z)
        
        all_latents = np.concatenate(all_latents, axis=0)
        
        # Reshape to (n_snippets, 6 * latent_dim)
        n_snippets = len(snippets)
        latent_dim = all_latents.shape[1]
        cadence_features = all_latents.reshape(n_snippets, 6 * latent_dim)
        
        return cadence_features


def run_inference(cadence_files: List[str],
                  encoder_path: str,
                  rf_path: str,
                  source_name: str = "Unknown",
                  threshold: float = 0.5,
                  n_chunks: int = 8) -> CadenceResult:
    """
    Run inference on a cadence.
    
    Args:
        cadence_files: List of 6 HDF5 file paths
        encoder_path: Path to encoder model
        rf_path: Path to random forest
        source_name: Source name
        threshold: Classification threshold
        n_chunks: Number of frequency chunks
        
    Returns:
        CadenceResult with classifications
    """
    pipeline = OptimizedSRTPipeline(
        encoder_path=encoder_path,
        rf_path=rf_path,
        threshold=threshold,
        n_chunks=n_chunks
    )
    
    return pipeline.process_cadence(cadence_files, source_name)
