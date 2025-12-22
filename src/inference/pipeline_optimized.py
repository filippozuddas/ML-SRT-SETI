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
from numba import jit, prange

from ..utils.preprocessing import normalize_log


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
        return sorted(self.snippets, key=lambda x: x.eti_probability, reverse=True)[:10]
    
    def to_dict(self) -> dict:
        return {
            'source_name': self.source_name,
            'n_snippets': len(self.snippets),
            'n_detections': self.n_detections,
            'top_candidates': [
                {'freq_mhz': s.freq_mhz, 'probability': s.eti_probability, 'center_channel': s.center_channel}
                for s in self.top_candidates
            ]
        }


class SRTPipelineOptimized:
    """
    Optimized SRT inference pipeline with chunked frequency processing.
    
    Key optimizations:
    - Processes frequency band in chunks (like original code)
    - Uses blimpy f_start/f_stop for partial loading
    - Batch encoding for GPU efficiency
    - Memory cleanup after each chunk
    """
    
    SNIPPET_WIDTH = 4096
    DOWNSAMPLE_FACTOR = 8
    FINAL_FREQ_BINS = SNIPPET_WIDTH // DOWNSAMPLE_FACTOR
    TIME_BINS = 16
    LATENT_DIM = 8
    ETI_THRESHOLD = 0.9
    
    def __init__(self,
                 encoder_path: Union[str, Path],
                 classifier_path: Union[str, Path],
                 threshold: float = 0.9,
                 n_chunks: int = 8,
                 batch_size: int = 256,
                 verbose: bool = True):
        """
        Initialize optimized pipeline.
        
        Args:
            encoder_path: Path to encoder model
            classifier_path: Path to classifier
            threshold: ETI detection threshold
            n_chunks: Number of frequency chunks to process
            batch_size: Batch size for GPU encoding
            verbose: Print progress
        """
        self.threshold = threshold
        self.n_chunks = n_chunks
        self.batch_size = batch_size
        self.verbose = verbose
        
        if self.verbose:
            print("Loading models...")
        
        self.encoder = self._load_encoder(encoder_path)
        self.classifier = joblib.load(classifier_path)
        
        if self.verbose:
            print(f"  Encoder: {encoder_path}")
            print(f"  Classifier: {classifier_path}")
            print(f"  Chunks: {n_chunks}")
            print(f"  Batch size: {batch_size}")
    
    def _load_encoder(self, path: Union[str, Path]) -> tf.keras.Model:
        path = Path(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tf.keras.models.load_model(str(path), compile=False)
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def get_frequency_info(self, file_path: str) -> dict:
        """Get frequency range from file header without loading data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = Waterfall(file_path, load_data=False)
        
        header = wf.header
        fch1 = header['fch1']
        foff = header['foff']
        nchans = header['nchans']
        
        # Calculate start/end frequencies
        if foff < 0:
            f_start = fch1 + foff * nchans
            f_end = fch1
        else:
            f_start = fch1
            f_end = fch1 + foff * nchans
        
        return {
            'f_start': f_start,
            'f_end': f_end,
            'nchans': nchans,
            'foff': foff,
            'bandwidth_mhz': abs(f_end - f_start),
            'source_name': header.get('source_name', 'Unknown')
        }
    
    def calculate_chunks(self, f_start: float, f_end: float) -> List[Tuple[float, float]]:
        """Divide frequency range into n_chunks."""
        bandwidth = f_end - f_start
        chunk_size = bandwidth / self.n_chunks
        
        chunks = []
        for i in range(self.n_chunks):
            chunk_start = f_start + i * chunk_size
            chunk_end = f_start + (i + 1) * chunk_size
            chunks.append((chunk_start, chunk_end))
        
        return chunks
    
    def load_chunk(self, file_paths: List[str], f_start: float, f_stop: float) -> np.ndarray:
        """
        Load a frequency chunk from all 6 observations.
        
        Returns: Array of shape (6, time, freq_chunk)
        """
        cadence_data = []
        
        for i, fp in enumerate(file_paths):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf = Waterfall(fp, f_start=f_start, f_stop=f_stop)
            
            data = wf.data.squeeze()
            if data.ndim == 1:
                data = data[np.newaxis, :]
            
            cadence_data.append(data)
        
        return np.stack(cadence_data, axis=0)  # (6, time, freq)
    
    def extract_and_process_snippets(self, chunk_data: np.ndarray, 
                                      chunk_freq_start: float,
                                      foff: float) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract 4096-channel snippets, downsample and normalize.
        
        Returns: List of (global_channel, freq_mhz, processed_snippet)
        """
        n_freq = chunk_data.shape[2]
        n_snippets = n_freq // self.SNIPPET_WIDTH
        
        snippets = []
        
        for i in range(n_snippets):
            start = i * self.SNIPPET_WIDTH
            end = (i + 1) * self.SNIPPET_WIDTH
            
            # Extract
            snippet = chunk_data[:, :, start:end]  # (6, 16, 4096)
            
            # Downsample
            snippet_ds = snippet.reshape(6, snippet.shape[1], self.FINAL_FREQ_BINS, 
                                         self.DOWNSAMPLE_FACTOR).mean(axis=3)
            
            # Normalize each observation
            processed = np.zeros_like(snippet_ds)
            for j in range(6):
                obs = snippet_ds[j].astype(np.float64)
                obs = np.log(np.abs(obs) + 1e-10)
                obs = obs - obs.min()
                max_val = obs.max()
                if max_val > 0:
                    obs = obs / max_val
                processed[j] = obs
            
            # Add channel dimension
            processed = processed[..., np.newaxis]  # (6, 16, 512, 1)
            
            # Calculate frequency
            center_channel = start + self.SNIPPET_WIDTH // 2
            freq_mhz = chunk_freq_start + center_channel * abs(foff)
            
            snippets.append((center_channel, freq_mhz, processed))
        
        return snippets
    
    def batch_encode(self, snippets: List[np.ndarray]) -> np.ndarray:
        """Encode a batch of snippets through VAE."""
        # Stack all observations from all snippets
        # Each snippet is (6, 16, 512, 1), we need (N*6, 16, 512, 1)
        all_obs = np.concatenate(snippets, axis=0)  # (N*6, 16, 512, 1)
        
        # Encode in batches
        outputs = self.encoder.predict(all_obs, batch_size=self.batch_size, verbose=0)
        
        if isinstance(outputs, list):
            z = outputs[2]
        else:
            z = outputs
        
        return z  # (N*6, 8)
    
    def process_cadence(self, file_paths: List[Union[str, Path]]) -> CadenceResult:
        """
        Process cadence with chunked frequency loading.
        
        Args:
            file_paths: List of 6 .h5 file paths [ON, OFF, ON, OFF, ON, OFF]
        
        Returns:
            CadenceResult with all candidates
        """
        file_paths = [str(p) for p in file_paths]
        
        self._log(f"\n{'='*60}")
        self._log("SRT OPTIMIZED PIPELINE - CHUNKED PROCESSING")
        self._log(f"{'='*60}")
        
        # Get frequency info
        info = self.get_frequency_info(file_paths[0])
        f_start, f_end = info['f_start'], info['f_end']
        
        self._log(f"\nObservation info:")
        self._log(f"  Source: {info['source_name']}")
        self._log(f"  Channels: {info['nchans']:,}")
        self._log(f"  Bandwidth: {info['bandwidth_mhz']:.2f} MHz")
        self._log(f"  Frequency: {f_start:.2f} - {f_end:.2f} MHz")
        
        # Calculate chunks
        chunks = self.calculate_chunks(f_start, f_end)
        self._log(f"\nProcessing in {self.n_chunks} chunks (~{info['bandwidth_mhz']/self.n_chunks:.1f} MHz each)")
        
        all_results = []
        total_snippets = 0
        
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            self._log(f"\n--- Chunk {chunk_idx+1}/{self.n_chunks}: {chunk_start:.2f} - {chunk_end:.2f} MHz ---")
            
            # Load chunk
            self._log(f"  Loading...")
            chunk_data = self.load_chunk(file_paths, chunk_start, chunk_end)
            self._log(f"  Loaded shape: {chunk_data.shape}")
            
            # Extract snippets
            snippets_data = self.extract_and_process_snippets(
                chunk_data, chunk_start, info['foff']
            )
            n_snippets = len(snippets_data)
            total_snippets += n_snippets
            self._log(f"  Snippets: {n_snippets}")
            
            if n_snippets == 0:
                del chunk_data
                gc.collect()
                continue
            
            # Batch encode
            self._log(f"  Encoding...")
            snippet_arrays = [s[2] for s in snippets_data]
            z_all = self.batch_encode(snippet_arrays)
            
            # Classify each snippet
            self._log(f"  Classifying...")
            for i, (center_ch, freq_mhz, _) in enumerate(snippets_data):
                # Get 48D features for this snippet
                z_snippet = z_all[i*6:(i+1)*6, :].flatten()
                
                # Classify
                proba = self.classifier.predict_proba(z_snippet.reshape(1, -1))[0]
                eti_prob = proba[1] if len(proba) > 1 else proba[0]
                is_eti = eti_prob > self.threshold
                
                all_results.append(SnippetResult(
                    eti_probability=eti_prob,
                    is_eti=is_eti,
                    center_channel=center_ch,
                    freq_mhz=freq_mhz,
                    latent_features=z_snippet
                ))
            
            # Cleanup
            del chunk_data, snippets_data, z_all
            gc.collect()
            
            # Progress
            n_eti = sum(1 for r in all_results if r.is_eti)
            self._log(f"  ETI candidates so far: {n_eti}")
        
        result = CadenceResult(
            source_name=info['source_name'],
            snippets=all_results
        )
        
        # Summary
        self._log(f"\n{'='*60}")
        self._log("RESULTS SUMMARY")
        self._log(f"{'='*60}")
        self._log(f"  Total snippets: {len(result.snippets)}")
        self._log(f"  ETI detections: {result.n_detections}")
        
        if result.n_detections > 0:
            self._log("\n  Top candidates:")
            for i, s in enumerate(result.top_candidates[:5]):
                self._log(f"    {i+1}. {s.freq_mhz:.6f} MHz - P(ETI)={s.eti_probability:.3f}")
        
        return result
