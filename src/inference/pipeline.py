"""
SRT Inference Pipeline.

End-to-end inference from HDF5 files to classification results.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import warnings

from ..data.loader import SRTDataLoader, Observation, Cadence
from ..utils.preprocessing import preprocess, downscale, normalize_log


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
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON/CSV export."""
        return {
            'source_name': self.source_name,
            'n_snippets': len(self.snippets),
            'n_detections': self.n_detections,
            'top_candidates': [
                {
                    'freq_mhz': s.freq_mhz,
                    'probability': s.eti_probability,
                    'center_channel': s.center_channel
                }
                for s in self.top_candidates
            ]
        }


class SRTPipeline:
    """
    End-to-end SRT inference pipeline.
    
    Loads observations, preprocesses to 512-channel snippets,
    encodes with VAE, and classifies with Random Forest.
    """
    
    # Configuration
    SNIPPET_WIDTH = 4096
    DOWNSAMPLE_FACTOR = 8
    FINAL_FREQ_BINS = SNIPPET_WIDTH // DOWNSAMPLE_FACTOR  # 512
    TIME_BINS = 16
    LATENT_DIM = 8
    ETI_THRESHOLD = 0.9
    
    def __init__(self,
                 encoder_path: Union[str, Path],
                 classifier_path: Union[str, Path],
                 threshold: float = 0.9,
                 verbose: bool = True):
        """
        Initialize pipeline.
        
        Args:
            encoder_path: Path to encoder model (.keras or .h5)
            classifier_path: Path to Random Forest classifier (.joblib)
            threshold: ETI detection threshold (default: 0.5)
            verbose: Print progress messages
        """
        self.threshold = threshold
        self.verbose = verbose
        self.loader = SRTDataLoader()
        
        # Load models
        if self.verbose:
            print("Loading models...")
        
        self.encoder = self._load_encoder(encoder_path)
        self.classifier = joblib.load(classifier_path)
        
        if self.verbose:
            print(f"  Encoder: {encoder_path}")
            print(f"  Classifier: {classifier_path}")
            print(f"  ETI threshold: {threshold}")
    
    def _load_encoder(self, path: Union[str, Path]) -> tf.keras.Model:
        """Load encoder model with error handling."""
        path = Path(path)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if path.suffix == '.h5':
                return tf.keras.models.load_model(str(path), compile=False)
            else:
                return tf.keras.models.load_model(str(path))
    
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def extract_snippets(self, 
                         cadence: Cadence,
                         center_channels: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Extract 4096-channel snippets from cadence.
        
        Args:
            cadence: Loaded cadence data
            center_channels: List of channel indices to extract around.
                           If None, extracts all non-overlapping snippets.
                           
        Returns:
            List of preprocessed snippet arrays, each of shape (6, 16, 512, 1)
        """
        # Get data as array (6, time, freq)
        data = cadence.to_array()
        n_freq = data.shape[2]
        
        if center_channels is None:
            # Extract all non-overlapping snippets
            n_snippets = n_freq // self.SNIPPET_WIDTH
            center_channels = [
                i * self.SNIPPET_WIDTH + self.SNIPPET_WIDTH // 2 
                for i in range(n_snippets)
            ]
        
        snippets = []
        
        for center in center_channels:
            # Calculate snippet boundaries
            start = center - self.SNIPPET_WIDTH // 2
            end = center + self.SNIPPET_WIDTH // 2
            
            # Skip if out of bounds
            if start < 0 or end > n_freq:
                continue
            
            # Extract snippet for all 6 observations
            snippet = data[:, :, start:end]  # (6, 16, 4096)
            
            # Downsample frequency by 8x
            snippet_ds = self._downsample(snippet)  # (6, 16, 512)
            
            # Apply log normalization per observation
            snippet_norm = self._normalize(snippet_ds)  # (6, 16, 512)
            
            # Add channel dimension
            snippet_final = snippet_norm[..., np.newaxis]  # (6, 16, 512, 1)
            
            snippets.append((center, snippet_final))
        
        return snippets
    
    def _downsample(self, data: np.ndarray) -> np.ndarray:
        """Downsample frequency axis by factor 8."""
        # data shape: (6, 16, 4096)
        new_freq = data.shape[2] // self.DOWNSAMPLE_FACTOR
        return data.reshape(data.shape[0], data.shape[1], new_freq, self.DOWNSAMPLE_FACTOR).mean(axis=3)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply log normalization to each observation."""
        result = np.zeros_like(data)
        for i in range(6):
            obs_data = data[i].astype(np.float64)
            obs_data = np.log(np.abs(obs_data) + 1e-10)
            obs_data = obs_data - obs_data.min()
            max_val = obs_data.max()
            if max_val > 0:
                obs_data = obs_data / max_val
            result[i] = obs_data
        return result
    
    def encode_snippet(self, snippet: np.ndarray) -> np.ndarray:
        """
        Encode a snippet through VAE to get latent features.
        
        Args:
            snippet: Array of shape (6, 16, 512, 1)
            
        Returns:
            Latent feature vector of shape (48,) for classifier
        """
        # Encoder expects (batch, 16, 512, 1)
        outputs = self.encoder.predict(snippet, verbose=0)
        
        # Get z (sampled latent)
        if isinstance(outputs, list):
            z = outputs[2]  # (6, 8)
        else:
            z = outputs
        
        # Flatten to 48D feature vector
        return z.flatten()  # (48,)
    
    def classify_snippet(self, latent_features: np.ndarray) -> Tuple[float, bool]:
        """
        Classify a snippet based on latent features.
        
        Args:
            latent_features: 48D feature vector
            
        Returns:
            Tuple of (ETI probability, is_ETI boolean)
        """
        proba = self.classifier.predict_proba(latent_features.reshape(1, -1))[0]
        eti_prob = proba[1] if len(proba) > 1 else proba[0]
        return eti_prob, eti_prob > self.threshold
    
    def process_cadence(self,
                        file_paths: List[Union[str, Path]],
                        f_start: Optional[float] = None,
                        f_stop: Optional[float] = None,
                        center_channels: Optional[List[int]] = None) -> CadenceResult:
        """
        Process a full cadence end-to-end.
        
        Args:
            file_paths: List of 6 .h5 file paths [ON, OFF, ON, OFF, ON, OFF]
            f_start: Start frequency in MHz (optional)
            f_stop: Stop frequency in MHz (optional)
            center_channels: Specific channels to analyze (optional)
            
        Returns:
            CadenceResult with all snippet results
        """
        self._log(f"\n{'='*60}")
        self._log("SRT PIPELINE - PROCESSING CADENCE")
        self._log(f"{'='*60}")
        
        # Load cadence
        self._log("\nLoading observations...")
        cadence = self.loader.load_cadence(file_paths, f_start, f_stop)
        
        # Extract snippets
        self._log(f"\nExtracting {self.SNIPPET_WIDTH}-channel snippets...")
        snippets_data = self.extract_snippets(cadence, center_channels)
        self._log(f"  Found {len(snippets_data)} snippets")
        
        # Process each snippet
        self._log("\nProcessing snippets...")
        results = []
        
        for i, (center_ch, snippet) in enumerate(snippets_data):
            # Encode
            latent_features = self.encode_snippet(snippet)
            
            # Classify
            eti_prob, is_eti = self.classify_snippet(latent_features)
            
            # Calculate frequency
            obs = cadence.observations[0]
            freq_per_ch = (obs.freq_end - obs.freq_start) / obs.n_freq
            freq_mhz = obs.freq_start + center_ch * freq_per_ch
            
            results.append(SnippetResult(
                eti_probability=eti_prob,
                is_eti=is_eti,
                center_channel=center_ch,
                freq_mhz=freq_mhz,
                latent_features=latent_features
            ))
            
            if self.verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(snippets_data)} snippets...")
        
        result = CadenceResult(
            source_name=cadence.source_name,
            snippets=results
        )
        
        # Summary
        self._log(f"\n{'='*60}")
        self._log("RESULTS SUMMARY")
        self._log(f"{'='*60}")
        self._log(f"  Total snippets: {len(results)}")
        self._log(f"  ETI detections: {result.n_detections}")
        
        if result.n_detections > 0:
            self._log("\n  Top candidates:")
            for i, s in enumerate(result.top_candidates[:5]):
                self._log(f"    {i+1}. {s.freq_mhz:.6f} MHz - P(ETI)={s.eti_probability:.3f}")
        
        return result
    
    def process_single_snippet(self,
                               file_paths: List[Union[str, Path]],
                               center_channel: int,
                               f_start: Optional[float] = None,
                               f_stop: Optional[float] = None) -> SnippetResult:
        """
        Process a single snippet at specified channel.
        
        Args:
            file_paths: List of 6 .h5 file paths
            center_channel: Center channel for snippet extraction
            f_start: Start frequency in MHz (optional)
            f_stop: Stop frequency in MHz (optional)
            
        Returns:
            SnippetResult for the specified channel
        """
        cadence = self.loader.load_cadence(file_paths, f_start, f_stop)
        snippets_data = self.extract_snippets(cadence, [center_channel])
        
        if not snippets_data:
            raise ValueError(f"Could not extract snippet at channel {center_channel}")
        
        center_ch, snippet = snippets_data[0]
        latent_features = self.encode_snippet(snippet)
        eti_prob, is_eti = self.classify_snippet(latent_features)
        
        obs = cadence.observations[0]
        freq_per_ch = (obs.freq_end - obs.freq_start) / obs.n_freq
        freq_mhz = obs.freq_start + center_channel * freq_per_ch
        
        return SnippetResult(
            eti_probability=eti_prob,
            is_eti=is_eti,
            center_channel=center_channel,
            freq_mhz=freq_mhz,
            latent_features=latent_features
        )
