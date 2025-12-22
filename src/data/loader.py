"""
SRT Data Loader.

Load HDF5 observation files from the Sardinian Radio Telescope.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from blimpy import Waterfall
import warnings


@dataclass
class Observation:
    """Single observation data container."""
    data: np.ndarray          # Shape: (time, freq) = (16, n_freq)
    freq_start: float         # Start frequency (MHz)
    freq_end: float           # End frequency (MHz)
    freq_resolution: float    # Hz per channel
    source_name: str
    file_path: str
    is_on: bool               # True for ON observation, False for OFF
    
    @property
    def n_freq(self) -> int:
        return self.data.shape[1]
    
    @property
    def n_time(self) -> int:
        return self.data.shape[0]
    
    @property
    def bandwidth_mhz(self) -> float:
        return abs(self.freq_end - self.freq_start)


@dataclass
class Cadence:
    """Full 6-observation cadence container."""
    observations: List[Observation] = field(default_factory=list)
    source_name: str = ""
    
    # Standard ON/OFF indices for ABACAD pattern
    ON_INDICES: List[int] = field(default_factory=lambda: [0, 2, 4])
    OFF_INDICES: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    def __post_init__(self):
        if len(self.observations) == 6 and not self.source_name:
            self.source_name = self.observations[0].source_name
    
    @property
    def on_observations(self) -> List[Observation]:
        """Get ON observations (A1, A2, A3)."""
        return [self.observations[i] for i in self.ON_INDICES]
    
    @property
    def off_observations(self) -> List[Observation]:
        """Get OFF observations (B, C, D)."""
        return [self.observations[i] for i in self.OFF_INDICES]
    
    def to_array(self) -> np.ndarray:
        """Stack all observations into array of shape (6, time, freq)."""
        return np.stack([obs.data for obs in self.observations], axis=0)
    
    def is_valid(self) -> bool:
        """Check if cadence has exactly 6 observations."""
        return len(self.observations) == 6


class SRTDataLoader:
    """
    Load SRT HDF5 observation files.
    
    Handles individual observations and full cadences.
    Supports frequency windowing and metadata extraction.
    """
    
    def __init__(self, time_bins: int = 16):
        """
        Initialize loader.
        
        Args:
            time_bins: Expected number of time integrations (default: 16)
        """
        self.time_bins = time_bins
    
    def load_observation(self, 
                         file_path: Union[str, Path],
                         f_start: Optional[float] = None,
                         f_stop: Optional[float] = None,
                         is_on: bool = True) -> Observation:
        """
        Load a single observation from HDF5 file.
        
        Args:
            file_path: Path to .h5 file
            f_start: Start frequency in MHz (optional, loads full band if not specified)
            f_stop: Stop frequency in MHz (optional)
            is_on: Whether this is an ON observation
            
        Returns:
            Observation dataclass with data and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load with blimpy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if f_start is not None and f_stop is not None:
                wf = Waterfall(str(file_path), f_start=f_start, f_stop=f_stop)
            else:
                wf = Waterfall(str(file_path))
        
        # Extract data - squeeze polarization dimension
        data = wf.data.squeeze()  # (time, freq)
        
        # Ensure correct shape
        if data.ndim == 1:
            data = data[np.newaxis, :]
        
        # Get metadata
        header = wf.header
        freq_resolution = abs(header.get('foff', 2.79e-6)) * 1e6  # Convert to Hz
        source_name = header.get('source_name', 'Unknown')
        
        # Calculate frequency range
        fch1 = header.get('fch1', 0)
        foff = header.get('foff', 0)
        nchans = data.shape[1]
        
        if foff < 0:
            freq_start = fch1 + foff * nchans
            freq_end = fch1
        else:
            freq_start = fch1
            freq_end = fch1 + foff * nchans
        
        return Observation(
            data=data.astype(np.float64),
            freq_start=freq_start,
            freq_end=freq_end,
            freq_resolution=freq_resolution,
            source_name=source_name,
            file_path=str(file_path),
            is_on=is_on
        )
    
    def load_cadence(self, 
                     file_paths: List[Union[str, Path]],
                     f_start: Optional[float] = None,
                     f_stop: Optional[float] = None) -> Cadence:
        """
        Load a full 6-observation cadence.
        
        Args:
            file_paths: List of 6 .h5 file paths in order [ON, OFF, ON, OFF, ON, OFF]
            f_start: Start frequency in MHz (optional)
            f_stop: Stop frequency in MHz (optional)
            
        Returns:
            Cadence dataclass with all observations
        """
        if len(file_paths) != 6:
            raise ValueError(f"Cadence requires exactly 6 files, got {len(file_paths)}")
        
        observations = []
        on_off_pattern = [True, False, True, False, True, False]  # ABACAD
        
        for i, (path, is_on) in enumerate(zip(file_paths, on_off_pattern)):
            obs = self.load_observation(path, f_start, f_stop, is_on=is_on)
            observations.append(obs)
            print(f"  Loaded obs {i+1}/6: {'ON ' if is_on else 'OFF'} - {Path(path).name}")
        
        return Cadence(observations=observations)
    
    def get_file_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get metadata from an observation file without loading full data.
        
        Args:
            file_path: Path to .h5 file
            
        Returns:
            Dictionary with file metadata
        """
        file_path = Path(file_path)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf = Waterfall(str(file_path), load_data=False)
        
        header = wf.header
        
        return {
            'source_name': header.get('source_name', 'Unknown'),
            'fch1': header.get('fch1', 0),
            'foff': header.get('foff', 0),
            'nchans': header.get('nchans', 0),
            'tsamp': header.get('tsamp', 0),
            'tstart': header.get('tstart', 0),
            'file_path': str(file_path),
            'freq_resolution_hz': abs(header.get('foff', 0)) * 1e6,
        }
    
    def find_cadence_files(self, 
                           directory: Union[str, Path],
                           source_name: Optional[str] = None) -> List[List[Path]]:
        """
        Find cadence file groups in a directory.
        
        Args:
            directory: Directory to search
            source_name: Filter by source name (optional)
            
        Returns:
            List of cadence file groups (each group is 6 files)
        """
        directory = Path(directory)
        h5_files = sorted(directory.glob("*.h5"))
        
        # Group files by source (simple approach - assumes naming convention)
        # This may need customization based on SRT file naming
        cadences = []
        
        # For now, just group consecutive files by 6
        for i in range(0, len(h5_files) - 5, 6):
            cadence_files = h5_files[i:i+6]
            if len(cadence_files) == 6:
                cadences.append(cadence_files)
        
        return cadences
