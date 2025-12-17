"""
Cadence generation for SETI training data.

Creates True, False, and SingleShot training samples by combining
noise backgrounds with signal injections following the SETI ON-OFF pattern.
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from .noise_generator import NoiseGenerator, NoiseParams
from .signal_generator import SignalGenerator, SignalParams, check_intersection


@dataclass
class CadenceParams:
    """Parameters for cadence generation."""
    tchans: int = 16              # Time channels per observation
    fchans: int = 4096            # Frequency channels
    num_observations: int = 6     # Cadence length
    
    # Signal parameters
    snr_base: float = 20
    snr_range: float = 40
    
    # Noise parameters  
    df: float = 2.7939677238464355
    dt: float = 18.25361108
    noise_mean: float = 58348559


class CadenceGenerator:
    """
    Generator for SETI training cadences.
    
    Creates three types of training samples:
    - True: ETI signal present in ON observations only
    - False: RFI pattern (same signal in all observations) or pure noise
    - SingleShot: Single signal injection for testing
    
    The ON-OFF pattern is:
    [A1, B, A2, C, A3, D] where A1,A2,A3 are ON (target) and B,C,D are OFF (reference)
    """
    
    def __init__(self, 
                 params: Optional[CadenceParams] = None,
                 plate: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        Initialize cadence generator.
        
        Args:
            params: Cadence generation parameters
            plate: Optional real observation plate. If None, uses synthetic noise.
                   Shape should be (num_samples, 6, tchans, fchans)
            seed: Random seed for reproducibility
        """
        self.params = params or CadenceParams()
        self.plate = plate
        self.rng = np.random.default_rng(seed)
        
        # Initialize sub-generators
        noise_params = NoiseParams(
            fchans=self.params.fchans,
            tchans=self.params.tchans,
            df=self.params.df,
            dt=self.params.dt,
            noise_mean=self.params.noise_mean
        )
        self.noise_gen = NoiseGenerator(noise_params)
        
        signal_params = SignalParams(
            df=self.params.df,
            dt=self.params.dt,
            snr_base=self.params.snr_base,
            snr_range=self.params.snr_range
        )
        self.signal_gen = SignalGenerator(signal_params, seed)
    
    def _get_background(self) -> np.ndarray:
        """
        Get a background cadence from plate or generate noise.
        
        Returns:
            Background array of shape (6, tchans, fchans)
        """
        if self.plate is not None:
            # Sample from real observations
            idx = self.rng.integers(0, self.plate.shape[0])
            return self.plate[idx].copy()
        else:
            # Generate synthetic noise
            return self.noise_gen.generate_cadence(
                self.params.num_observations,
                self.params.fchans,
                self.params.tchans
            )
    
    def _stack_cadence(self, cadence: np.ndarray) -> np.ndarray:
        """
        Stack cadence observations vertically.
        
        Args:
            cadence: Array of shape (6, tchans, fchans)
            
        Returns:
            Stacked array of shape (6*tchans, fchans)
        """
        stacked = np.zeros((self.params.num_observations * self.params.tchans, 
                           self.params.fchans))
        for i in range(self.params.num_observations):
            start = i * self.params.tchans
            end = (i + 1) * self.params.tchans
            stacked[start:end, :] = cadence[i]
        return stacked
    
    def _unstack_cadence(self, stacked: np.ndarray) -> np.ndarray:
        """
        Unstack vertically stacked data back to cadence format.
        
        Args:
            stacked: Array of shape (6*tchans, fchans)
            
        Returns:
            Cadence array of shape (6, tchans, fchans)
        """
        cadence = np.zeros((self.params.num_observations, 
                           self.params.tchans, 
                           self.params.fchans))
        for i in range(self.params.num_observations):
            start = i * self.params.tchans
            end = (i + 1) * self.params.tchans
            cadence[i] = stacked[start:end, :]
        return cadence
    
    def create_true_sample(self, 
                           snr: Optional[float] = None,
                           factor: float = 1.0,
                           ensure_non_crossing: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Create a TRUE sample (ETI signal pattern).
        
        A true sample has a signal that appears in ON observations (A1, A2, A3)
        but not in OFF observations (B, C, D). This is achieved by injecting
        two signals with different trajectories:
        - First signal: appears in all observations (like an ETI signal)
        - Second signal: injected only to create the ON-only pattern
        
        The result is that ON observations have both signals (different patterns)
        while OFF observations only have one, creating distinguishable
        representations in latent space.
        
        Args:
            snr: Signal SNR. If None, randomly sampled.
            factor: SNR multiplier for second signal
            ensure_non_crossing: Ensure the two signals don't cross
            
        Returns:
            Tuple of (cadence array shape (6, tchans, fchans), metadata dict)
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)
        
        if snr is None:
            snr = self.rng.uniform(self.params.snr_base, 
                                   self.params.snr_base + self.params.snr_range)
        
        # Inject first signal (full cadence)
        if ensure_non_crossing:
            max_attempts = 100
            for _ in range(max_attempts):
                injected1, info1 = self.signal_gen.inject_cadence_signal(stacked, snr)
                injected2, info2 = self.signal_gen.inject_cadence_signal(injected1, snr * factor)
                
                if check_intersection(info1['slope'], info2['slope'],
                                      info1['intercept'], info2['intercept']):
                    break
            else:
                # Fallback without checking
                injected1, info1 = self.signal_gen.inject_cadence_signal(stacked, snr)
                injected2, info2 = self.signal_gen.inject_cadence_signal(injected1, snr * factor)
        else:
            injected1, info1 = self.signal_gen.inject_cadence_signal(stacked, snr)
            injected2, info2 = self.signal_gen.inject_cadence_signal(injected1, snr * factor)
        
        # Create output: ON gets injected2 (both signals), OFF gets injected1 (one signal)
        result = np.zeros((6, self.params.tchans, self.params.fchans))
        
        # ON observations (A1, A2, A3) - indices 0, 2, 4
        result[0] = injected2[0:16, :]
        result[2] = injected2[32:48, :]
        result[4] = injected2[64:80, :]
        
        # OFF observations (B, C, D) - indices 1, 3, 5
        result[1] = injected1[16:32, :]
        result[3] = injected1[48:64, :]
        result[5] = injected1[80:96, :]
        
        metadata = {
            'sample_type': 'true',
            'snr': snr,
            'signal1': info1,
            'signal2': info2
        }
        
        return result, metadata
    
    def create_true_sample_fast(self, 
                                snr: Optional[float] = None,
                                factor: float = 1.0) -> np.ndarray:
        """
        Fast version of create_true_sample without intersection checking.
        
        For training efficiency when some crossing signals are acceptable.
        
        Args:
            snr: Signal SNR
            factor: SNR multiplier for second signal
            
        Returns:
            Cadence array of shape (6, tchans, fchans)
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)
        
        if snr is None:
            snr = self.rng.uniform(self.params.snr_base, 
                                   self.params.snr_base + self.params.snr_range)
        
        injected1, _ = self.signal_gen.inject_cadence_signal(stacked, snr)
        injected2, _ = self.signal_gen.inject_cadence_signal(injected1, snr * factor)
        
        result = np.zeros((6, self.params.tchans, self.params.fchans))
        result[0] = injected2[0:16, :]
        result[2] = injected2[32:48, :]
        result[4] = injected2[64:80, :]
        result[1] = injected1[16:32, :]
        result[3] = injected1[48:64, :]
        result[5] = injected1[80:96, :]
        
        return result
    
    def create_false_sample(self, snr: Optional[float] = None) -> np.ndarray:
        """
        Create a FALSE sample (RFI pattern or pure noise).
        
        A false sample is either:
        - RFI pattern: Same signal in all observations (if snr provided, always this)
        - Pure background noise (50% chance, only when snr=None)
        
        Args:
            snr: Signal SNR for RFI case. If provided, always creates RFI pattern.
            
        Returns:
            Cadence array of shape (6, tchans, fchans)
        """
        # If SNR explicitly provided, always create RFI pattern
        # (for fair SNR sensitivity testing)
        if snr is not None:
            background = self._get_background()
            stacked = self._stack_cadence(background)
            injected, _ = self.signal_gen.inject_cadence_signal(stacked, snr)
            return self._unstack_cadence(injected)
        
        # Otherwise, random choice between RFI and pure noise
        choice = self.rng.random()
        
        if choice > 0.5:
            # RFI pattern: same signal in all observations
            background = self._get_background()
            stacked = self._stack_cadence(background)
            
            snr = self.rng.uniform(self.params.snr_base, 
                                   self.params.snr_base + self.params.snr_range)
            
            injected, _ = self.signal_gen.inject_cadence_signal(stacked, snr)
            return self._unstack_cadence(injected)
        else:
            # Pure noise/background
            return self._get_background()
    
    def create_single_shot_sample(self, snr: Optional[float] = None) -> np.ndarray:
        """
        Create a SINGLE SHOT sample (one signal injection).
        
        Used for testing sensitivity - single signal appears in ON observations.
        
        Args:
            snr: Signal SNR
            
        Returns:
            Cadence array of shape (6, tchans, fchans)
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)
        
        if snr is None:
            snr = self.rng.uniform(self.params.snr_base, 
                                   self.params.snr_base + self.params.snr_range)
        
        injected, _ = self.signal_gen.inject_cadence_signal(stacked, snr)
        
        result = np.zeros((6, self.params.tchans, self.params.fchans))
        
        # Signal only in ON observations
        result[0] = injected[0:16, :]
        result[1] = stacked[16:32, :]  # Original background
        result[2] = injected[32:48, :]
        result[3] = stacked[48:64, :]
        result[4] = injected[64:80, :]
        result[5] = stacked[80:96, :]
        
        return result
    
    def generate_batch(self,
                       sample_type: str,
                       batch_size: int,
                       snr_base: Optional[float] = None,
                       snr_range: Optional[float] = None,
                       factor: float = 1.0) -> np.ndarray:
        """
        Generate a batch of samples.
        
        Args:
            sample_type: One of 'true', 'true_fast', 'false', 'single_shot'
            batch_size: Number of samples to generate
            snr_base: Override SNR base
            snr_range: Override SNR range
            factor: SNR factor for true samples
            
        Returns:
            Array of shape (batch_size, 6, tchans, fchans)
        """
        if snr_base is not None:
            self.params.snr_base = snr_base
        if snr_range is not None:
            self.params.snr_range = snr_range
        
        batch = np.zeros((batch_size, self.params.num_observations,
                         self.params.tchans, self.params.fchans))
        
        for i in range(batch_size):
            if sample_type == 'true':
                batch[i], _ = self.create_true_sample(factor=factor)
            elif sample_type == 'true_fast':
                batch[i] = self.create_true_sample_fast(factor=factor)
            elif sample_type == 'false':
                batch[i] = self.create_false_sample()
            elif sample_type == 'single_shot':
                batch[i] = self.create_single_shot_sample()
            else:
                raise ValueError(f"Unknown sample type: {sample_type}")
        
        return batch
