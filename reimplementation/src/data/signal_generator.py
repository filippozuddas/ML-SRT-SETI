"""
Signal injection for SETI simulations.

Uses setigen to inject synthetic narrowband drifting signals
into spectrogram data.
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SignalParams:
    """Parameters for signal injection."""
    df: float = 2.7939677238464355  # Hz per channel
    dt: float = 18.25361108         # Seconds per time bin
    fch1: float = 0                 # MHz (0 for injection on existing data)
    
    # SNR parameters
    snr_min: float = 10
    snr_max: float = 80
    snr_base: float = 20
    snr_range: float = 40
    
    # Signal width parameters
    width_base: float = 50          # Base width in Hz
    width_drift_factor: float = 18  # Additional width per Hz/s drift


class SignalGenerator:
    """
    Generator for synthetic SETI signals.
    
    Creates narrowband drifting signals using setigen for injection
    into noise or real observation backgrounds.
    """
    
    def __init__(self, params: Optional[SignalParams] = None, seed: Optional[int] = None):
        """
        Initialize signal generator.
        
        Args:
            params: Signal generation parameters. If None, uses defaults.
            seed: Random seed for reproducibility
        """
        self.params = params or SignalParams()
        self.rng = np.random.default_rng(seed)
    
    def _calculate_drift_rate(self, 
                              start_channel: int, 
                              width: int, 
                              total_time_bins: int) -> float:
        """
        Calculate drift rate to traverse the observation.
        
        The drift rate is calculated such that the signal drifts
        from the start position across the full time duration.
        
        Args:
            start_channel: Starting frequency channel
            width: Total frequency width
            total_time_bins: Total time bins in observation
            
        Returns:
            Drift rate in Hz/s
        """
        # FIXED: Use uniform distribution for direction to avoid original bias
        direction = self.rng.choice([-1, 1])
        
        if direction > 0:
            # Positive drift: signal drifts from lower to higher frequencies
            true_slope = total_time_bins / start_channel
        else:
            # Negative drift: signal drifts from higher to lower frequencies
            true_slope = total_time_bins / (start_channel - width)
        
        # Convert slope to drift rate
        # Add small random perturbation for variety
        slope = true_slope * (self.params.dt / self.params.df) + self.rng.uniform(0, 3) * direction
        drift_rate = -1 / slope
        
        return drift_rate, true_slope
    
    def _calculate_width(self, drift_rate: float) -> float:
        """
        Calculate signal width based on drift rate.
        
        Wider signals for faster drift rates to account for
        frequency smearing during integration.
        
        Args:
            drift_rate: Drift rate in Hz/s
            
        Returns:
            Signal width in Hz
        """
        base_width = self.rng.uniform(0, self.params.width_base)
        drift_width = abs(drift_rate) * self.params.width_drift_factor
        return base_width + drift_width
    
    def inject_signal(self,
                      data: np.ndarray,
                      snr: Optional[float] = None,
                      start_channel: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Inject a signal into existing spectrogram data.
        
        Args:
            data: Input spectrogram of shape (time, freq)
            snr: Signal SNR. If None, randomly sampled.
            start_channel: Starting frequency channel. If None, random.
            
        Returns:
            Tuple of (injected data, signal parameters dict)
        """
        tchans, fchans = data.shape
        
        # Random SNR if not specified
        if snr is None:
            snr = self.rng.uniform(self.params.snr_base, 
                                   self.params.snr_base + self.params.snr_range)
        
        # Random start channel ensuring signal stays in band
        if start_channel is None:
            start_channel = self.rng.integers(1, fchans - 1)
        
        # Calculate drift rate and width
        drift_rate, true_slope = self._calculate_drift_rate(start_channel, fchans, tchans)
        width = self._calculate_width(drift_rate)
        
        # Calculate intercept for tracking
        b = tchans - true_slope * start_channel
        
        # Create frame from existing data
        frame = stg.Frame.from_data(
            df=self.params.df * u.Hz,
            dt=self.params.dt * u.s,
            fch1=self.params.fch1 * u.MHz,
            ascending=False,  # Frequencies descend (typical for radio data)
            data=data
        )
        
        # Inject signal
        frame.add_signal(
            stg.constant_path(
                f_start=frame.get_frequency(index=start_channel),
                drift_rate=drift_rate * u.Hz / u.s
            ),
            stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
            stg.gaussian_f_profile(width=width * u.Hz),
            stg.constant_bp_profile(level=1)
        )
        
        signal_info = {
            'snr': snr,
            'drift_rate': drift_rate,
            'start_channel': start_channel,
            'width': width,
            'slope': true_slope,
            'intercept': b
        }
        
        return frame.data, signal_info
    
    def inject_cadence_signal(self,
                              stacked_data: np.ndarray,
                              snr: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """
        Inject a signal that drifts across a full stacked cadence.
        
        The signal is designed to traverse all 6 observations in the
        cadence, appearing in ON observations (A1, A2, A3) and 
        potentially in OFF observations (B, C, D) if it's RFI.
        
        Args:
            stacked_data: Stacked observations of shape (total_time, freq)
                          where total_time = 6 * time_per_obs
            snr: Signal SNR
            
        Returns:
            Tuple of (injected data, signal parameters dict)
        """
        return self.inject_signal(stacked_data, snr)


def check_intersection(m1: float, m2: float, b1: float, b2: float,
                       num_observations: int = 6, 
                       tchans: int = 16) -> bool:
    """
    Check if two signal paths intersect within observation windows.
    
    Used to ensure two injected signals don't cross within the
    active observation regions.
    
    Args:
        m1, m2: Slopes of the two signals
        b1, b2: Y-intercepts of the two signals
        num_observations: Number of observations
        tchans: Time channels per observation
        
    Returns:
        True if signals don't intersect in observation windows
    """
    if m1 == m2:
        return True  # Parallel lines don't intersect
    
    # Find intersection point
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    
    total_time = num_observations * tchans
    
    # Check if intersection is in any observation window
    # ON observations: 0-16, 32-48, 64-80
    # OFF observations: 16-32, 48-64, 80-96
    on_windows = [(0, 16), (32, 48), (64, 80)]
    off_windows = [(16, 32), (48, 64), (80, 96)]
    
    for start, end in on_windows + off_windows:
        if start <= y <= end:
            return False
    
    return True
