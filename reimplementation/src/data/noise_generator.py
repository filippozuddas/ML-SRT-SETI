"""
Synthetic noise generation for SETI simulations.

Uses setigen to create realistic radio telescope noise backgrounds.
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class NoiseParams:
    """Parameters for noise generation."""
    fchans: int = 4096          # Frequency channels
    tchans: int = 16            # Time channels
    df: float = 2.7939677238464355  # Hz per channel
    dt: float = 18.25361108     # Seconds per time bin
    fch1: float = 6095.214842353016  # MHz center frequency
    noise_mean: float = 58348559    # Chi-squared mean
    noise_type: str = 'chi2'    # Noise distribution type


class NoiseGenerator:
    """
    Generator for synthetic telescope noise backgrounds.
    
    Uses setigen's Frame class to create realistic chi-squared
    distributed noise that mimics radio telescope observations.
    """
    
    def __init__(self, params: Optional[NoiseParams] = None):
        """
        Initialize noise generator.
        
        Args:
            params: Noise generation parameters. If None, uses defaults.
        """
        self.params = params or NoiseParams()
    
    def generate_frame(self, 
                       fchans: Optional[int] = None,
                       tchans: Optional[int] = None,
                       noise_mean: Optional[float] = None) -> np.ndarray:
        """
        Generate a single noise frame.
        
        Args:
            fchans: Override frequency channels
            tchans: Override time channels  
            noise_mean: Override noise mean
            
        Returns:
            Noise array of shape (tchans, fchans)
        """
        fchans = fchans or self.params.fchans
        tchans = tchans or self.params.tchans
        noise_mean = noise_mean or self.params.noise_mean
        
        frame = stg.Frame(
            fchans=fchans * u.pixel,
            tchans=tchans * u.pixel,
            df=self.params.df * u.Hz,
            dt=self.params.dt * u.s,
            fch1=self.params.fch1 * u.MHz
        )
        
        frame.add_noise(x_mean=noise_mean, noise_type=self.params.noise_type)
        
        return frame.data
    
    def generate_cadence(self, 
                         num_observations: int = 6,
                         fchans: Optional[int] = None,
                         tchans: Optional[int] = None) -> np.ndarray:
        """
        Generate a full cadence of noise observations.
        
        Args:
            num_observations: Number of observations (default: 6 for SETI)
            fchans: Override frequency channels
            tchans: Override time channels
            
        Returns:
            Array of shape (num_observations, tchans, fchans)
        """
        fchans = fchans or self.params.fchans
        tchans = tchans or self.params.tchans
        
        cadence = np.zeros((num_observations, tchans, fchans))
        
        for i in range(num_observations):
            cadence[i] = self.generate_frame(fchans, tchans)
        
        return cadence
    
    def generate_batch(self,
                       batch_size: int,
                       num_observations: int = 6,
                       fchans: Optional[int] = None,
                       tchans: Optional[int] = None) -> np.ndarray:
        """
        Generate a batch of noise cadences.
        
        Args:
            batch_size: Number of cadences to generate
            num_observations: Observations per cadence
            fchans: Override frequency channels
            tchans: Override time channels
            
        Returns:
            Array of shape (batch_size, num_observations, tchans, fchans)
        """
        fchans = fchans or self.params.fchans
        tchans = tchans or self.params.tchans
        
        batch = np.zeros((batch_size, num_observations, tchans, fchans))
        
        for i in range(batch_size):
            batch[i] = self.generate_cadence(num_observations, fchans, tchans)
        
        return batch
    
    def generate_stacked_frame(self, 
                               num_observations: int = 6,
                               fchans: Optional[int] = None,
                               tchans: Optional[int] = None) -> Tuple[np.ndarray, stg.Frame]:
        """
        Generate stacked noise for signal injection.
        
        Creates a single large frame representing the full cadence
        stacked vertically, suitable for signal injection that spans
        across all observations.
        
        Args:
            num_observations: Number of observations
            fchans: Frequency channels per observation
            tchans: Time channels per observation
            
        Returns:
            Tuple of (stacked data array, setigen Frame for injection)
        """
        fchans = fchans or self.params.fchans
        tchans = tchans or self.params.tchans
        
        total_tchans = tchans * num_observations
        
        # Create large frame for stacked data
        frame = stg.Frame(
            fchans=fchans * u.pixel,
            tchans=total_tchans * u.pixel,
            df=self.params.df * u.Hz,
            dt=self.params.dt * u.s,
            fch1=self.params.fch1 * u.MHz
        )
        
        frame.add_noise(x_mean=self.params.noise_mean, noise_type=self.params.noise_type)
        
        return frame.data, frame


def create_noise_plate(num_samples: int,
                       num_observations: int = 6,
                       tchans: int = 16,
                       fchans: int = 4096,
                       noise_params: Optional[NoiseParams] = None) -> np.ndarray:
    """
    Create a "plate" of noise samples for training.
    
    This is similar to the real observation plates used in the original
    implementation, but with purely synthetic noise.
    
    Args:
        num_samples: Number of noise samples to generate
        num_observations: Observations per cadence
        tchans: Time channels per observation
        fchans: Frequency channels
        noise_params: Optional noise parameters
        
    Returns:
        Array of shape (num_samples, num_observations, tchans, fchans)
    """
    generator = NoiseGenerator(noise_params)
    return generator.generate_batch(num_samples, num_observations, fchans, tchans)
