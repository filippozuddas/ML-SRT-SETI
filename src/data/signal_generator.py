"""
Signal injection for SETI simulations — RST injection logic.

Ported from RST (rst_seti/data/signal_generator.py) to ML-SRT-SETI for
fair architecture comparison. Key differences from the original ML-SRT-SETI
signal_generator:

  - SNR: log-uniform in [snr_min, snr_max] (was: uniform [20, 60])
  - Drift rate: log-uniform in [min_nonzero, max_drift] with random sign
                (was: corner-targeting geometric computation)
  - Start channel: constrained so signal stays in-frame through ON₂
                   (was: unconstrained)
  - Signal width: split into ETI (narrowband) and RFI (broader) formulae
                  (was: single formula for both)
  - Freq profile: weighted gaussian (80%) / sinc² (20%) (was: always gaussian)
  - Time profile: weighted constant (60%) / scintillating (40%) (was: always constant)
  - RFI types: linear, stationary, random_walk, scintillating (was: only linear)

The fchans default is kept at 4096 (ML-SRT-SETI snippet width) and
max_drift_rate is computed accordingly.
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


# Available RFI types
RFI_TYPES: List[str] = [
    'linear',        # Standard linear drift (same path model as ETI)
    'stationary',    # Fixed frequency with small jitter
    'random_walk',   # Frequency wanders randomly over time
    'scintillating', # Intensity oscillates sinusoidally
]


def compute_max_drift_rate(
    snippet_width: int,
    df: float,
    dt: float,
    n_scans: int = 4,
    bins_per_scan: int = 16,
) -> float:
    """
    Calculate the maximum theoretical drift rate before a signal exits the
    snippet window within n_scans observations.

    Args:
        snippet_width: Width of the spectral window (frequency bins).
        df: Frequency resolution (Hz per bin).
        dt: Time resolution (seconds per bin).
        n_scans: Number of observations the signal must survive (default 4,
                 i.e. through the second ON).
        bins_per_scan: Time bins per observation (default 16).

    Returns:
        Maximum drift rate in Hz/s.
    """
    total_bandwidth_hz = snippet_width * df
    total_time_s = n_scans * bins_per_scan * dt
    return float(total_bandwidth_hz / total_time_s)


@dataclass
class SignalParams:
    """Parameters for signal injection (RST logic, fchans adapted for ML-SRT-SETI)."""
    df: float = 2.7939677238464355   # Hz per channel (SRT native resolution)
    dt: float = 18.25361108          # Seconds per time bin
    fch1: float = 0                  # MHz (0 = inject on existing data)

    # SNR — log-uniform sampling
    snr_min: float = 10.0
    snr_max: float = 50.0

    # ETI width: |DR|×dt + U(offset_min, offset_max)  [narrowband]
    eti_width_offset_min: float = 1.0    # Hz
    eti_width_offset_max: float = 10.0   # Hz

    # RFI width: |DR|×dt + U(offset_min, offset_max)  [broader]
    rfi_width_offset_min: float = 1.0    # Hz
    rfi_width_offset_max: float = 55.0   # Hz

    # Drift rate — log-uniform in [min_nonzero, max_drift] with random sign
    # max_drift_rate is computed for fchans=4096 (ML-SRT-SETI snippet width)
    max_drift_rate: float = field(
        default_factory=lambda: compute_max_drift_rate(
            snippet_width=4096, df=2.7939677238464355, dt=18.25361108, n_scans=4
        )
    )
    min_nonzero_drift: float = 0.01  # Hz/s — minimum non-zero drift rate
    zero_drift_prob: float = 0.05    # Probability of exactly zero drift

    # Frequency profile: gaussian (80%) or sinc² (20%)
    freq_profiles: tuple = ('gaussian', 'sinc2')
    freq_profile_weights: tuple = (0.8, 0.2)

    # Temporal profile: constant (60%) or scintillating (40%)
    time_profiles: tuple = ('constant', 'scintillating')
    time_profile_weights: tuple = (0.6, 0.4)

    # RFI type weights (linear, stationary, random_walk, scintillating)
    rfi_types: tuple = ('linear', 'stationary', 'random_walk', 'scintillating')
    rfi_type_weights: tuple = (0.40, 0.15, 0.25, 0.20)


class SignalGenerator:
    """
    Generator for synthetic SETI signals — RST injection logic.

    Methods:
        inject_signal          — ETI-like narrowband drifting signal (True samples)
        inject_rfi_signal      — Diverse RFI patterns (False samples)
        inject_cadence_signal  — Thin wrapper over inject_signal for stacked data

    Sampling strategies (all matching RST):
        SNR:        log-uniform in [snr_min, snr_max]
        Drift rate: log-uniform in [min_nonzero, max_drift] with random sign
        Freq profile: weighted random (gaussian 80%, sinc² 20%)
        Time profile: weighted random (constant 60%, scintillating 40%)
    """

    # Minimum time bin through which the ETI signal must still be in-frame.
    # This corresponds to the end of ON₂ in a 6-obs cadence (16+16+16-1 = 47).
    _MIN_VISIBLE_BIN = 47

    def __init__(self, params: Optional[SignalParams] = None, seed: Optional[int] = None):
        self.params = params or SignalParams()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_snr(self) -> float:
        """Log-uniform SNR in [snr_min, snr_max]. Produces more low-SNR samples."""
        log_min = np.log10(self.params.snr_min)
        log_max = np.log10(self.params.snr_max)
        return float(10 ** self.rng.uniform(log_min, log_max))

    def _sample_drift_rate(self) -> Tuple[float, float]:
        """
        Log-uniform drift rate in [min_nonzero, max_drift] with random sign.
        5% chance of exactly zero drift.

        Returns:
            (drift_rate_hz_per_s, true_slope)
        """
        if self.rng.random() < self.params.zero_drift_prob:
            drift_rate = 0.0
        else:
            log_min = np.log10(self.params.min_nonzero_drift)
            log_max = np.log10(self.params.max_drift_rate)
            magnitude = 10 ** self.rng.uniform(log_min, log_max)
            drift_rate = float(magnitude * self.rng.choice([-1, 1]))

        # true_slope for metadata / intersection bookkeeping
        if abs(drift_rate) < 1e-9:
            true_slope = 1e9  # vertical signal
        else:
            slope = -1.0 / drift_rate
            true_slope = slope / (self.params.dt / self.params.df)

        return drift_rate, true_slope

    def _calculate_eti_width(self, drift_rate: float) -> float:
        """ETI width: |DR|×dt + U(eti_offset_min, eti_offset_max). Narrowband."""
        drift_component = abs(drift_rate) * self.params.dt
        offset = self.rng.uniform(
            self.params.eti_width_offset_min, self.params.eti_width_offset_max
        )
        return drift_component + offset

    def _calculate_rfi_width(self, drift_rate: float) -> float:
        """RFI width: |DR|×dt + U(rfi_offset_min, rfi_offset_max). Broader."""
        drift_component = abs(drift_rate) * self.params.dt
        offset = self.rng.uniform(
            self.params.rfi_width_offset_min, self.params.rfi_width_offset_max
        )
        return drift_component + offset

    def _select_f_profile(self, width: float):
        """Select frequency profile (gaussian or sinc²) with configured weights."""
        profiles = list(self.params.freq_profiles)
        weights = list(self.params.freq_profile_weights)
        choice = str(self.rng.choice(profiles, p=weights))

        if choice == 'sinc2':
            return stg.sinc2_f_profile(width=width * u.Hz), 'sinc2'
        # default: gaussian
        return stg.gaussian_f_profile(width=width * u.Hz), 'gaussian'

    def _select_t_profile(self, intensity: float):
        """Select temporal profile (constant or scintillating) with configured weights."""
        profiles = list(self.params.time_profiles)
        weights = list(self.params.time_profile_weights)
        choice = str(self.rng.choice(profiles, p=weights))

        if choice == 'scintillating':
            period = self.rng.uniform(50, 300) * u.s
            amplitude = intensity * self.rng.uniform(0.2, 0.5)
            return (
                stg.sine_t_profile(period=period, amplitude=amplitude, level=intensity),
                'scintillating',
            )
        # default: constant
        return stg.constant_t_profile(level=intensity), 'constant'

    def _select_rfi_type(self) -> str:
        """Select RFI type with configured weights."""
        types = list(self.params.rfi_types)
        weights = list(self.params.rfi_type_weights)
        return str(self.rng.choice(types, p=weights))

    def _make_frame(self, data: np.ndarray) -> stg.Frame:
        """Create a setigen Frame from a COPY of existing data."""
        return stg.Frame.from_data(
            df=self.params.df * u.Hz,
            dt=self.params.dt * u.s,
            fch1=self.params.fch1 * u.MHz,
            ascending=False,
            data=data.copy(),
        )

    # ------------------------------------------------------------------
    # ETI signal injection (used for True samples)
    # ------------------------------------------------------------------

    def inject_signal(
        self,
        data: np.ndarray,
        snr: Optional[float] = None,
        start_channel: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Inject a narrowband drifting ETI signal.

        Uses log-uniform SNR and drift rate. Start channel is automatically
        constrained so the signal stays in-frame through at least ON₂
        (time bin 47), guaranteeing visibility in ≥ 2 of 3 ON windows.

        Args:
            data: Input spectrogram (tchans, fchans).
            snr: Signal-to-noise ratio. If None, sampled log-uniformly.
            start_channel: Starting frequency channel. If None, derived from
                           drift rate to satisfy the in-frame constraint.

        Returns:
            (injected data, signal_info dict)
        """
        tchans, fchans = data.shape

        if snr is None:
            snr = self._sample_snr()

        drift_rate, true_slope = self._sample_drift_rate()

        if start_channel is None:
            # Constrain start_channel so signal stays in-frame through ON₂
            drift_channels = (
                abs(drift_rate) / self.params.df * self._MIN_VISIBLE_BIN * self.params.dt
            )
            if drift_rate > 0:
                max_start = max(1, int(fchans - 1 - drift_channels))
                start_channel = int(self.rng.integers(1, max_start + 1))
            elif drift_rate < 0:
                min_start = min(fchans - 2, int(drift_channels + 1))
                start_channel = int(self.rng.integers(min_start, fchans - 1))
            else:
                start_channel = int(self.rng.integers(1, fchans - 1))

        width = self._calculate_eti_width(drift_rate)
        b = tchans - true_slope * start_channel

        frame = self._make_frame(data)
        intensity = frame.get_intensity(snr=snr)

        f_profile, f_name = self._select_f_profile(width)
        t_profile, t_name = self._select_t_profile(intensity)

        frame.add_signal(
            stg.constant_path(
                f_start=frame.get_frequency(index=start_channel),
                drift_rate=drift_rate * u.Hz / u.s,
            ),
            t_profile,
            f_profile,
            stg.constant_bp_profile(level=1),
        )

        signal_info = {
            'snr': snr,
            'drift_rate': drift_rate,
            'start_channel': start_channel,
            'width': width,
            'slope': true_slope,
            'intercept': b,
            'f_profile': f_name,
            't_profile': t_name,
        }
        return frame.data, signal_info

    def inject_cadence_signal(
        self,
        stacked_data: np.ndarray,
        snr: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Inject an ETI signal across a full stacked cadence (thin wrapper)."""
        return self.inject_signal(stacked_data, snr)

    # ------------------------------------------------------------------
    # RFI signal injection (used for False samples)
    # ------------------------------------------------------------------

    def inject_rfi_signal(
        self,
        data: np.ndarray,
        snr: Optional[float] = None,
        rfi_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Inject a realistic RFI signal.

        Args:
            data: Input spectrogram (tchans, fchans).
            snr: Signal-to-noise ratio. If None, sampled log-uniformly.
            rfi_type: One of RFI_TYPES. If None, picked by weighted random.

        Returns:
            (injected data, info dict)
        """
        tchans, fchans = data.shape

        if snr is None:
            snr = self._sample_snr()
        if rfi_type is None:
            rfi_type = self._select_rfi_type()

        start_channel = self.rng.integers(1, fchans - 1)
        frame = self._make_frame(data)
        f_start = frame.get_frequency(index=start_channel)
        intensity = frame.get_intensity(snr=snr)

        if rfi_type == 'linear':
            # Same drift model as ETI, but injected across ALL obs (not masked)
            drift_rate, _ = self._sample_drift_rate()
            width = self._calculate_rfi_width(drift_rate)
            path = stg.constant_path(
                f_start=f_start, drift_rate=drift_rate * u.Hz / u.s
            )
            t_prof = stg.constant_t_profile(level=intensity)
            f_prof = stg.gaussian_f_profile(width=width * u.Hz)

        elif rfi_type == 'stationary':
            spread = self.rng.uniform(50, 500) * u.Hz
            drift_rate = self.rng.uniform(-0.1, 0.1)
            width = self.rng.uniform(10, 80) * u.Hz
            path = stg.simple_rfi_path(
                f_start=f_start,
                drift_rate=drift_rate * u.Hz / u.s,
                spread=spread,
                spread_type='normal',
                rfi_type='stationary',
            )
            t_prof = stg.constant_t_profile(level=intensity)
            f_prof = stg.box_f_profile(width=width)

        elif rfi_type == 'random_walk':
            spread = self.rng.uniform(30, 300) * u.Hz
            drift_rate = self.rng.uniform(-0.5, 0.5)
            width = self._calculate_rfi_width(drift_rate)
            path = stg.simple_rfi_path(
                f_start=f_start,
                drift_rate=drift_rate * u.Hz / u.s,
                spread=spread,
                spread_type='normal',
                rfi_type='random_walk',
            )
            t_prof = stg.constant_t_profile(level=intensity)
            f_prof = stg.gaussian_f_profile(width=width * u.Hz)

        elif rfi_type == 'scintillating':
            drift_rate, _ = self._sample_drift_rate()
            width = self._calculate_rfi_width(drift_rate)
            period = self.rng.uniform(50, 300) * u.s
            amplitude = intensity * self.rng.uniform(0.3, 0.8)
            path = stg.constant_path(
                f_start=f_start, drift_rate=drift_rate * u.Hz / u.s
            )
            t_prof = stg.sine_t_profile(
                period=period, amplitude=amplitude, level=intensity
            )
            f_prof = stg.gaussian_f_profile(width=width * u.Hz)

        else:
            raise ValueError(
                f"Unknown RFI type: {rfi_type!r}. Choose from {RFI_TYPES}"
            )

        frame.add_signal(path, t_prof, f_prof, stg.constant_bp_profile(level=1))

        signal_info = {
            'rfi_type': rfi_type,
            'snr': snr,
            'start_channel': int(start_channel),
        }
        return frame.data, signal_info
