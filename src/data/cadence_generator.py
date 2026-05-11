"""
Cadence generation for SETI training data — RST injection logic.

Ported from RST (rst_seti/data/cadence_generator.py) to ML-SRT-SETI for
fair architecture comparison. Key differences from the original ML-SRT-SETI
cadence_generator:

TRUE samples
  Old: two overlapping signals (signal₁ in all 6 obs, signal₂ in all 6 obs);
       ON gets both signals, OFF gets signal₁ only; non-crossing check required.
  New: single ETI signal in ON obs only (40% ETI-only) OR
       ETI in ON + 1-2 RFI across ALL obs (60% ETI+RFI).

FALSE samples
  Old: 50% pure background, 50% single RFI (linear drift) in all obs.
  New: 20% Hard False (perfect ETI in ALL 6 obs, labeled 0 to trap the model)
       60% RFI: 1-4 diverse RFI signals across all obs (weighted count)
       40% pure background

NoiseGenerator fallback is preserved (needed for --no-plate mode in
train_large_scale.py). The plate is still optional.
"""

import numpy as np
import setigen as stg
from astropy import units as u
from typing import Optional, Tuple
from dataclasses import dataclass, field
try:
    from .noise_generator import NoiseGenerator, NoiseParams
    from .signal_generator import SignalGenerator, SignalParams
except ImportError:
    from noise_generator import NoiseGenerator, NoiseParams  # type: ignore
    from signal_generator import SignalGenerator, SignalParams  # type: ignore


@dataclass
class CadenceParams:
    """Parameters for cadence generation (RST injection logic)."""
    tchans: int = 16              # Time channels per observation
    fchans: int = 4096            # Frequency channels (ML-SRT-SETI snippet width)
    num_observations: int = 6     # Cadence length (ON-OFF-ON-OFF-ON-OFF)

    # Signal parameters — passed through to SignalGenerator
    signal_params: SignalParams = field(default_factory=SignalParams)

    # True sample composition
    eti_only_fraction: float = 0.4      # 40% ETI-only, 60% ETI+RFI
    max_disturbance_rfi: int = 2        # Max extra RFI signals in ETI+RFI mode

    # False sample composition
    rfi_fraction: float = 0.6           # 60% RFI injected, 40% pure background
    hard_false_fraction: float = 0.2    # 20% Hard False (trap) samples
    rfi_count_weights: tuple = (0.4, 0.3, 0.2, 0.1)  # P(1), P(2), P(3), P(4)

    # Noise fallback (only used when plate=None)
    noise_mean: float = 58348559


class CadenceGenerator:
    """
    Generator for SETI training cadences — RST injection logic.

    Creates three types of samples:
    - True: ETI signal present in ON observations only.
        • ETI-only (40%): one ETI in ON, clean background in OFF.
        • ETI+RFI (60%): ETI in ON + 1-2 RFI across ALL obs.
    - False: No ETI-only pattern.
        • Hard False (20%): perfect ETI in ALL 6 obs (labeled 0).
        • RFI (60%): 1-4 diverse RFI signals across all obs.
        • Pure background (40%): no injection.
    - SingleShot: One ETI injection in ON, used for sensitivity testing.

    The plate (real observation backgrounds) is optional; if None, synthetic
    chi-squared noise is generated as fallback for --no-plate mode.
    """

    def __init__(
        self,
        params: Optional[CadenceParams] = None,
        plate: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            params: Cadence generation parameters.
            plate: Real observation backgrounds, shape (N, 6, tchans, fchans).
                   If None, synthetic chi-squared noise is used as fallback.
            seed: Random seed (None for random).
        """
        self.params = params or CadenceParams()
        self.plate = plate
        self.rng = np.random.default_rng(seed)

        # Noise fallback (only instantiated when plate is None)
        if self.plate is None:
            noise_params = NoiseParams(
                fchans=self.params.fchans,
                tchans=self.params.tchans,
            )
            self.noise_gen = NoiseGenerator(noise_params)
        else:
            self.noise_gen = None

        # Signal generator with centralized params
        self.signal_gen = SignalGenerator(self.params.signal_params, seed)

    # ------------------------------------------------------------------
    # Background helpers
    # ------------------------------------------------------------------

    def _get_background(self) -> np.ndarray:
        """Return a random background cadence of shape (6, tchans, fchans)."""
        if self.plate is not None:
            idx = self.rng.integers(0, self.plate.shape[0])
            return self.plate[idx].copy()
        else:
            return self.noise_gen.generate_cadence(
                self.params.num_observations,
                self.params.fchans,
                self.params.tchans,
            )

    def _stack_cadence(self, cadence: np.ndarray) -> np.ndarray:
        """Stack (6, tchans, fchans) → (6*tchans, fchans)."""
        stacked = np.zeros(
            (self.params.num_observations * self.params.tchans, self.params.fchans)
        )
        for i in range(self.params.num_observations):
            s, e = i * self.params.tchans, (i + 1) * self.params.tchans
            stacked[s:e, :] = cadence[i]
        return stacked

    def _unstack_cadence(self, stacked: np.ndarray) -> np.ndarray:
        """Unstack (6*tchans, fchans) → (6, tchans, fchans)."""
        cadence = np.zeros(
            (self.params.num_observations, self.params.tchans, self.params.fchans)
        )
        for i in range(self.params.num_observations):
            s, e = i * self.params.tchans, (i + 1) * self.params.tchans
            cadence[i] = stacked[s:e, :]
        return cadence

    def _sample_rfi_count(self) -> int:
        """Sample number of RFI signals from the configured weighted distribution."""
        weights = np.array(self.params.rfi_count_weights, dtype=float)
        weights /= weights.sum()
        counts = np.arange(1, len(weights) + 1)
        return int(self.rng.choice(counts, p=weights))

    # ------------------------------------------------------------------
    # True samples
    # ------------------------------------------------------------------

    def create_true_sample(
        self, snr: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Create a TRUE sample (ETI signal in ON observations only).

        Randomly selects between two modes:
        - ETI-only (40%): single ETI in ON, background in OFF.
        - ETI+RFI (60%): ETI in ON + 1-2 RFI across all 6 obs.

        Returns:
            (cadence array of shape (6, tchans, fchans), metadata dict)
        """
        if self.rng.random() < self.params.eti_only_fraction:
            return self._create_eti_only_sample(snr)
        else:
            return self._create_eti_with_rfi_sample(snr)

    def _create_eti_only_sample(
        self, snr: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        ETI signal visible in ON observations only; OFF keeps original background.
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)

        # Inject ETI across the full stacked cadence
        injected, eti_info = self.signal_gen.inject_cadence_signal(stacked, snr)

        t = self.params.tchans
        result = np.zeros((6, t, self.params.fchans))
        # ON observations (indices 0, 2, 4) get the injected signal
        result[0] = injected[0:t, :]
        result[1] = stacked[t:2*t, :]       # OFF — original background
        result[2] = injected[2*t:3*t, :]
        result[3] = stacked[3*t:4*t, :]     # OFF — original background
        result[4] = injected[4*t:5*t, :]
        result[5] = stacked[5*t:6*t, :]     # OFF — original background

        metadata = {
            'sample_type': 'true',
            'true_mode': 'eti_only',
            'eti_signal': eti_info,
        }
        return result, metadata

    def _create_eti_with_rfi_sample(
        self, snr: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        ETI signal in ON + 1-2 RFI signals across ALL observations.

        The RFI is injected first (visible in all obs), then the ETI is
        injected on top (remains only in ON after the ON/OFF masking step).
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)

        # Step 1: inject 1-max_disturbance_rfi RFI signals across all obs
        n_rfi = self.rng.integers(1, self.params.max_disturbance_rfi + 1)
        rfi_infos = []
        current = stacked.copy()
        for _ in range(n_rfi):
            current, rfi_info = self.signal_gen.inject_rfi_signal(current)
            rfi_infos.append(rfi_info)

        # Step 2: inject ETI across the full cadence
        injected, eti_info = self.signal_gen.inject_cadence_signal(current, snr)

        t = self.params.tchans
        result = np.zeros((6, t, self.params.fchans))
        # ON: ETI + RFI | OFF: RFI only (no ETI)
        result[0] = injected[0:t, :]
        result[1] = current[t:2*t, :]       # RFI only
        result[2] = injected[2*t:3*t, :]
        result[3] = current[3*t:4*t, :]     # RFI only
        result[4] = injected[4*t:5*t, :]
        result[5] = current[5*t:6*t, :]     # RFI only

        metadata = {
            'sample_type': 'true',
            'true_mode': 'eti_with_rfi',
            'eti_signal': eti_info,
            'n_rfi': n_rfi,
            'rfi_signals': rfi_infos,
        }
        return result, metadata

    def create_true_sample_fast(self, snr: Optional[float] = None) -> np.ndarray:
        """Fast version — returns only the cadence array (no metadata)."""
        result, _ = self.create_true_sample(snr)
        return result

    # ------------------------------------------------------------------
    # False samples
    # ------------------------------------------------------------------

    def create_false_sample(self, snr: Optional[float] = None) -> np.ndarray:
        """
        Create a FALSE sample — no ETI-only pattern.

        Three sub-types (in priority order):
        1. Hard False (20%): perfect ETI-like signal across ALL 6 obs (labeled 0).
           Forces the model to check OFF observations.
        2. RFI (60%): 1-4 diverse RFI signals across all obs.
        3. Pure background (40%): no injection.

        Returns:
            Array of shape (6, tchans, fchans).
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)

        # 1. Hard False trap
        if self.rng.random() < self.params.hard_false_fraction:
            return self._create_hard_false_sample(stacked, snr)

        # 2. Pure background (when no forced SNR)
        if snr is None and self.rng.random() > self.params.rfi_fraction:
            return self._unstack_cadence(stacked)

        # 3. Standard RFI injection (1-4 signals)
        n_rfi = self._sample_rfi_count()
        for _ in range(n_rfi):
            stacked, _ = self.signal_gen.inject_rfi_signal(stacked, snr)

        return self._unstack_cadence(stacked)

    def _create_hard_false_sample(
        self,
        stacked_background: np.ndarray,
        snr: Optional[float] = None,
    ) -> np.ndarray:
        """
        Hard False: inject a perfect ETI-like signal across ALL 6 obs.
        Labeled as False (0) to force the model to analyse OFF scans.
        """
        injected, _ = self.signal_gen.inject_cadence_signal(stacked_background, snr)
        return self._unstack_cadence(injected)

    # ------------------------------------------------------------------
    # Sensitivity testing
    # ------------------------------------------------------------------

    def create_single_shot_sample(self, snr: Optional[float] = None) -> np.ndarray:
        """
        Single-shot sample: one ETI injection in ON obs only.
        Used for SNR sensitivity testing.

        Returns:
            Array of shape (6, tchans, fchans).
        """
        background = self._get_background()
        stacked = self._stack_cadence(background)

        injected, _ = self.signal_gen.inject_cadence_signal(stacked, snr)

        t = self.params.tchans
        result = np.zeros((6, t, self.params.fchans))
        result[0] = injected[0:t, :]
        result[1] = stacked[t:2*t, :]       # Original background
        result[2] = injected[2*t:3*t, :]
        result[3] = stacked[3*t:4*t, :]
        result[4] = injected[4*t:5*t, :]
        result[5] = stacked[5*t:6*t, :]

        return result

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        sample_type: str,
        batch_size: int,
        snr_base: Optional[float] = None,  # kept for API compatibility, ignored
        snr_range: Optional[float] = None, # kept for API compatibility, ignored
        factor: float = 1.0,               # kept for API compatibility, ignored
    ) -> np.ndarray:
        """
        Generate a batch of samples.

        Args:
            sample_type: 'true', 'true_fast', 'false', 'single_shot'.
            batch_size: Number of samples.
            snr_base, snr_range, factor: Legacy parameters kept for
                backward-compatibility with train_large_scale.py — ignored.
                SNR is now sampled log-uniformly from SignalParams.

        Returns:
            Array of shape (batch_size, 6, tchans, fchans).
        """
        batch = np.zeros(
            (batch_size, self.params.num_observations,
             self.params.tchans, self.params.fchans)
        )
        for i in range(batch_size):
            if sample_type == 'true':
                batch[i], _ = self.create_true_sample()
            elif sample_type == 'true_fast':
                batch[i] = self.create_true_sample_fast()
            elif sample_type == 'false':
                batch[i] = self.create_false_sample()
            elif sample_type == 'single_shot':
                batch[i] = self.create_single_shot_sample()
            else:
                raise ValueError(f"Unknown sample type: {sample_type!r}")

        return batch
