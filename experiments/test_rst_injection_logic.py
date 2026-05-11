#!/usr/bin/env python3
"""
Verification test for the rst-injection-logic branch.
Run from the ML-SRT-SETI root directory with the rst conda env:

    conda run -n rst python experiments/test_rst_injection_logic.py
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import sys
import os
import types
import warnings
warnings.filterwarnings('ignore')

# Make src importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Block tensorflow (not installed in rst env)
_tf = types.ModuleType('tensorflow')
_tf.keras = types.ModuleType('tensorflow.keras')
sys.modules['tensorflow'] = _tf
sys.modules['tensorflow.keras'] = _tf.keras

# Mock numba with no-op decorators (preprocessing uses @jit and prange)
_numba = types.ModuleType('numba')
_numba.jit = lambda *a, **kw: (lambda f: f)   # @jit(...) → identity decorator
_numba.prange = range                          # prange → built-in range
sys.modules['numba'] = _numba

import numpy as np

# Direct imports (relative imports resolved by adding src/data to path temporarily)
sys.path.insert(0, os.path.join(ROOT, 'src', 'data'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'utils'))

from signal_generator import SignalGenerator, SignalParams, compute_max_drift_rate
from noise_generator import NoiseGenerator, create_noise_plate, NoiseParams
from cadence_generator import CadenceGenerator, CadenceParams
from preprocessing import preprocess, downscale, combine_cadences

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'

errors = []

# -----------------------------------------------------------------------
# TEST 1: SignalGenerator
# -----------------------------------------------------------------------
print('=' * 55)
print('TEST 1: SignalGenerator')
print('=' * 55)

max_dr = compute_max_drift_rate(4096, 2.7939677238464355, 18.25361108)
print(f'  max_drift_rate (fchans=4096): {max_dr:.4f} Hz/s')

sg = SignalGenerator(SignalParams(), seed=42)
rng = np.random.default_rng(0)
data = (rng.random((96, 4096)) * 58348559).astype(np.float32)

# ETI injection
injected, info = sg.inject_signal(data)
ok = injected.shape == (96, 4096)
print(f'  inject_signal shape (96,4096):  {PASS if ok else FAIL}')
print(f'    SNR={info["snr"]:.1f}, DR={info["drift_rate"]:.4f} Hz/s, '
      f'f={info["f_profile"]}, t={info["t_profile"]}')
if not ok:
    errors.append('inject_signal shape')

# All 4 RFI types
for rfi_type in ['linear', 'stationary', 'random_walk', 'scintillating']:
    rfi_data, rfi_info = sg.inject_rfi_signal(data, rfi_type=rfi_type)
    ok = rfi_data.shape == (96, 4096)
    print(f'  inject_rfi_signal({rfi_type}):  {PASS if ok else FAIL}')
    if not ok:
        errors.append(f'inject_rfi_signal({rfi_type})')

# Verify log-uniform SNR distribution
snrs = [sg._sample_snr() for _ in range(1000)]
log_snrs = np.log10(snrs)
ok_min = 9.0 <= min(snrs) <= 12.0
ok_max = 40.0 <= max(snrs) <= 52.0
median = float(np.median(snrs))
ok_median = 18.0 <= median <= 24.0  # median of log-uniform [10,50] ≈ sqrt(10*50) ≈ 22.4
print(f'  SNR log-uniform [10,50]:  {PASS if (ok_min and ok_max and ok_median) else FAIL}')
print(f'    min={min(snrs):.1f}  max={max(snrs):.1f}  median={median:.1f}  (expected ~22)')
if not (ok_min and ok_max and ok_median):
    errors.append('SNR distribution')

# Verify drift rate distribution
drs = [sg._sample_drift_rate()[0] for _ in range(1000)]
zero_count = sum(1 for d in drs if d == 0.0)
ok_zero = 20 <= zero_count <= 80  # ~5% zero drift, ±3σ
abs_drs = [abs(d) for d in drs if d != 0.0]
ok_range = all(0.009 <= d <= max_dr * 1.01 for d in abs_drs)
print(f'  Drift rate log-uniform:  {PASS if (ok_zero and ok_range) else FAIL}')
print(f'    zero_drift={zero_count}/1000 (~50 expected), |DR| in [0.01, {max_dr:.2f}]: {ok_range}')
if not (ok_zero and ok_range):
    errors.append('Drift rate distribution')

# -----------------------------------------------------------------------
# TEST 2: CadenceGenerator
# -----------------------------------------------------------------------
print()
print('=' * 55)
print('TEST 2: CadenceGenerator')
print('=' * 55)

plate = create_noise_plate(30, fchans=4096)
print(f'  Synthetic plate: {plate.shape}')
gen = CadenceGenerator(CadenceParams(fchans=4096), plate=plate, seed=42)

# ETI-only true sample
ts, meta = gen._create_eti_only_sample()
ok = ts.shape == (6, 16, 4096) and meta['true_mode'] == 'eti_only'
print(f'  True (ETI-only) shape + mode:  {PASS if ok else FAIL}')
if not ok:
    errors.append('True ETI-only')

# ETI+RFI true sample
ts2, meta2 = gen._create_eti_with_rfi_sample()
ok = ts2.shape == (6, 16, 4096) and meta2['true_mode'] == 'eti_with_rfi'
print(f'  True (ETI+RFI) shape + mode:   {PASS if ok else FAIL}  (n_rfi={meta2["n_rfi"]})')
if not ok:
    errors.append('True ETI+RFI')

# Hard False
hard = gen._create_hard_false_sample(gen._stack_cadence(gen._get_background()))
ok = hard.shape == (6, 16, 4096)
print(f'  False (Hard False) shape:       {PASS if ok else FAIL}')
if not ok:
    errors.append('False Hard False')

# Random false
fs = gen.create_false_sample()
ok = fs.shape == (6, 16, 4096)
print(f'  False (random) shape:           {PASS if ok else FAIL}')
if not ok:
    errors.append('False random')

# SingleShot
ss = gen.create_single_shot_sample()
ok = ss.shape == (6, 16, 4096)
print(f'  SingleShot shape:               {PASS if ok else FAIL}')
if not ok:
    errors.append('SingleShot')

# generate_batch interface (backwards compat with train_large_scale.py)
for sample_type in ['true_fast', 'false', 'single_shot']:
    b = gen.generate_batch(sample_type, 4)
    ok = b.shape == (4, 6, 16, 4096)
    print(f'  generate_batch("{sample_type}", 4):  {PASS if ok else FAIL}  -> {b.shape}')
    if not ok:
        errors.append(f'generate_batch({sample_type})')

# -----------------------------------------------------------------------
# TEST 3: Full pipeline
# -----------------------------------------------------------------------
print()
print('=' * 55)
print('TEST 3: Full Pipeline Integration')
print('=' * 55)

batch = gen.generate_batch('true_fast', 5)
batch_ds = downscale(batch, factor=8)
ok_ds = batch_ds.shape == (5, 6, 16, 512)
print(f'  downscale (8x): {batch.shape} -> {batch_ds.shape}  {PASS if ok_ds else FAIL}')
if not ok_ds:
    errors.append('downscale')

batch_pp = preprocess(batch_ds, add_channel=True)
ok_pp = batch_pp.shape == (5, 6, 16, 512, 1)
print(f'  preprocess:     {batch_ds.shape} -> {batch_pp.shape}  {PASS if ok_pp else FAIL}')
if not ok_pp:
    errors.append('preprocess')

flat = combine_cadences(batch_pp)
ok_fl = flat.shape == (30, 16, 512, 1)
print(f'  combine:        {batch_pp.shape} -> {flat.shape}  {PASS if ok_fl else FAIL}')
if not ok_fl:
    errors.append('combine_cadences')

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print()
print('=' * 55)
if errors:
    print(f'FAILED — {len(errors)} error(s):')
    for e in errors:
        print(f'  {FAIL} {e}')
    sys.exit(1)
else:
    print(f'{PASS} ALL TESTS PASSED')
print('=' * 55)
