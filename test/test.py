# %% [markdown]
# === Setup & Costanti (Fase 1 - Generazione dati sintetici) ===

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import setigen as stg

# Costanti paper / input rete
WIDTH_BIN = 4096          # frequenza grezza
FACTOR = 8                # downsample in frequenza
FCHANS = WIDTH_BIN
TCHANS = 16               # tempo grezzo
FCHANS_DS = FCHANS // FACTOR  # 512

# Parametri fisici predefiniti (coerenti con default BL in setigen)
DF = 2.79396772 * u.Hz            # risoluzione in frequenza
DT = 18.25361101 * u.s            # integrazione per time-bin
FCH1 = 1420.0 * u.MHz             # solo per riferimento; ascending=False => fch1 = freq massima
ASCENDING = False

CADENCE_ORDER = "ABABAB"          # pattern ON-OFF
N_FRAMES = len(CADENCE_ORDER)     # 6

os.makedirs("results/fase1", exist_ok=True)

# %% [markdown]
# === Utility ===

def _create_empty_frame(seed=None):
    """Costruisce un Frame vuoto (16 x 4096) con parametri DF/DT specificati."""
    return stg.Frame(
        fchans=FCHANS,
        tchans=TCHANS,
        df=DF,
        dt=DT,
        fch1=FCH1,
        ascending=ASCENDING,
        seed=seed
    )

def _log_norm_and_downsample_16x4096_to_16x512(arr_16x4096, factor=FACTOR):
    """
    Log-normalizzazione + downsampling per media di blocco (contiguo) lungo la frequenza.
    Input:  (16, 4096)
    Output: (16, 512, 1)
    """
    x = np.asarray(arr_16x4096, dtype=np.float32)
    # Garantisce positività e stabilità numerica: il rumore radiometrico è non negativo,
    # ma eventuali arrotondamenti/operazioni possono introdurre ~0 o <0
    min_val = x.min()
    if min_val <= 0:
        x = x - min_val + 1e-6

    # Log-normalizzazione (compressione dinamica)
    x = np.log1p(x)

    # Downsampling per media di blocco (conservativo, evita aliasing rispetto a resample FFT)
    # Assicura divisibilità esatta
    fch = x.shape[1]
    usable = (fch // factor) * factor
    if usable != fch:
        x = x[:, :usable]
    x = x.reshape(TCHANS, -1, factor).mean(axis=2)  # (16, 512)

    # Aggiunge canale esplicito
    return x[..., np.newaxis]  # (16, 512, 1)

def _cadence_preprocess_to_batch(cadence):
    """
    Itera i frame nella cadence (nessun uso di c.data!), applica preprocessing e
    impila in (6, 16, 512, 1)
    """
    out = []
    for fr in cadence:                       # OrderedCadence è indicizzabile/iterabile
        arr = fr.get_data()                  # (16, 4096) - API ufficiale
        out.append(_log_norm_and_downsample_16x4096_to_16x512(arr))
    X = np.stack(out, axis=0)                # (6, 16, 512, 1)
    return X

def _plot_cadence(cadence, savepath):
    """Plot rapido della cadence grezza (comodo per QA)."""
    fig = plt.figure(figsize=(10, 10))
    cadence.plot()
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight", dpi=160)
    plt.close(fig)

def _plot_preprocessed_stack(X, order, savepath):
    """
    Visualizza i 6 frame preprocessati (16x512) in una griglia verticale, con label A/B.
    X: (6, 16, 512, 1)
    """
    fig, axes = plt.subplots(nrows=X.shape[0], ncols=1, figsize=(9, 10), sharex=True)
    if X.shape[0] == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(X[i, :, :, 0], aspect="auto", origin="lower")
        ax.set_ylabel(f"{order[i]}  t")
        if i == X.shape[0]-1:
            ax.set_xlabel("freq (downsampled)")
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight", dpi=160)
    plt.close(fig)

# %% [markdown]
# === Generatori dataset (Classe 0: FALSO, Classe 1: VERO) ===

def generate_falso_sample(
    *,
    mode="noise",                # "noise" oppure "rfi"
    x_mean=10.0,                 # media rumore radiometrico per add_noise
    rfi_snr=30.0,                # SNR del tono RFI costante (se mode == "rfi")
    rfi_f_start_index=1024,      # indice frequenza di partenza RFI
    t_slew_s=None,               # gap tra frame; se None, usa durata del frame
    seed=None
):
    """
    Genera una cadenza FALSO:
      - mode="noise": solo rumore chi^2 in tutti i frame.
      - mode="rfi"  : rumore + tono stazionario (drift_rate=0) in TUTTI i frame.
    Ritorna:
      X_pre: (6, 16, 512, 1)  preprocessato
      cadence: stg.OrderedCadence per plotting/diagnostica
    """
    rng = np.random.default_rng(seed)
    frames = [_create_empty_frame(seed=int(rng.integers(1e9))) for _ in range(N_FRAMES)]

    # t_slew: se non specificato, usa la lunghezza osservazione di un frame
    t_slew = (TCHANS * DT).to_value(u.s) if t_slew_s is None else float(t_slew_s)

    cadence = stg.OrderedCadence(frames, order=CADENCE_ORDER, t_slew=t_slew, t_overwrite=True)

    # Rumore chi^2 in ogni frame
    cadence.apply(lambda fr: fr.add_noise(x_mean=x_mean))  # add_noise(..., noise_type='chi2') di default

    if mode.lower() == "rfi":
        # Tono RFI costante (drift_rate=0) presente in tutti i frame
        f_start = frames[0].get_frequency(rfi_f_start_index)
        level = frames[0].get_intensity(snr=rfi_snr)
        cadence.add_signal(
            stg.constant_path(f_start=f_start, drift_rate=0.0 * u.Hz/u.s),
            stg.constant_t_profile(level=level),
            stg.sinc2_f_profile(width=40 * u.Hz),
            stg.constant_bp_profile(level=1.0),
            doppler_smearing=False
        )

    # Preprocessing -> batch (6, 16, 512, 1)
    X_pre = _cadence_preprocess_to_batch(cadence)
    return X_pre, cadence


def generate_vero_sample(
    *,
    snr=30.0,                      # SNR del segnale ETI
    drift_hz_s=0.2,                # drift rate != 0 (Hz/s)
    f_start_index=2048,            # centro del segnale nel frame 0
    width_hz=40.0,                 # larghezza spettrale del tono
    x_mean=10.0,                   # media rumore radiometrico
    t_slew_s=None,                 # gap tra frame; se None, usa durata del frame
    seed=None
):
    """
    Genera una cadenza VERO:
      - Pattern ABABAB (ON-OFF) usando OrderedCadence.
      - Segnale ETI con drift_rate != 0 iniettato SOLO nei frame "A".
    Ritorna:
      X_pre: (6, 16, 512, 1)  preprocessato
      cadence: stg.OrderedCadence per plotting/diagnostica
    """
    rng = np.random.default_rng(seed)
    frames = [_create_empty_frame(seed=int(rng.integers(1e9))) for _ in range(N_FRAMES)]

    # Imposta tempi di inizio coerenti (simula delta_t fra osservazioni)
    t_slew = (TCHANS * DT).to_value(u.s) if t_slew_s is None else float(t_slew_s)
    cadence = stg.OrderedCadence(frames, order=CADENCE_ORDER, t_slew=t_slew, t_overwrite=True)

    # Rumore chi^2 in tutti i frame
    cadence.apply(lambda fr: fr.add_noise(x_mean=x_mean))

    # Segnale drifting SOLO nei frame "A"
    f_start = frames[0].get_frequency(f_start_index)
    level = frames[0].get_intensity(snr=snr)

    cadence.by_label("A").add_signal(
        stg.constant_path(f_start=f_start, drift_rate=drift_hz_s * u.Hz/u.s),
        stg.constant_t_profile(level=level),
        stg.sinc2_f_profile(width=width_hz * u.Hz),
        stg.constant_bp_profile(level=1.0),
        doppler_smearing=True
    )

    # Preprocessing -> batch (6, 16, 512, 1)
    X_pre = _cadence_preprocess_to_batch(cadence)
    return X_pre, cadence

# %% [markdown]
# === Sanity Check: esempio per classe + salvataggi ===

if __name__ == "__main__":
    # Classe 0 - FALSO (solo rumore)
    X_false_noise, cad_false_noise = generate_falso_sample(mode="noise", seed=42)
    _plot_cadence(cad_false_noise, "results/fase1/cadence_FALSO_noise.png")
    _plot_preprocessed_stack(X_false_noise, CADENCE_ORDER, "results/fase1/preproc_FALSO_noise.png")

    # Classe 0 - FALSO (RFI costante su tutti i frame)
    X_false_rfi, cad_false_rfi = generate_falso_sample(mode="rfi", rfi_snr=30, seed=43)
    _plot_cadence(cad_false_rfi, "results/fase1/cadence_FALSO_rfi.png")
    _plot_preprocessed_stack(X_false_rfi, CADENCE_ORDER, "results/fase1/preproc_FALSO_rfi.png")

    # Classe 1 - VERO (ABABAB, drift != 0 solo nei frame 'A')
    X_true, cad_true = generate_vero_sample(snr=30, drift_hz_s=0.2, seed=44)
    _plot_cadence(cad_true, "results/fase1/cadence_VERO.png")
    _plot_preprocessed_stack(X_true, CADENCE_ORDER, "results/fase1/preproc_VERO.png")

    print("Salvataggi completi in results/fase1/")
