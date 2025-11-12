import setigen as stg
import numpy as np
from astropy import units as u
from astropy.time import Time
from random import random as rand, uniform, randint

from . import config, preprocessing


def get_start_time(start_mjd=56789): 
    """
    Calcola i tempi di inizio per i 6 frame considerando lo slew time 
    (tempo che impiega il radiotelescopio per cambiare puntamento)
    """
    t_start_arr = [Time(start_mjd, format='mjd').unix]
    for i in range(1, config.N_FRAMES):
        # Start = Fine del precedente + Slew Time
        prev_start = t_start_arr[i - 1]
        next_start = prev_start + config.OBS_DURATION + config.SLEW_TIME_S
        t_start_arr.append(next_start)
    return t_start_arr

def create_false_sample():
  """
    Genera un campione FALSO (Classe 0) 100% sintetico.
    Logica: 50% solo rumore, 50% RFI (segnale drift=0 iniettato in TUTTI i 6 frame).
    Ritorna:
        - data_processed (np.array): L'array numpy finale (6, 16, 512, 1)
        - cadence (stg.Cadence): L'oggetto cadenza per il plotting
  """

  frame_list = [stg.Frame(fchans=config.FCHANS,
                                tchans=config.TCHANS,
                                df=config.DF,
                                dt=config.DT,
                                fch1=config.FCH1) for _ in range(config.N_FRAMES)]
  cadence = stg.Cadence(frame_list=frame_list)
  cadence.apply(lambda frame: frame.add_noise(x_mean=config.NOISE_MEAN,
                                                    x_std=config.NOISE_MEAN/4,
                                                    noise_type='chi2'))

  if rand() > 0.5:
      pass
  else:
      snr = uniform(10, 300)
      start_freq = cadence[0].get_frequency(index=randint(100, config.FCHANS - 100))

      cadence.add_signal(
          stg.constant_path(f_start=start_freq, drift_rate=0 * u.Hz/u.s),
          stg.constant_t_profile(level=cadence[0].get_intensity(snr=snr)),
          stg.gaussian_f_profile(width=uniform(10, 50) * u.Hz)
      )

  data_processed = preprocessing.preprocess_cadence(cadence)

  return data_processed, cadence

def create_true_sample():
  """
    Genera un campione VERO (Classe 1) 100% sintetico.
    Logica: Segnale ETI (drift != 0) iniettato in pattern "ABABAB".
    Ritorna:
        - data_processed (np.array): L'array numpy finale (6, 16, 512, 1)
        - cadence (stg.Cadence): L'oggetto cadenza per il plotting
  """
  
  t_start = get_start_time()
  
  frame_list = [stg.Frame(fchans=config.FCHANS,
                              tchans=config.TCHANS,
                              df=config.DF,
                              dt=config.DT,
                              fch1=config.FCH1,
                              t_start=t_start[i]) for i in range(config.N_FRAMES)]

  cadence = stg.OrderedCadence(frame_list=frame_list, order="ABABAB")
  cadence.apply(lambda frame: frame.add_noise(x_mean=config.NOISE_MEAN,
                                                  x_std=config.NOISE_MEAN/4,
                                                  noise_type='chi2'))

  snr = uniform(10, 300)
  drift_rate = (rand() * 2 + 1) * (-1)**randint(1,3) # Non-zero drift
  start_freq = cadence[0].get_frequency(index=randint(100, config.FCHANS - 100))

  cadence.by_label("A").add_signal(
      stg.constant_path(f_start=start_freq, drift_rate=drift_rate * u.Hz/u.s),
      stg.constant_t_profile(level=cadence[0].get_intensity(snr=snr)),
      stg.gaussian_f_profile(width=uniform(10, 50) * u.Hz),
      stg.constant_bp_profile(level=1)
  )

  data_processed = preprocessing.preprocess_cadence(cadence)

  return data_processed, cadence