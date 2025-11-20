import numpy as np
import tensorflow as tf
import setigen as stg
from astropy import units as u
from .preprocessingv2 import preprocess_pipeline

class SetiHybridGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 background_plates=None,  # Se None -> Usa rumore sintetico. Se Array -> Usa dati SRT
                 batch_size=32, 
                 samples_per_epoch=1000,
                 fchans=4096, 
                 tchans=16):
        
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.fchans = fchans
        self.tchans = tchans
        self.plates = background_plates
        
        # Parametri simulazione fisica
        self.df = 2.7939677238464355 * u.Hz
        self.dt = 18.25361108 * u.s

    def __len__(self):
        return int(self.samples_per_epoch // self.batch_size)

    def __getitem__(self, index):
        # Generazione batch "True" (Segnali ETI solo su ON)
        true_batch = self._generate_batch(label='true')
        
        # Generazione batch "False" (Solo rumore/RFI)
        false_batch = self._generate_batch(label='false')
        
        # Creazione del Mixed Batch per l'Encoder (metà True, metà False)
        half = self.batch_size // 2
        mixed_batch = np.concatenate([true_batch[:half], false_batch[half:]], axis=0)
        
        # Output per il modello (Input Multipli, Output Ricostruzione)
        return [mixed_batch, true_batch, false_batch], mixed_batch

    def _get_background(self):
        """
        Restituisce un frame di sfondo (Background).
        - Fase 1: Genera rumore chi-quadro con Setigen.
        - Fase 2: Estrae un crop casuale dai dati SRT caricati.
        """
        if self.plates is None:
            # MODALITÀ SINTETICA (FASE 1)
            frame = stg.Frame(fchans=self.fchans, tchans=self.tchans, df=self.df, dt=self.dt)
            noise = frame.add_noise(x_mean=10, noise_type='chi2')
            return frame
        else:
            # MODALITÀ REALE (FASE 2)
            # Estraiamo un indice casuale dal dataset di background
            idx = np.random.randint(0, self.plates.shape[0])
            real_data = self.plates[idx] 
            
            # Creiamo il frame Setigen a partire dai dati reali
            # Assumiamo una fch1 fittizia per coerenza
            frame = stg.Frame.from_data(self.df, self.dt, fch1=6000*u.MHz, data=real_data)
            return frame

    def _generate_batch(self, label):
        batch_data = []
        
        for _ in range(self.batch_size):
            cadence = []
            
            # Parametri randomici del segnale ETI
            drift_rate = np.random.uniform(-5, 5)
            start_index = np.random.randint(200, self.fchans - 200)
            width = np.random.uniform(20, 40)
            snr = np.random.uniform(15, 40)
            
            # Genera 6 osservazioni (ON-OFF-ON-OFF-ON-OFF)
            for i in range(6):
                frame = self._get_background()
                
                # Logica di iniezione
                is_on = (i % 2 == 0) # 0, 2, 4 sono ON
                inject_signal = False
                
                if label == 'true' and is_on:
                    inject_signal = True
                
                # (Opzionale) Aggiungi RFI casuale nel "False" batch o ovunque
                if label == 'false' and np.random.rand() > 0.5:
                     # Simula una RFI verticale (drift=0)
                     frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=np.random.randint(0, self.fchans)),
                                                        drift_rate=0*u.Hz/u.s),
                                      stg.constant_t_profile(level=frame.get_intensity(snr=snr*2)),
                                      stg.gaussian_f_profile(width=width*u.Hz))

                if inject_signal:
                    # Calcolo frequenza corrente basata sul tempo
                    # dt * tchans = durata di uno scan
                    t_elapsed = i * (self.tchans * self.dt.value)
                    f_drift = drift_rate * t_elapsed # Drift totale in Hz
                    
                    freq_val = frame.get_frequency(index=start_index)
                    if hasattr(freq_val, 'value'):
                        freq_val = freq_val.value
                    
                    current_f_start = (freq_val * u.Hz) + (f_drift * u.Hz)
                    
                    try:
                        frame.add_signal(stg.constant_path(f_start=current_f_start,
                                                           drift_rate=drift_rate*u.Hz/u.s),
                                         stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                                         stg.gaussian_f_profile(width=width*u.Hz),
                                         stg.constant_bp_profile(level=1))
                    except Exception as e:
                        # Ignora errori se il segnale esce dalla banda (comune con drift alti ai bordi)
                        pass 

                cadence.append(frame.data)
            
            # Preprocessing e aggiunta al batch
            cadence_np = np.array(cadence) # (6, 16, 4096)
            processed = preprocess_pipeline(cadence_np) # (6, 16, 512, 1)
            batch_data.append(processed)
            
        return np.array(batch_data)