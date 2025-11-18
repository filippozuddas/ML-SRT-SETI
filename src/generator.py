import numpy as np
import setigen as stg
from astropy import units as u
import tensorflow as tf
from .preprocessingv2 import CadencePreprocessor

class SetigenGenerator:
    def __init__(self, batch_size=32, snr_range=(10, 50), drift_range=(0.5, 8.0)):
        """
        Generatore Fedele all'implementazione di Peter Ma.
        """
        self.batch_size = batch_size
        self.snr_min, self.snr_max = snr_range
        self.drift_min, self.drift_max = drift_range
        
        # Parametri fisici GBT/SRT
        self.df = 2.7939677238464355 
        self.dt = 18.25361108 
        self.fch1 = 6095.214842353016 
        
        # Dimensioni Raw
        self.raw_time = 16
        self.raw_freq = 4096
        self.total_time = self.raw_time * 6 # 96 bin totali per la cadenza impilata

        # Inizializza Preprocessor
        self.preprocessor = CadencePreprocessor(target_shape=(16, 512), downsample_factor=8)

    def _get_random_drift(self):
        """Calcola drift e width casuali."""
        rate = np.random.uniform(self.drift_min, self.drift_max)
        sign = np.random.choice([1, -1]) # Fix bug README: uso uniforme
        drift_rate = rate * sign
        
        # Width casuale come nel codice originale
        width = np.random.uniform(5, 30) + abs(drift_rate) * 18.0
        return drift_rate, width

    def _create_frame_from_data(self, data_array):
        """Helper per creare un frame setigen da un array numpy esistente."""
        frame = stg.Frame(fchans=data_array.shape[1]*u.pixel,
                          tchans=data_array.shape[0]*u.pixel,
                          df=self.df*u.Hz,
                          dt=self.dt*u.s,
                          fch1=self.fch1*u.MHz)
        # Sovrascriviamo i dati interni
        frame.data = data_array
        return frame

    def _inject_signal(self, input_data, snr):
        """
        Inietta un segnale in un blocco dati 96x4096.
        Replica la funzione 'new_cadence' di Peter Ma.
        """
        # Creiamo il frame setigen dai dati esistenti
        frame = self._create_frame_from_data(np.copy(input_data))
        
        # CRITICAL FIX: Calcoliamo e assegniamo la deviazione standard del rumore
        # Setigen ne ha bisogno per calcolare l'intensità basata sull'SNR.
        # Poiché input_data è dominato dal rumore, np.std() è una buona stima.
        frame.noise_std = np.std(input_data)
        
        drift_rate, width = self._get_random_drift()
        
        # Logica di start casuale (evitando i bordi estremi)
        start_index = np.random.randint(int(self.raw_freq*0.1), int(self.raw_freq*0.9))
        start_freq = frame.get_frequency(index=start_index)
        
        # Iniezione
        # Nota: frame.get_intensity usa frame.noise_std che abbiamo appena settato
        frame.add_signal(stg.constant_path(f_start=start_freq,
                                           drift_rate=drift_rate*u.Hz/u.s),
                         stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                         stg.gaussian_f_profile(width=width*u.Hz),
                         stg.constant_bp_profile(level=1))
        
        return frame.data

    def _generate_base_noise(self):
        """Genera il rumore di fondo (96x4096)."""
        # Creiamo un frame vuoto
        frame = stg.Frame(fchans=self.raw_freq*u.pixel,
                          tchans=self.total_time*u.pixel,
                          df=self.df*u.Hz,
                          dt=self.dt*u.s,
                          fch1=self.fch1*u.MHz)
        # Aggiungiamo rumore chi2 (come nel paper)
        frame.add_noise(x_mean=10, noise_type='chi2')
        return frame.data

    def _slice_cadence(self, stacked_data):
        """Taglia l'array (96, 4096) in (6, 16, 4096)."""
        panels = []
        for i in range(6):
            # Prende blocchi di 16 righe
            panel = stacked_data[i*16 : (i+1)*16, :]
            panels.append(panel)
        return np.array(panels)

    def generate_batch(self):
        """Genera un singolo campione (X, y)."""
        
        # 1. Genera Sfondo (Noise) continuo per 96 time steps
        base_noise = self._generate_base_noise()
        
        label = np.random.choice([0, 1]) # 0=False, 1=True
        
        if label == 0: # Categoria FALSE (Noise o RFI)
            if np.random.rand() > 0.5:
                # Pure Noise
                final_stack = base_noise
            else:
                # Noise + RFI (RFI è un segnale presente ovunque)
                # Iniettiamo il segnale sullo sfondo base
                rfi_snr = np.random.uniform(self.snr_min, self.snr_max)
                final_stack = self._inject_signal(base_noise, rfi_snr)
            
            cadence_raw = self._slice_cadence(final_stack)
            
        else: # Categoria TRUE (Segnale SETI)
            # Qui dobbiamo replicare la logica "True" e "True + RFI"
            
            target_snr = np.random.uniform(self.snr_min, self.snr_max)
            
            if np.random.rand() > 0.5:
                # Case A: True Single Shot (Solo ETI, niente RFI)
                # Iniettiamo ETI sul rumore base
                eti_stack = self._inject_signal(base_noise, target_snr)
                
                # Tagliamo entrambi
                panels_eti = self._slice_cadence(eti_stack)
                panels_noise = self._slice_cadence(base_noise)
                
                # Costruiamo ON-OFF
                cadence_frames = []
                for i in range(6):
                    if i % 2 == 0: # ON (0, 2, 4)
                        cadence_frames.append(panels_eti[i])
                    else: # OFF (1, 3, 5)
                        cadence_frames.append(panels_noise[i])
                cadence_raw = np.array(cadence_frames)
                
            else:
                # Case B: True + RFI (Difficile)
                # Prima creiamo lo sfondo con RFI
                rfi_snr = np.random.uniform(self.snr_min, self.snr_max)
                rfi_stack = self._inject_signal(base_noise, rfi_snr)
                
                # Poi iniettiamo ETI SOPRA lo sfondo RFI
                # (Usiamo una copia per non sporcare i pannelli OFF)
                eti_on_rfi_stack = self._inject_signal(rfi_stack, target_snr)
                
                panels_eti_rfi = self._slice_cadence(eti_on_rfi_stack)
                panels_rfi = self._slice_cadence(rfi_stack)
                
                cadence_frames = []
                for i in range(6):
                    if i % 2 == 0: # ON -> ETI + RFI
                        cadence_frames.append(panels_eti_rfi[i])
                    else: # OFF -> Solo RFI
                        cadence_frames.append(panels_rfi[i])
                cadence_raw = np.array(cadence_frames)

        # 4. Preprocessing finale (Downsampling + Norm)
        # Input: (6, 16, 4096) -> Output: (6, 16, 512)
        cadence_processed = self.preprocessor.process_cadence(cadence_raw)
        
        return cadence_processed, label

    def get_dataset(self):
        """Ritorna un tf.data.Dataset."""
        def generator():
            while True:
                data, label = self.generate_batch()
                # Aggiungi channel dim: (6, 16, 512) -> (6, 16, 512, 1)
                data = np.expand_dims(data, axis=-1)
                yield data, label

        output_signature = (
            tf.TensorSpec(shape=(6, 16, 512, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset