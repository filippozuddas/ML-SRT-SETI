import numpy as np
import setigen as stg
from astropy import units as u
from tqdm import tqdm
import os
from .preprocessingv2 import preprocess_pipeline

base_dir = os.path.dirname("/home/filippo/TirocinioSETI/ML-SRT-SETI/")
data_path = os.path.join(base_dir, 'data')

class SetiHybridGenerator():
    def __init__(self, 
                 background_plates=None,  # Se None -> Usa rumore sintetico. Se Array -> Usa dati SRT
                 batch_size=32, 
                 total_samples=1000,      # Totale campioni da generare
                 fchans=4096, 
                 tchans=16):
        
        self.background_plates = background_plates
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.fchans = fchans
        self.tchans = tchans
        
        self.df = 2.7939677238464355 * u.Hz
        self.dt = 18.25361108 * u.s
        
    def _get_background(self):
        """
        Restituisce un frame di sfondo (Background).
        - Fase 1: Genera rumore chi-quadro con Setigen.
        - Fase 2: Estrae un crop casuale dai dati SRT caricati.
        """
        if self.background_plates is None:
            # MODALITÀ SINTETICA (FASE 1)
            frame = stg.Frame(fchans=self.fchans, tchans=self.tchans, df=self.df, dt=self.dt, fch1=6000*u.MHz)
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

    

    def _generate_batch_data(self, label):
        """
        Genera un batch di dati con segnali ETI (label='true') o RFI/rumore (label='false').
        
        - True: Segnale con drift in frame ON (0, 2, 4)
        - False: 50% probabilità di RFI (drift~0) in TUTTI i 6 frame, altrimenti solo rumore
        """
        batch_data = []
        
        for _ in range(self.batch_size):
            cadence = []
            
            # RFI injection (una volta per cadenza, non per frame)
            add_rfi = (label == 'false' and np.random.rand() > 0.5)
            
            # Parametri del segnale (True o RFI)
            if label == 'true':
                # Segnale ETI: drift significativo
                drift_rate = np.random.uniform(-5, 5)
                width = np.random.uniform(20, 40)
                snr = np.random.uniform(15, 40)
                start_index = np.random.randint(200, self.fchans - 200)
            elif add_rfi:
                # RFI: drift quasi nullo, larghezza stretta
                drift_rate = np.random.uniform(-0.5, 0.5)  # Quasi verticale
                width = np.random.uniform(5, 15)
                snr = np.random.uniform(40, 50)  # SNR più alto per visibilità
                start_index = np.random.randint(200, self.fchans - 200)
            else:
                # Solo rumore (nessun segnale)
                drift_rate = 0
                width = 0
                snr = 0
                start_index = 0
            
            # Genera 6 osservazioni (ON-OFF-ON-OFF-ON-OFF)
            for i in range(6):
                frame = self._get_background()
                
                # Logica di iniezione
                should_inject = False
                
                if label == 'true':
                    # Segnale ETI: solo nei frame ON (0, 2, 4)
                    if i % 2 == 0:
                        should_inject = True
                elif add_rfi:
                    # RFI: in TUTTI i 6 frame
                    should_inject = True
                
                if should_inject:
                    # Calcolo frequenza con drift temporale
                    t_elapsed = i * (self.tchans * self.dt.value)
                    f_drift = drift_rate * t_elapsed
                    
                    freq_val = frame.get_frequency(index=start_index)
                    if hasattr(freq_val, 'value'):
                        freq_val = freq_val.value
                    
                    current_f_start = (freq_val * u.Hz) + (f_drift * u.Hz)
                    
                    try:
                        frame.add_signal(
                            stg.constant_path(f_start=current_f_start,
                                            drift_rate=drift_rate * u.Hz / u.s),
                            stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                            stg.gaussian_f_profile(width=width * u.Hz),
                            stg.constant_bp_profile(level=1))
                    except Exception as e:
                        # Ignora se il segnale esce dalla banda
                        pass
                
                cadence.append(frame.data)
            
            # Preprocessing e aggiunta al batch
            cadence_np = np.array(cadence)  # (6, 16, 4096)
            processed = preprocess_pipeline(cadence_np)  # (6, 16, 512, 1)
            batch_data.append(processed)
        
        return np.array(batch_data)
    
    def generate_and_save(self, filename= f"{data_path}/synthetic_dataset.npz"):
        """
        Genera i dati ciclando per il numero di batch necessari e salva su disco.
        """
        num_batches = self.total_samples // self.batch_size
        print(f"Inizio generazione di {self.total_samples} campioni ({num_batches} batches)...")
        
        all_mixed = []
        all_true = []
        all_false = []
        
        for _ in tqdm(range(num_batches)):
            # 1. Genera batch True
            true_batch = self._generate_batch_data(label='true')
            
            # 2. Genera batch False (RFI o Rumore)
            false_batch = self._generate_batch_data(label='false')
            
            # 3. Crea Mixed Batch (Metà True, Metà False)
            # Questa è la logica esatta del tuo __getitem__ originale
            half = self.batch_size // 2
            mixed_batch = np.concatenate([true_batch[:half], false_batch[half:]], axis=0)
            
            all_true.append(true_batch)
            all_false.append(false_batch)
            all_mixed.append(mixed_batch)
            
        # Concatenazione finale in grandi array NumPy
        print("Concatenazione array in memoria...")
        final_true = np.vstack(all_true)
        final_false = np.vstack(all_false)
        final_mixed = np.vstack(all_mixed)
        
        print(f"Salvataggio su {filename}...")
        print(f"Shape finali -> Mixed: {final_mixed.shape}, True: {final_true.shape}, False: {final_false.shape}")
        
        np.savez_compressed(filename, 
                            mixed_cadence=final_mixed, 
                            true_cadence=final_true, 
                            false_cadence=final_false)
        print("Salvataggio completato.")
        

        
if __name__ == "__main__":
    gen = SetiHybridGenerator(background_plates=None, batch_size=32, total_samples=128)
    gen.generate_and_save()