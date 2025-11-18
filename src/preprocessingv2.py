import numpy as np
from skimage.transform import downscale_local_mean

class CadencePreprocessor:
    def __init__(self, target_shape=(16, 512), downsample_factor=8):
        """
        Gestisce la trasformazione dei dati grezzi in input per il VAE
        """
        self.target_shape = target_shape
        self.downsample_factor = downsample_factor

    def process_cadence(self, cadence_data):
        """
        Processa un'intera cadenza (6 osservazioni) congiuntamente.
        
        Input: Array numpy di shape (6, 16, input_freq) es. (6, 16, 4096)
        Output: Array numpy di shape (6, 16, target_freq) es. (6, 16, 512) normalizzato 0-1.
        """
        
        # Eseguiamo Downsampling Frequenziale (Binning) 
        data_binned = downscale_local_mean(cadence_data, (1, 1, self.downsample_factor))
        
        # Log Normalization (con epsilon di sicurezza per evitare log(0))
        data_log = np.log(data_binned + 1e-9)
        
        """
        Min-Max Scaling Congiunto (Joint Normalization)
        Calcoliamo min e max su TUTTI i 6 frame contemporaneamente
        Questo preserva il contrasto relativo tra segnale (ON) e rumore (OFF)
        """
        min_val = np.min(data_log)
        max_val = np.max(data_log)
        
        if max_val == min_val:
            return np.zeros_like(data_log, dtype=np.float32)
            
        data_scaled = (data_log - min_val) / (max_val - min_val)
        
        return data_scaled.astype(np.float32)