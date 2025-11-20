import numpy as np
from skimage.transform import downscale_local_mean

def resize_spectrogram(data, time_factor=1, freq_factor=8):
    """
    Ridimensiona lo spettrogramma riducendo la risoluzione (downsampling).
    4096 canali -> 512 canali (fattore 8)
    """
    # Data shape attesa: (Time, Freq)
    # downscale_local_mean esegue una media locale (pooling)
    resized = downscale_local_mean(data, (time_factor, freq_factor))
    return resized

def normalize_spectrogram(data):
    """
    Normalizza lo spettrogramma tra 0 e 1 usando scala logaritmica.
    """
    # Aggiungiamo una costante epsilon per evitare log(0)
    data = np.log(data + 1e-9)
    
    # Min-Max scaling
    min_val = data.min()
    data = data - min_val
    
    max_val = data.max()
    if max_val != 0:
        data = data / max_val
        
    return data

def preprocess_pipeline(cadence_stack, resize_factor=8):
    """
    Applica la pipeline completa a una intera cadenza (6 osservazioni).
    Input: (6, 16, 4096)
    Output: (6, 16, 512, 1) -> Formato pronto per TensorFlow
    """
    processed_scans = []
    
    for scan in cadence_stack:
        # 1. Resize
        resized = resize_spectrogram(scan, freq_factor=resize_factor)
        # 2. Normalize
        normalized = normalize_spectrogram(resized)
        processed_scans.append(normalized)
        
    # Stack e aggiunta dimensione canale (richiesto da Conv2D: H, W, Channels)
    # Risultato: (6, 16, 512, 1)
    return np.expand_dims(np.array(processed_scans), axis=-1)