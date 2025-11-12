import numpy as np
from skimage.transform import downscale_local_mean
from . import config

def resize_data(data_batch): 
    """
    Downsampling usando la media locale (pooling).
    Input: (N_Frames, 16, 4096)
    Output: (N_Frames, 16, 512)
    """
    # (1, 1, factor) riduce solo l'ultimo asse
    return downscale_local_mean(data_batch, (1, 1, config.RESIZE_FACTOR))

def normalize_data(data):
    """
    Applica log e normalizzazione Z-score ai dati.
    """
    
    # Log per comprimere il range dinamico
    data_log = np.log(data + 1e-9) # epsilon per evitare log(0)
    
    # Z-Score Normalization (per frame)
    mean = np.mean(data_log)
    std = np.std(data_log)
    
    data_norm = (data_log - mean) / (std + 1e-9)
    return data_norm

def preprocess_cadence(cadence_obj): 
    """
    Prende un oggetto setigen.Cadence e restituisce l'array pronto per la rete.
    Output: (6, 16, 512, 1)
    """
    # Estrazione dati grezzi
    raw_data = np.array([frame.data for frame in cadence_obj])
    
    # Resize
    resized = resize_data(raw_data)
    
    # Normalize
    normalized = normalize_data(resized)
    
    # Add Channel Dimension (necessario per Conv2D)
    return normalized[..., np.newaxis]