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

def normalize_frame(frame_data):
    """
    Applica log e normalizzazione Z-score ai dati.
    """
    
    # Log per comprimere il range dinamico
    data_log = np.log(frame_data + 1e-9) # epsilon per evitare log(0)
    
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
    resized_data = resize_data(raw_data)

    processed_frames = []
    
    for i in range(resized_data.shape[0]):
        frame_norm = normalize_frame(resized_data[i])
        processed_frames.append(frame_norm)
        
    final_data = np.array(processed_frames)
    
    # Add Channel Dimension (necessario per Conv2D)
    return final_data[..., np.newaxis]