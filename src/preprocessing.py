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
    Log normalization + min-max scaling per singolo frame.
    """
    
    # Log per comprimere il range dinamico
    data_log = np.log(frame_data + 1e-9) # epsilon per evitare log(0)
    
    min_val = np.min(data_log)
    max_val = np.max(data_log)
    
    # Gestisce il caso di frame piatto (divisione per zero)
    if (max_val - min_val) == 0:
        return np.zeros_like(data_log)
        
    data_norm = (data_log - min_val) / (max_val - min_val)
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