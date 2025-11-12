from astropy import units as u

# Data dimensions
FCHANS = 4096
TCHANS = 16
N_FRAMES = 6
RESIZE_FACTOR = 8
FCHANS_FINAL = FCHANS // RESIZE_FACTOR

# Setigen parameters
DF = 2.7939677238464355 * u.Hz
DT = 18.25361108 * u.s
FCH1 = 6095.214842353016 * u.MHz
NOISE_MEAN = 58348559

# Parametri temporali
OBS_DURATION = (TCHANS * DT).to(u.s).value
SLEW_TIME_S = 15

# Parametri Training
LATENT_DIM = 64
BATCH_SIZE = 32
EPOCHS = 50
