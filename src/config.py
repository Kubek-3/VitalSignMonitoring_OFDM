# ==============================
# GLOBAL RADAR & SCENE CONFIG
# ==============================

import numpy as np

# ------------------------------
# ML
# ------------------------------
WINDOW_SEC = 15.0
STEP_SEC = 2.0
MIN_DUR = 4.0  # minimum duration of detected anomaly region [s]
MODEL_PATH = "models/isoforest_resp.pkl"
# ------------------------------
# PHYSICAL CONSTANTS
# ------------------------------
C = 299792458.0          # speed of light [m/s]

# ------------------------------
# SAMPLING
# ------------------------------
FIRST_SAMPLES = 30_000      # number of initial samples to use from chest motion data
DATA_FS = 100.0             # chest motion sampling rate [Hz]
ups_factor = 1              # upsampling factor for chest motion
FS_SLOW = DATA_FS * ups_factor                    # slow-time sampling rate [Hz]

K = 1024              # number of OFDM subcarriers
Delta_f = 960e3        # subcarrier spacing
Fs_base = 1e9         # base sampling rate
M = 16                # 16-QAM
fc = 26.5e9           # carrier
b = K * Delta_f      # OFDM occupied BW (approx)
desired_Fs = 2.5e9     # oversampled rate

print("Bandwidth (Hz):", b)

L = int(np.ceil(desired_Fs / Fs_base))
Nfft_time = K * L
Fs_high = L * Fs_base

# ------------------------------
# RADAR PARAMETERS
# ------------------------------
cf = 26.5e9              # carrier frequency [Hz]
#b = 1.0e9                # bandwidth [Hz]
K = 1024                 # number of subcarriers
TX_power_dBm = 20.0             # transmit power [dBm]
ref_cof = 0.685          # reflection coefficient
chair_ref_cof = 0.2     # reflection coefficient for chair
snr_db = 20.0            # SNR [dB]
t_snr_db = 20
NF = 10.0                  # noise figure [dB]
#Delta_f = b / K                    # Subcarrier spacing (assuming contiguous K subcarriers over bandwidth b)
#start_f = cf - Delta_f * (K // 2)   # Start frequency of the first subcarrier
#freqs = start_f + Delta_f * np.arange(K)  # Frequencies of each subcarrier
M = 16                   # 16-QAM modulation order
#Fs_high = 2.5e9  # High-rate sampling frequency for OFDM [Hz]
print("High-rate sampling Fs_high (Hz):", Fs_high)
# Nfft_time = 16384      # Number of FFT points for time-domain oversampling
freqs = cf + Delta_f * (np.arange(K) - K//2)  # Subcarrier frequencies



# ------------------------------
# SCENE GEOMETRY (meters)
# ------------------------------
xh, yh = [2.5, 1.0]   # human position (x,y)
xh_2, yh_2 = [15, 1.0]   # second human position (x,y)
xh_3, yh_3 = [9, 0.5]   # third human position (x,y)
x_chair, y_chair = [7, 0.5]  # chair position
xtx, ytx = [3.0, 3.0]  # transmitter position
xrx, yrx = [3.0, 3.0]   # receiver position

# ------------------------------
# ML / SIGNAL PROCESSING
# ------------------------------
RESP_LOW = 0.1
RESP_HIGH = 0.5

HEART_LOW = 0.8
HEART_HIGH = 2.0

# ------------------------------
# PATHS
# ------------------------------
DATA_NORMAL = "data/normal/"
DATA_IRREG  = "data/irregular/"
DATA_HOLD   = "data/breath_hold/"

FILE_PATHS = [
    "data/normal/N002.mat",
    "data/normal/N041.mat",
    "data/normal/N034.mat",
    ]

MODELS_PATH = "models/"
