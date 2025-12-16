import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft

# ------------------------------------------------------------
# PARAMETERS (RADAR-GRADE, CORRECT)
# ------------------------------------------------------------
N = 1024                 # subcarriers
BW = 1e9                 # 1 GHz bandwidth
Delta_f = BW / N         # ~976 kHz subcarrier spacing
Fs_base = 2.5e9          # base sampling rate
desired_Fs = 2.5e9       # no extra oversampling
Fs_high = desired_Fs 

M = 16                   # 16-QAM
fc = 26.5e9              # carrier
TX_power_dBm = 20
TX_power_W = 10 ** ((TX_power_dBm - 30) / 10)

print(f"OFDM BW = {BW/1e6:.1f} MHz, Δf = {Delta_f/1e3:.1f} kHz")

# ------------------------------------------------------------
# QAM CONSTELLATION (UNIT AVG POWER)
# ------------------------------------------------------------
m_vals = np.arange(M)
I = 2 * ((m_vals % 4) - 1.5)
Q = 2 * ((m_vals // 4) - 1.5)
constellation = (I + 1j * Q) / np.sqrt(10)

# ------------------------------------------------------------
# GENERATE MANY OFDM SYMBOLS (KEY FIX)
# ------------------------------------------------------------
N_symbols = 500           # <<< THIS IS CRITICAL
Nfft_time = int(Fs_high / Delta_f)    # no oversampling in time
ofdm_all = []

for _ in range(N_symbols):
    data = np.random.randint(0, M, N)
    qamSymbols = constellation[data]

    X = np.zeros(Nfft_time, dtype=complex)
    startIdx = Nfft_time // 2 - N // 2
    X[startIdx:startIdx + N] = qamSymbols

    ofdm = ifft(fftshift(X))

    # normalize to 1 W
    ofdm /= np.sqrt(np.mean(np.abs(ofdm) ** 2))
    ofdm *= np.sqrt(TX_power_W)

    ofdm_all.append(ofdm)

ofdm_all = np.concatenate(ofdm_all)

# ------------------------------------------------------------
# TIME AXIS
# ------------------------------------------------------------
t = np.arange(len(ofdm_all)) / Fs_high

# ------------------------------------------------------------
# PASSBAND SIGNAL
# ------------------------------------------------------------
ofdm_passband = ofdm_all * np.exp(1j * 2 * np.pi * fc * t)


# ------------------------------------------------------------
# PSD COMPUTATION (CORRECT)
# ------------------------------------------------------------
Nfft_spec = 262144

spec = fftshift(fft(ofdm_passband, Nfft_spec))
Pxx = (np.abs(spec) ** 2) / (Nfft_spec * Fs_high)
Pxx_dBHz = 10 * np.log10(Pxx + 1e-15)

f_axis = np.linspace(-Fs_high / 2, Fs_high / 2, Nfft_spec)

# ------------------------------------------------------------
# PLOT: OFFSET FROM CARRIER (BEST VIEW)
# ------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(f_axis / 1e6, Pxx_dBHz)
plt.xlim([-BW / 1e6, BW / 1e6])
plt.xlabel("Frequency offset from fc (MHz)")
plt.ylabel("PSD (dB/Hz)")
plt.title("mmWave OFDM PSD (26.5 GHz center, 1 GHz BW)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# OPTIONAL: ABSOLUTE RF VIEW
# ------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot((f_axis + fc) / 1e9, Pxx_dBHz)
plt.xlim([(fc - BW) / 1e9, (fc + BW) / 1e9])
plt.xlabel("Frequency (GHz)")
plt.ylabel("PSD (dB/Hz)")
plt.title("Passband PSD around 26.5 GHz")
plt.grid(True)
plt.tight_layout()
plt.show()
