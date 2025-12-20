# The code is developed by SalimWireless.com (Python version)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft
from scipy.signal import welch

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
N = 1024                # subcarriers
Delta_f = 960e3        # subcarrier spacing
Fs_base = 1e9         # base sampling rate
M = 16                # 16-QAM
fc = 26.5e9             # carrier
BW = N * Delta_f      # OFDM occupied BW (approx)
desired_Fs = 2.5e9     # oversampled rate
TX_power = 20        # transmit power in dbm
TX_power_W = 10**((TX_power - 30)/10)  # in watts``
print("Bandwidth (GHz):", BW)

# ------------------------------------------------------------
# OVERSAMPLING
# ------------------------------------------------------------
L = int(np.ceil(desired_Fs / Fs_base))
Fs_high = L * Fs_base
print(f"Fs_base = {Fs_base/1e3:.1f} kHz, Fs_high = {Fs_high/1e3:.1f} kHz, L = {L}")

# ------------------------------------------------------------
# GENERATE 16-QAM SYMBOLS
# (unit-average-power mapping)
# ------------------------------------------------------------
# MATLAB qammod(...,'UnitAveragePower',true)
m_vals = np.arange(M)
# Gray-coded 16-QAM
I = 2*((m_vals % 4) - 1.5)
Q = 2*((m_vals // 4) - 1.5)
constellation = (I + 1j*Q) / np.sqrt(10)     # normalized to unit avg power

data = np.random.randint(0, M, N)
qamSymbols = constellation[data]

# ------------------------------------------------------------
# MAP TO IFFT BINS (CENTERED)
# ------------------------------------------------------------
Nfft_time = N * L
X = np.zeros(Nfft_time, dtype=complex)

startIdx = Nfft_time // 2 - N//2
X[startIdx:startIdx + N] = qamSymbols

# ------------------------------------------------------------
# TIME-DOMAIN OFDM SIGNAL
# ------------------------------------------------------------
ofdm_oversampled = ifft(fftshift(X))

# Scale to 1 watt average power
power_signal = np.mean(np.abs(ofdm_oversampled)**2)
ofdm_oversampled = ofdm_oversampled / np.sqrt(power_signal)

ofdm_oversampled = ofdm_oversampled * np.sqrt(TX_power_W)
# ------------------------------------------------------------
# TIME VECTOR
# ------------------------------------------------------------
t = np.arange(Nfft_time) / Fs_high

# ------------------------------------------------------------
# BASEBAND PSD
# ------------------------------------------------------------
Nfft_spec = 10384
spec_base = fftshift(fft(ofdm_oversampled, Nfft_spec))
Pxx_base = (np.abs(spec_base)**2) / (Nfft_spec * Fs_high)
Pxx_base_dBHz = 10*np.log10(Pxx_base + 1e-15)
f_axis_base = np.linspace(-Fs_high/2, Fs_high/2, Nfft_spec)

# ------------------------------------------------------------
# PASSBAND UP-CONVERSION
# ------------------------------------------------------------
ofdm_passband = np.real(ofdm_oversampled * np.exp(1j*2*np.pi*fc*t))

# Passband PSD

# num_symbols = 10384
# signal = []

# for _ in range(num_symbols):
#     data = np.random.randint(0, M, N)
#     qamSymbols = constellation[data]

#     X = np.zeros(Nfft_time, dtype=complex)
#     X[startIdx:startIdx + N] = qamSymbols

#     ofdm = ifft(fftshift(X))
#     signal.append(ofdm)

# ofdm_oversampled = np.concatenate(signal)

# f, Pxx = welch(
#     ofdm_oversampled,
#     fs=Fs_high,
#     window='boxcar',
#     nperseg = 10384,
#     noverlap = 5192,
#     return_onesided=False
# )


# Pxx_dBHz = 10*np.log10(Pxx + 1e-15)


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------

plt.figure(figsize=(8,4))
plt.plot(f_axis_pass/1e9, Pxx_pass_dBHz)
plt.xlabel("Frequency (GHz)")
plt.ylabel("PSD (dB/Hz)")
plt.title("Passband PSD")
plt.grid(True)

plt.figure(figsize=(8,4))
plt.plot(f_axis_pass/1e9 + fc/1e9, Pxx_pass_dBHz)
plt.xlabel("Frequency (GHz)")
plt.ylabel("PSD (dB/Hz)")
plt.title("Passband PSD centered at 26.5 GHz")
plt.grid(True)


# plt.figure(figsize=(8,4))
# plt.plot(Pxx_fc/1e9, Pxx_pass_dBHz)
# plt.xlim([(fc - 2*BW)/1e9, (fc + 2*BW)/1e9])
# plt.xlabel("Frequency (GHz)")
# plt.ylabel("PSD (dB/Hz)")
# plt.title("Passband PSD (zoom around fc)")
# plt.grid(True)

plt.show()
