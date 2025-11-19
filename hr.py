import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the MAT file
data = sio.loadmat('Multimodal chest surface motion\Free_T10'
    '.mat')

# Extract ECG data (mp36_s). 
# Rows correspond to channels — typically row 1 is a good ECG candidate.
mp36 = data['mp36_s']
ecg = mp36[2]        # change to mp36[row] if needed

# --- Set sampling frequency ---
# You MUST replace this with the correct sampling rate if known.
fs = 100  # Hz (placeholder)

# --- Peak detection ---
# Detect R-peaks using prominence or height
peaks, _ = find_peaks(ecg,
                      distance=0.3 * fs,          # min 300 ms between beats
                      height=np.mean(ecg) + 0.5 * np.std(ecg))

# --- Heart Rate Calculation ---
duration_sec = len(ecg) / fs
heart_rate = len(peaks) / duration_sec * 60

print(f"Estimated Heart Rate: {heart_rate:.1f} bpm")

# --- Plot ECG ---
# plt.figure(figsize=(14,5))
# plt.plot(ecg, label="ECG")
# plt.scatter(peaks, ecg[peaks], s=40)
# plt.title(f"ECG with Detected R-Peaks — HR ≈ {heart_rate:.1f} bpm")
# plt.xlabel("Sample")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()
