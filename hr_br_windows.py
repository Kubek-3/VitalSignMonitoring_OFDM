# ============================================================
# Sliding-Window BR (smoothed peaks) & HR (FFT) Extraction
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from scipy.signal import butter, filtfilt, find_peaks, detrend, savgol_filter
from src.signal_processing.chest_motion import load_chest_motion

# ============================================================
# USER SETTINGS
# ============================================================
fs = 100

WINDOW_SEC = 15        # sliding window length
STEP_SEC   = 2         # sliding step

HR_BAND = (0.8, 2.0)   # HR frequency band (Hz)

input_folder = "data/post_exercise/"
output_csv = "BR_HR_results_window_15_post_exercise.csv"

# ============================================================
# FILTERS
# ============================================================

def bandpass_filter(sig, fs, f_low, f_high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [f_low/nyq, f_high/nyq], btype="band")
    return filtfilt(b, a, sig)

# ============================================================
# HR: FFT-BASED (unchanged, robust)
# ============================================================

def dominant_frequency(sig, fs, f_low, f_high):
    sig = sig * np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1/fs)

    band = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(band):
        return np.nan
    idx = np.argmax(spec[band])
    return freqs[band][idx]

def dominant_frequency_interp(sig, fs, f_low, f_high, nfft=8192):
    sig = sig * np.hanning(len(sig))

    spec = np.abs(np.fft.rfft(sig, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1/fs)

    band = (freqs >= f_low) & (freqs <= f_high)
    spec_b = spec[band]
    freqs_b = freqs[band]

    if len(spec_b) < 3:
        return np.nan

    k = np.argmax(spec_b)

    # Quadratic interpolation around the peak
    if 0 < k < len(spec_b) - 1:
        a, b, c = spec_b[k-1], spec_b[k], spec_b[k+1]
        p = 0.5 * (a - c) / (a - 2*b + c)
        df = freqs_b[1] - freqs_b[0]
        return freqs_b[k] + p * df

    return freqs_b[k]


# ============================================================
# BR: SMOOTHING + PEAK DETECTION (NEW)
# ============================================================

def estimate_br_from_smoothed(sig, fs):
    """
    Estimate breathing rate from chest motion using smoothing + peaks.
    Designed for signals where FFT BR fails.
    """

    # Smooth to suppress heart + noise
    smooth = savgol_filter(sig, window_length=301, polyorder=3)
    smooth = smooth - np.mean(smooth)

    # Detect breathing peaks
    peaks, _ = find_peaks(
        smooth,
        distance=int(2.5 * fs),                 # min ~2.5 s between breaths
        prominence=0.2 * np.std(smooth)
    )

    if len(peaks) < 2:
        return np.nan, (np.nan, np.nan, np.nan)

    intervals = np.diff(peaks) / fs

    BR = 60 / np.mean(intervals)
    return BR, (np.mean(intervals), np.min(intervals), np.max(intervals))

def peak_to_peak_stats(sig, fs, min_distance_sec):
    peaks, _ = find_peaks(sig, distance=int(min_distance_sec * fs))
    if len(peaks) < 2: 
        return np.nan, np.nan, np.nan
    intervals = np.diff(peaks) / fs
    return np.mean(intervals), np.min(intervals), np.max(intervals)

# ============================================================
# MAIN SLIDING-WINDOW FUNCTION
# ============================================================

def extract_br_hr_features(chest, fs):

    chest = detrend(chest)
    chest = chest / np.std(chest)

    chest_hr = bandpass_filter(chest, fs, *HR_BAND)

    win = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)

    rows = []
    t_axis, BR_t, HR_t, HR_t_interp = [], [], [], []

    for start in range(0, len(chest) - win, step):
        end = start + win
        t_mid = (start + end) / 2 / fs

        seg_chest = chest[start:end]
        seg_hr    = chest_hr[start:end]

        # -------- BR (SMOOTHED PEAKS) --------
        BR, br_p2p = estimate_br_from_smoothed(seg_chest, fs)

        # -------- HR (FFT) --------
        hr_hz = dominant_frequency(seg_hr, fs, *HR_BAND)
        HR = hr_hz * 60 if not np.isnan(hr_hz) else np.nan
        hr_p2p = peak_to_peak_stats(seg_hr, fs, min_distance_sec=0.5)
        hr_hz_interp = dominant_frequency_interp(seg_hr, fs, *HR_BAND)
        HR_interp = hr_hz_interp * 60 if not np.isnan(hr_hz_interp) else np.nan

        rows.append({
            "time_s": t_mid,
            "BR_bpm": BR,
            "BR_p2p_mean_s": br_p2p[0],
            "BR_p2p_min_s": br_p2p[1],
            "BR_p2p_max_s": br_p2p[2],
            "HR_bpm": HR,
            "HR_p2p_mean_s": hr_p2p[0],
            "HR_p2p_min_s": hr_p2p[1],
            "HR_p2p_max_s": hr_p2p[2],
        })

        t_axis.append(t_mid)
        BR_t.append(BR)
        HR_t.append(HR)
        HR_t_interp.append(HR_interp)

    return pd.DataFrame(rows), np.array(t_axis), np.array(BR_t), np.array(HR_t), np.array(HR_t_interp)

# ============================================================
# PROCESS FILES
# ============================================================

all_results = []

files = sorted(glob.glob(os.path.join(input_folder, "*.mat")))

for filepath in files:
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}")

    chest = load_chest_motion(filepath)

    df, t, BR_t, HR_t, HR_t_interp = extract_br_hr_features(chest, fs)
    df.insert(0, "file", filename)
    all_results.append(df)

    # ------------------ PLOTS ------------------
    plt.figure(figsize=(12,6))

    plt.subplot(3,1,1)
    plt.title(f"Estimated Breathing Rate & Heart Rate - {filename}")
    plt.plot(t, BR_t, marker="o")
    plt.ylabel("BR (bpm)")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, HR_t, marker="o")
    plt.ylabel("HR (bpm)")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, HR_t_interp, marker="o", color="orange")
    plt.ylabel("HR interp. (bpm)")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.tight_layout()
    out_png = os.path.join(input_folder, filename.replace(".mat", "_BR_HR.png"))
    plt.savefig(out_png, dpi=200)
    plt.close()

# ============================================================
# SAVE CSV
# ============================================================

final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(output_csv, index=False)

print(f"\nSaved results to: {output_csv}")
print(final_df.head())
