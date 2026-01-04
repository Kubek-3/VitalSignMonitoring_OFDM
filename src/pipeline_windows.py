import os
from turtle import pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from src.signal_processing.new_radar_channel_2_persons import simulate_ofdm_radar_end_to_end_2
from src.signal_processing.ofdm_radar_fixed import simulate_ofdm_radar_fixed
from src.visualisation.plot_first_graphs import plot_first_graphs

from .config import (
    xh, yh, xtx, ytx, xrx, yrx,
    xh_2, yh_2,
    xh_3, yh_3,
    DATA_FS, ups_factor, FS_SLOW,
    cf,
    FIRST_SAMPLES,
    FILE_PATHS,
    WINDOW_SEC,
    STEP_SEC,
    MIN_DUR,
    RESP_LOW, RESP_HIGH,
    HEART_LOW, HEART_HIGH
)

from .signal_processing.chest_motion import (
    dist,
    load_chest_motion,
    upsample_signal,
    chest_displacement,
)

from .signal_processing.filters import bp_filter


from .signal_processing.chest_motion import (
    load_chest_motion,
    upsample_signal,
    chest_displacement
)
from .old_ml.radar_channel import radar_channel
from .signal_processing.new_radar_channel import simulate_ofdm_radar_end_to_end
from .signal_processing.radar_model import compute_phase, free_space_path_loss, pw_recvd_dBm, pw_recvd_w
from .visualisation.range_profile import plot_range_profile
from src.visualisation.plot_breathing_spectrum import plot_breathing_spectrum
import pandas as pd
from scipy.signal import find_peaks, detrend, savgol_filter

input_folder = "data/post_exercise"
output_csv = "BR_HR_results_radar_window_15_post_exercise.csv"
output_folder = "results/post_exercise/sliding_window/"

def estimate_rate(sig, fs, min_dist):
    """
    Estimate breathing rate from chest motion using smoothing + peaks.
    Designed for signals where FFT BR fails.
    """

    # Detect breathing peaks
    peaks, _ = find_peaks(
        sig,
        distance=int(min_dist * fs),                 # min ~2.5 s between breaths
        prominence=0.2 * np.std(sig)
    )

    if len(peaks) < 2:
        return np.nan, (np.nan, np.nan, np.nan)

    intervals = np.diff(peaks) / fs

    rate = 60 / np.mean(intervals)
    return rate, (np.mean(intervals), np.min(intervals), np.max(intervals))

def peak_to_peak_stats(sig, fs, min_distance_sec):
    peaks, _ = find_peaks(sig, distance=int(min_distance_sec * fs))
    if len(peaks) < 2: 
        return np.nan, np.nan, np.nan
    intervals = np.diff(peaks) / fs
    return np.mean(intervals), np.min(intervals), np.max(intervals)

def dominant_frequency(sig, fs, f_low, f_high):
    sig = sig * np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1/fs)

    band = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(band):
        return np.nan
    idx = np.argmax(spec[band])
    return freqs[band][idx]

def extract_br_hr_features(phase_resp, phase_heart, fs):

    win = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)

    rows = []
    t_axis, BR_t, HR_t = [], [], []

    for start in range(0, len(phase_resp) - win, step):
        end = start + win
        t_mid = (start + end) / 2 / fs

        seg_chest = phase_resp[start:end]
        seg_hr    = phase_heart[start:end]

        # # -------- BR (SMOOTHED PEAKS) --------
        # br_hz = dominant_frequency(seg_chest, fs, RESP_LOW, RESP_HIGH)
        # BR = br_hz * 60 if not np.isnan(br_hz) else np.nan
        # br_p2p = peak_to_peak_stats(seg_chest, fs, min_distance_sec=1.0)

        # # -------- HR (FFT) --------
        # hr_hz = dominant_frequency(seg_hr, fs, HEART_LOW, HEART_HIGH)
        # HR = hr_hz * 60 if not np.isnan(hr_hz) else np.nan
        # hr_p2p = peak_to_peak_stats(seg_hr, fs, min_distance_sec=0.5)

        BR, br_p2p = estimate_rate(seg_chest, fs, min_dist=1.0)
        HR, hr_p2p = estimate_rate(seg_hr, fs, min_dist=0.5)

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

    return pd.DataFrame(rows), np.array(t_axis), np.array(BR_t), np.array(HR_t)

def main():
    bed_tot = dist(xh, yh, xtx, ytx) + dist(xh, yh, xrx, yrx)

    files = sorted(glob.glob(os.path.join(input_folder, "*.mat")))

    all_results = []

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")

        chest = load_chest_motion(filepath)

        d_tot = chest_displacement(chest, xh, yh, xtx, ytx, xrx, yrx)
        
        h_slow, avg_profile, r_bin, phase_resp, phase_heart, p_coeff, t_slow  = simulate_ofdm_radar_end_to_end(d_tot, FS_SLOW, filename, output_folder)   # radar channel slow-time signal

        df, t, BR_t, HR_t = extract_br_hr_features(phase_resp, phase_heart, FS_SLOW)
        df.insert(0, "file", filename)
        all_results.append(df)

        # ------------------ PLOTS ------------------
        plt.figure(figsize=(12,6))

        plt.subplot(2,1,1)
        plt.title(f"Estimated Breathing Rate & Heart Rate - {filename}")
        plt.plot(t, BR_t, marker="o")
        plt.ylabel("BR (bpm)")
        plt.xlabel("Time (s)")
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(t, HR_t, marker="o")
        plt.ylabel("HR (bpm)")
        plt.xlabel("Time (s)")
        plt.grid(True)

        plt.tight_layout()
        out_png = os.path.join(output_folder, filename.replace(".mat", "_BR_HR_from_radar.png"))
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=200)
        plt.close()

# ============================================================
# SAVE CSV
# ============================================================

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(output_csv, index=False)

    print(f"\nSaved results to: {output_csv}")
    print(final_df.head())



if __name__ == "__main__":
    main()