import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import glob
import os

# -------------------------
# Settings
# -------------------------
fs = 100   # sampling frequency (Hz)
# input_folder = "Post Exercise"      # folder containing .mat files
# output_csv = "BR_HR_results_Post_Exercise.csv"  # summary file

# input_folder = "Irregular Breathing"      # folder containing .mat files
# output_csv = "BR_HR_results_Irregular_Breathing.csv"  # summary file

# input_folder = "Breath Hold"      # folder containing .mat files
# output_csv = "BR_HR_results_Breath_Hold.csv"  # summary file

input_folder = "data/irregular"      # folder containing .mat files
output_csv = "BR_HR_results_irregular_breathing.csv"  # summary file


results = []

# -------------------------
# Function to compute HR + BR
# -------------------------
def compute_br_hr(nasal, ecg, fs):
    # Detect HR peaks
    hr_peaks, _ = find_peaks(
        ecg,
        distance=0.3 * fs, 
        height=np.mean(ecg) + 0.5 * np.std(ecg),
        prominence=0.5
    )

    # Detect BR peaks using *relative prominence*
    br_peaks, _ = find_peaks(
        nasal,
        distance=1.0 * fs,
        prominence=0.1
    )

    # Compute rates
    duration_sec = len(ecg) / fs
    HR = len(hr_peaks) / duration_sec * 60
    BR = len(br_peaks) / duration_sec * 60

    return HR, BR, hr_peaks, br_peaks


# -------------------------
# Process all .mat files
# -------------------------
files = sorted(glob.glob(os.path.join(input_folder, "*.mat")))

for filepath in files:
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}")

    # Load the MAT file
    data = sio.loadmat(filepath)

    mp36 = data['mp36_s']
    ecg = mp36[2]        # channel 3
    nasal = mp36[3]      # channel 4

    # Compute metrics
    HR, BR, hr_peaks, br_peaks = compute_br_hr(nasal, ecg, fs)

    # Save values in results list
    results.append({
        "file": filename,
        "BreathRate_BPM": BR,
        "HeartRate_BPM": HR
    })

    original_time = np.arange(len(ecg)) / fs

    # -------------------------
    # Save the graph as PNG
    # -------------------------
    plt.figure(figsize=(14,6))

    # Nasal airflow
    plt.subplot(2,1,1)
    plt.plot(original_time, nasal, color='orange')
    plt.scatter(original_time[br_peaks], nasal[br_peaks], s=20)
    plt.xlabel("Czas (s)")
    plt.ylabel("Amplituda Sygnału Przepływu Nosowego")
    plt.title(f"{filename} — BR ≈ {BR:.2f} bpm")
    plt.grid()

    # ECG
    plt.subplot(2,1,2)
    plt.plot(original_time, ecg, color='red')
    plt.scatter(original_time[hr_peaks], ecg[hr_peaks], s=20)
    plt.xlabel("Czas (s)")
    plt.ylabel("Amplituda Sygnału EKG")
    plt.title(f"{filename} — HR ≈ {HR:.2f} bpm")
    plt.grid()

    # save as: filename.png
    out_png = os.path.join(input_folder, filename.replace(".mat", "_PL.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

print("Done!")


# -------------------------
# Save all results to CSV
# -------------------------
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\nSaved summary to: {output_csv}\n")
print(df)
