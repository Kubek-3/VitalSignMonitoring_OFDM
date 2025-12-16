import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from src.config import Fs_slow as Fs

# ------------------------------------------------------------
# USER: import YOUR radar phase extraction
# ------------------------------------------------------------
# This function MUST return:
#   phase_detr : 1D numpy array (rad)
#   t          : 1D numpy array (seconds)
#   Fs_slow    : slow-time sampling frequency (Hz)
#
# Example signature:
# phase_detr, t, Fs_slow = extract_phase_from_radar_file(path)

from src.signal_processing.phase_for_ml import extract_phase_from_radar_file


# ------------------------------------------------------------
# WINDOWING
# ------------------------------------------------------------
def sliding_windows(x, win_len, hop):
    windows = []
    indices = []
    for i in range(0, len(x) - win_len, hop):
        windows.append(x[i:i + win_len])
        indices.append((i, i + win_len))
    return np.array(windows), indices


# ------------------------------------------------------------
# FEATURE EXTRACTION (RADAR-PHYSICS-BASED)
# ------------------------------------------------------------
def extract_features(w, fs):
    # --- Time domain ---
    var = np.var(w)
    rms = np.sqrt(np.mean(w ** 2))
    zero_cross = np.mean(np.diff(np.sign(w)) != 0)

    # --- Frequency domain ---
    freqs = np.fft.rfftfreq(len(w), 1 / fs)
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))

    band = (freqs >= 0.1) & (freqs <= 0.5)  # respiration band
    band_energy = np.sum(spec[band])

    peak_freq = (
        freqs[band][np.argmax(spec[band])]
        if np.any(band_energy)
        else 0.0
    )

    return np.array([
        var,
        rms,
        zero_cross,
        band_energy,
        peak_freq
    ])


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def run_anomaly_detection(
    normal_files,
    test_file,
    win_sec=10,
    hop_sec=1,
    contamination=0.05
):
    print("Extracting normal breathing features...")

    X_train = []

    for f in normal_files:
        phase_detr, t_slow = extract_phase_from_radar_file(f)

        win_len = int(win_sec * Fs)
        hop = int(hop_sec * Fs)

        windows, _ = sliding_windows(phase_detr, win_len, hop)
        feats = np.array([extract_features(w, Fs) for w in windows])
        X_train.append(feats)

    X_train = np.vstack(X_train)

    # --------------------------------------------------------
    # Train Isolation Forest
    # --------------------------------------------------------
    clf = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=0
    )
    clf.fit(X_train)

    print("Model trained on NORMAL breathing only.")

    # --------------------------------------------------------
    # Run inference on test file
    # --------------------------------------------------------
    phase_detr, t_slow = extract_phase_from_radar_file(test_file)

    win_len = int(win_sec * Fs)
    hop = int(hop_sec * Fs)

    windows, indices = sliding_windows(phase_detr, win_len, hop)
    X_test = np.array([extract_features(w, Fs) for w in windows])

    labels = clf.predict(X_test)  # +1 normal, -1 anomaly

    # --------------------------------------------------------
    # Convert window labels → time mask
    # --------------------------------------------------------
    anomaly_mask = np.zeros(len(phase_detr), dtype=bool)

    for (start, end), lbl in zip(indices, labels):
        if lbl == -1:
            anomaly_mask[start:end] = True

    # --------------------------------------------------------
    # PLOT (FINAL RESULT)
    # --------------------------------------------------------
    plt.figure(figsize=(13, 4))
    plt.plot(t_slow, phase_detr, label="Radar phase (detrended)", color="blue")

    plt.fill_between(
        t_slow,
        phase_detr.min(),
        phase_detr.max(),
        where=anomaly_mask,
        color="red",
        alpha=0.3,
        label="Detected irregularity"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.title("Radar Phase at Human Range with Detected Breathing Irregularities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return anomaly_mask


# ------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------
if __name__ == "__main__":

    normal_dir = "data/normal"
    normal_files = sorted(
        glob.glob(os.path.join(normal_dir, "*.mat"))
    )

    print(f"Found {len(normal_files)} normal breathing files.")

    test_file = "data/breath_hold/BH001.mat"

    run_anomaly_detection(
        normal_files=normal_files,
        test_file=test_file,
        win_sec=10,
        hop_sec=1,
        contamination=0.05
    )
