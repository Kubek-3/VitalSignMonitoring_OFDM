import numpy as np
import joblib
from scipy.signal import hilbert, find_peaks
from scipy.fft import rfft
from src.signal_processing.filters import bp_filter
from src.ml.features import extract_resp_features
from src.signal_processing.phase_for_ml import extract_phase_from_radar_file
from src.config import Fs_slow, cf, c

MODEL_PATH = "models/isoforest_resp.pkl"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def phase_to_displacement(phase):
    lam = c / cf
    return (lam / (2 * np.pi)) * phase


def spectral_entropy(mag):
    p = mag / (np.sum(mag) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))


def merge_regions(indices, win_s, hop_s, min_dur=5.0):
    regions = []
    in_region = False
    start = 0

    for i, flag in enumerate(indices):
        if flag and not in_region:
            in_region = True
            start = i
        elif not flag and in_region:
            end = i - 1
            t0 = start * hop_s
            t1 = end * hop_s + win_s
            if t1 - t0 >= min_dur:
                regions.append((t0, t1))
            in_region = False

    if in_region:
        end = len(indices) - 1
        t0 = start * hop_s
        t1 = end * hop_s + win_s
        if t1 - t0 >= min_dur:
            regions.append((t0, t1))

    return regions


# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------
def detect_anomalies_from_radar_file(
    mat_path,
    window_sec=12.0,
    hop_sec=2.0,
    flat_factor=0.25,
    ml_percentile=1.0,
    min_region_dur=5.0,
):
    """
    Detect breathing anomalies from OFDM radar file.

    Returns:
      phase_detr
      t_slow
      flat_regions
      irregular_regions
      ml_scores
    """

    # --------------------------------------------------------
    # 1) Load model
    # --------------------------------------------------------
    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # 2) Extract phase
    # --------------------------------------------------------
    phase_detr, t_slow = extract_phase_from_radar_file(mat_path)
    disp = phase_to_displacement(phase_detr)

    # Respiration-band phase (used for ML)
    phase_resp = bp_filter(phase_detr, Fs_slow, 0.1, 0.5)

    # --------------------------------------------------------
    # 3) Sliding windows
    # --------------------------------------------------------
    win = int(window_sec * Fs_slow)
    hop = int(hop_sec * Fs_slow)

    features = []
    p2p_vals = []
    entropy_vals = []

    for i in range(0, len(disp) - win, hop):
        seg_disp = disp[i:i + win]
        seg_phase = phase_resp[i:i + win]

        # ML features
        features.append(extract_resp_features(seg_phase, Fs_slow))

        # Flatness
        p2p_vals.append(np.ptp(seg_disp))

        # Entropy
        spec = np.abs(rfft(seg_disp * np.hanning(len(seg_disp))))
        entropy_vals.append(spectral_entropy(spec))

    X = np.array(features)
    p2p_vals = np.array(p2p_vals)
    entropy_vals = np.array(entropy_vals)

    # --------------------------------------------------------
    # 4) Flat (breath-hold) detection
    # --------------------------------------------------------
    flat_thresh = flat_factor * np.median(p2p_vals)
    is_flat = p2p_vals < flat_thresh

    flat_regions = merge_regions(
        is_flat, window_sec, hop_sec, min_dur=min_region_dur
    )

    # --------------------------------------------------------
    # 5) ML-based irregular breathing
    # --------------------------------------------------------
    ml_scores = model.decision_function(X)

    score_thresh = np.percentile(ml_scores, ml_percentile)
    is_irregular_ml = ml_scores < score_thresh

    irregular_regions = merge_regions(
        is_irregular_ml, window_sec, hop_sec, min_dur=min_region_dur
    )

    # --------------------------------------------------------
    # 6) Return everything needed for plotting
    # --------------------------------------------------------
    return phase_detr, t_slow, disp, flat_regions, irregular_regions, ml_scores, p2p_vals, entropy_vals
