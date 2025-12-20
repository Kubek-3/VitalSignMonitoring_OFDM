import numpy as np
import joblib
from src.ml.features import extract_resp_features
from src.ml.windowing import sliding_windows
from src.signal_processing.phase_for_ml import extract_phase_from_radar_file
from src.config import Fs_slow, window_sec, step_sec, min_dur, MODEL_PATH

def windows_to_regions(anomaly_flags):
    regions = []
    in_region = False
    start_idx = 0

    for i, flag in enumerate(anomaly_flags):
        if flag and not in_region:
            in_region = True
            start_idx = i
        elif not flag and in_region:
            end_idx = i - 1
            start_t = start_idx * step_sec
            end_t = end_idx * step_sec + window_sec
            if end_t - start_t >= min_dur:
                regions.append((start_t, end_t))
            in_region = False

    if in_region:
        end_idx = len(anomaly_flags) - 1
        start_t = start_idx * step_sec
        end_t = end_idx * step_sec + window_sec
        if end_t - start_t >= min_dur:
            regions.append((start_t, end_t))

    return regions



def detect_anomalies_from_radar_file(mat_path):
    """
    Run Isolation Forest on radar file.
    Returns:
      t_mid, anomaly_flags, anomaly_scores
    """

    # 1) Load trained model
    model = joblib.load(MODEL_PATH)

    # 2) Extract phase
    phase_detr, t_slow = extract_phase_from_radar_file(mat_path)
    print(f"Loaded phase from {mat_path}, samples: {len(phase_detr)}")

    windows, _ = sliding_windows(phase_detr, Fs_slow)

    X = []

    for w in windows:
        feats = extract_resp_features(w, Fs_slow)
        X.append(feats)

    X = np.array(X)

    # 4) Predict
    preds = model.predict(X)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)

    anomaly_flags = (preds == -1)

    irregular_regions = windows_to_regions(anomaly_flags)

    return irregular_regions, anomaly_flags, scores, phase_detr, t_slow

