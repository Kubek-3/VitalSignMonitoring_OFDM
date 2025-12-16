import numpy as np
import joblib
from src.ml.anomaly_features import window_features
from src.signal_processing.phase_for_ml import extract_phase_from_radar_file
from src.config import Fs_slow

def detect_anomalies_from_radar_file(path):

    model = joblib.load("models/br_iforest.pkl")
    scaler = joblib.load("models/br_scaler.pkl")

    phase_detr, t_slow = extract_phase_from_radar_file(path)

    X, centers = window_features(
        sig=phase_detr,
        fs=Fs_slow,
        win_s=10,
        step_s=2,
        f_low=0.1,
        f_high=0.5
    )

    if len(X) == 0:
        return None

    Xs = scaler.transform(X)
    preds = model.predict(Xs)

    anomalies = preds == -1

    return {
        "file": path,
        "time_centers": centers,
        "anomaly_windows": anomalies,
        "num_anomalies": np.sum(anomalies)
    }
