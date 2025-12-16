import numpy as np
import joblib
from src.ml.feature_extractor import extract_features_from_phase
from src.signal_processing.phase_for_ml import extract_phase_from_radar_file
from src.config import Fs_slow

MODEL_PATH = "models/isoforest_resp.pkl"


def detect_anomalies_from_radar_file(mat_path,
                                     window_sec=10.0,
                                     step_sec=2.0):
    """
    Run Isolation Forest on radar file.
    Returns:
      t_mid, anomaly_flags, anomaly_scores
    """

    # 1) Load trained model
    model = joblib.load(MODEL_PATH)

    # 2) Extract phase
    phase_detr, t_slow = extract_phase_from_radar_file(mat_path)

    # 3) Extract features
    X, t_mid = extract_features_from_phase(
        phase_detr,
        Fs_slow,
        window_sec=window_sec,
        step_sec=step_sec
    )

    # 4) Predict
    preds = model.predict(X)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)

    anomaly_flags = (preds == -1)

    return t_mid, anomaly_flags, scores, phase_detr, t_slow
