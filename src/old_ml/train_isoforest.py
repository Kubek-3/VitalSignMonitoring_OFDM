import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.ml.anomaly_features import window_features
from src.signal_processing.phase_for_ml import extract_phase_from_radar_file
from src.config import DATA_NORMAL, Fs_slow, cf   # update paths

import glob
import os

def train_isolation_forest():

    X_all = []

    print("Loading NORMAL radar files...")
    files = glob.glob(os.path.join(DATA_NORMAL, "*.mat"))

    for f in files:
        phase_detr, t_slow = extract_phase_from_radar_file(f)

        X, centers = window_features(
            sig=phase_detr,
            fs=Fs_slow,
            win_s=10,
            step_s=2,
            f_low=0.1,
            f_high=0.5
        )

        if len(X) > 0:
            X_all.append(X)

    X_all = np.vstack(X_all)
    print("Training feature matrix:", X_all.shape)

    # scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)

    model = IsolationForest(
        n_estimators=500,
        contamination=0.20,
        max_samples=0.7,
        random_state=0
    )

    model.fit(Xs)

    # save
    joblib.dump(model, "models/br_iforest.pkl")
    joblib.dump(scaler, "models/br_scaler.pkl")

    print("Isolation Forest trained and saved.")

if __name__ == "__main__":
    train_isolation_forest()
