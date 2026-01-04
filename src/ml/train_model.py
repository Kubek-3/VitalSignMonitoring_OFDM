import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import NearestNeighbors
import joblib
import glob
import os
import matplotlib.pyplot as plt

from src.ml.features import extract_resp_features
from src.ml.windowing import sliding_windows
from src.signal_processing.phase_for_ml import extract_phase_from_radar_file
from src.config import FS_SLOW, WINDOW_SEC, STEP_SEC

# ----------------------------
# CONFIG
# ----------------------------
normal_dir = "data/normal"
# normal_files = sorted(
#     glob.glob(os.path.join(normal_dir, "*.mat"))
# )
normal_files = ['data/normal\\N001.mat',
                'data/normal\\N002.mat',
                'data/normal\\N003.mat', 
                'data/normal\\N004.mat', 
                'data/normal\\N005.mat', 
                #'data/normal\\N006.mat',
                'data/normal\\N007.mat', 
                'data/normal\\N008.mat', 
                'data/normal\\N009.mat', 
                'data/normal\\N010.mat', 
                'data/normal\\N011.mat', 
                'data/normal\\N012.mat', 
                'data/normal\\N013.mat', 
                'data/normal\\N014.mat', 
                'data/normal\\N015.mat', 
                'data/normal\\N016.mat', 
                'data/normal\\N017.mat', 
                'data/normal\\N018.mat', 
                'data/normal\\N019.mat', 
                'data/normal\\N020.mat', 
                'data/normal\\N021.mat', 
                'data/normal\\N022.mat', 
                'data/normal\\N023.mat', 
                'data/normal\\N024.mat', 
                'data/normal\\N025.mat', 
                'data/normal\\N026.mat', 
                'data/normal\\N027.mat', 
                'data/normal\\N028.mat', 
                'data/normal\\N029.mat', 
                'data/normal\\N030.mat', 
                'data/normal\\N031.mat', 
                'data/normal\\N032.mat', 
                'data/normal\\N033.mat', 
                'data/normal\\N034.mat', 
                'data/normal\\N035.mat', 
                'data/normal\\N036.mat', 
                'data/normal\\N037.mat', 
                'data/normal\\N038.mat', 
                'data/normal\\N039.mat', 
                'data/normal\\N040.mat', 
                'data/normal\\N041.mat', 
                'data/normal\\N042.mat', 
                'data/normal\\N043.mat', 
                'data/normal\\N044.mat', 
                'data/normal\\N045.mat', 
                'data/normal\\N046.mat', 
                'data/normal\\N047.mat', 
                'data/normal\\N048.mat', 
                'data/normal\\N049.mat', 
                'data/normal\\N050.mat', 
                'data/normal\\N051.mat'
               ]


# ----------------------------
# FEATURE COLLECTION
# ----------------------------
X = []

for path in normal_files:
    phase_detr, t_slow = extract_phase_from_radar_file(path)
    print(f"Loaded phase from {path}, samples: {len(phase_detr)}")

    windows, _ = sliding_windows(phase_detr, FS_SLOW)

    for w in windows:
        feats = extract_resp_features(w, FS_SLOW)
        X.append(feats)

X = np.array(X)

# print(f"Training samples: {X.shape}")

# -----------------------
# Isolation Forest
# -----------------------
iso = IsolationForest(
    n_estimators=300,
    contamination=0.005,
    random_state=42,
    n_jobs=-1
)
iso.fit(X)

scores = iso.decision_function(X)

joblib.dump(iso, "models/isoforest.pkl")

# -----------------------
# Elliptic Envelope
# -----------------------
ell = EllipticEnvelope(
    contamination=0.005,
    support_fraction=0.9,
    random_state=42
)
ell.fit(X)

joblib.dump(ell, "models/elliptic.pkl")

# -----------------------
# kNN (distance-based)
# -----------------------
knn = NearestNeighbors(
    n_neighbors=10,
    metric="euclidean"
)
knn.fit(X)

# Precompute training distances
dists, _ = knn.kneighbors(X)
knn_threshold = np.percentile(dists.mean(axis=1), 99.5)

joblib.dump((knn, knn_threshold), "models/knn.pkl")

print("All models trained and saved.")

plt.figure(figsize=(6,4))
plt.hist(scores, bins=50)
plt.xlabel("Isolation Forest score")
plt.ylabel("Count")
plt.title("Anomaly score distribution (training data)")
plt.grid(True)
plt.show()

print("All models trained and saved.")
