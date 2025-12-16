import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.ml.features import extract_resp_features
from src.ml.windowing import sliding_windows
from src.signal_processing.phase_for_ml import extract_phase_from_radar_file
from src.config import Fs_slow, data_fs, ups_factor

# ----------------------------
# CONFIG
# ----------------------------
WIN_SEC = 10.0
STEP_SEC = 2.0

MODEL_PATH = "models/isoforest_resp.pkl"
TEST_FILE = "data/irregular/IR003.mat"

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load(MODEL_PATH)

# ----------------------------
# LOAD RADAR PHASE
# ----------------------------
phase_detr, t_slow = extract_phase_from_radar_file(TEST_FILE)

windows, win_times = sliding_windows(phase_detr, Fs_slow, WIN_SEC, STEP_SEC)

# ----------------------------
# FEATURE + ANOMALY SCORE
# ----------------------------
X = np.array([
    extract_resp_features(w, Fs_slow) for w in windows
])

scores = model.decision_function(X)
anomaly = scores < 0   # True = anomaly

original_time = np.arange(len(phase_detr)) / data_fs / ups_factor

# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(12, 5))
plt.plot(original_time, phase_detr, label="Radar phase (detrended)", lw=1)

for is_anom, center_t in zip(anomaly, win_times):
    if is_anom:
        plt.axvspan(
            center_t - WIN_SEC / 2,
            center_t + WIN_SEC / 2,
            color="red",
            alpha=0.2
        )

plt.xlabel("Time (s)")
plt.ylabel("Phase (rad)")
plt.title("Radar Phase at Human Range with Detected Breathing Irregularities")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ----------------------------