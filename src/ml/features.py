from scipy.signal import hilbert, find_peaks
import numpy as np

from src.pipeline_windows import estimate_rate
from src.signal_processing.filters import lowpass_filter
from src.config import FS_SLOW
from matplotlib.mlab import detrend

def extract_resp_features(phase_detr, Fs):
    """
    Physiology-aware respiration features.
    """

    phase_for_resp = detrend(phase_detr, "linear")
    phase_for_resp = phase_for_resp - np.mean(phase_for_resp)
    phase_resp = lowpass_filter(phase_for_resp, FS_SLOW, 0.5)

    # Analytic signal
    analytic = hilbert(phase_resp)
    amp = np.abs(analytic)

    # Rate estimation
    br, (mean_T, min_T, max_T) = estimate_rate(
        amp,
        FS_SLOW,
        min_dist=2.0  # seconds (30 bpm max)
    )

    # Handle apnea / flat
    if np.isnan(br):
        br = 0.0
        mean_T, min_T, max_T = 0.0, 0.0, 0.0

    # Amplitude features
    amp_mean = np.mean(amp)
    amp_std  = np.std(amp)
    amp_p2p  = np.ptp(amp)

    # Regularity
    period_cv = (
        (max_T - min_T) / (mean_T + 1e-6)
        if mean_T > 0 else 1.0
    )

    features = np.array([
        br,             # breaths per minute
        amp_mean,
        amp_std,
        amp_p2p,
        mean_T,
        period_cv
    ])

    return features

