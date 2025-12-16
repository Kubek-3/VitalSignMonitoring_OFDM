import numpy as np
from scipy.signal import hilbert

def extract_resp_features(phase_resp, Fs):
    """
    Extract respiration features from detrended + bandpassed phase.

    Returns: feature vector (1D numpy array)
    """
    # Analytic signal
    analytic = hilbert(phase_resp)
    amp = np.abs(analytic)

    # Instantaneous frequency
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) * Fs / (2 * np.pi)

    # Robust trimming
    inst_freq = inst_freq[np.isfinite(inst_freq)]
    amp = amp[:len(inst_freq)]

    features = np.array([
        np.mean(amp),
        np.std(amp),
        np.ptp(amp),                 # amplitude variability
        np.mean(inst_freq),
        np.std(inst_freq),           # breathing regularity
    ])

    return features
