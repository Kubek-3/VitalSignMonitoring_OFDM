from scipy.signal import hilbert, find_peaks
import numpy as np

def extract_resp_features(phase_resp, Fs):
    analytic = hilbert(phase_resp)
    amp = np.abs(analytic)

    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) * Fs / (2 * np.pi)
    inst_freq = inst_freq[np.isfinite(inst_freq)]
    amp = amp[:len(inst_freq)]

    # Peak timing
    peaks, _ = find_peaks(amp, distance=Fs*1.0)
    if len(peaks) > 2:
        periods = np.diff(peaks) / Fs
        mean_period = np.mean(periods)
        std_period = np.std(periods)
        period_cv = std_period / (mean_period + 1e-6)
    else:
        mean_period = 0
        std_period = 0
        period_cv = 1.0

   
    p2p = np.ptp(phase_resp)

    features = np.array([
        np.mean(amp),
        np.std(amp),
        np.mean(inst_freq),
        np.std(inst_freq),
        mean_period,
        std_period,
        period_cv,
        p2p            # <<< ADD THIS
    ])

    return features
