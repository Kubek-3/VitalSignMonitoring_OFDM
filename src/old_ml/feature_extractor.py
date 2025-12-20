import numpy as np
from scipy.signal import welch


def extract_features_from_phase(phase,
                                Fs_slow,
                                window_sec=10.0,
                                step_sec=2.0):
    """
    Convert phase signal into ML features using sliding windows.

    Returns:
        X     : (N_windows, N_features)
        t_mid : mid-time of each window
    """

    win_len = int(window_sec * Fs_slow)
    step = int(step_sec * Fs_slow)

    features = []
    t_mid = []

    for start in range(0, len(phase) - win_len, step):
        seg = phase[start:start + win_len]

        # ---- Time-domain features ----
        std_phase = np.std(seg)
        ptp_phase = np.ptp(seg)

        # ---- Frequency-domain features ----
        f, Pxx = welch(seg, fs=Fs_slow, nperseg=min(256, len(seg)))
        Pxx = Pxx + 1e-12
        Pxx_norm = Pxx / np.sum(Pxx)

        # Dominant frequency (breathing)
        dom_freq = f[np.argmax(Pxx)]

        # Spectral entropy
        spec_entropy = -np.sum(Pxx_norm * np.log(Pxx_norm))

        # Energy in respiration band (0.1–0.5 Hz)
        resp_band = (f >= 0.1) & (f <= 0.5)
        resp_energy = np.sum(Pxx[resp_band])

        features.append([
            std_phase,
            ptp_phase,
            dom_freq,
            spec_entropy,
            resp_energy
        ])

        t_mid.append((start + win_len / 2) / Fs_slow)

    X = np.array(features)
    t_mid = np.array(t_mid)

    return X, t_mid
