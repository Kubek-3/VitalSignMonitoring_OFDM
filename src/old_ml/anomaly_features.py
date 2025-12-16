import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

def bandpass(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def spectral_entropy(mag):
    p = mag / (np.sum(mag) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

def window_features(sig, fs, win_s=10, step_s=2, f_low=0.1, f_high=0.5):
    win = int(win_s * fs)
    step = int(step_s * fs)

    features = []
    t_centers = []

    for i in range(0, len(sig)-win, step):
        seg = sig[i:i+win]
        t_centers.append((i + win/2) / fs)

        # time-domain metrics
        rms = np.sqrt(np.mean(seg**2))
        p2p = np.ptp(seg)
        zcr = np.mean(np.diff(np.sign(seg)) != 0)

        # frequency-domain metrics
        spec = rfft(seg * np.hanning(len(seg)))
        freqs = rfftfreq(len(seg), 1/fs)
        mag = np.abs(spec)

        mask = (freqs >= f_low) & (freqs <= f_high)

        if np.any(mask):
            dom = freqs[mask][np.argmax(mag[mask])]
            prom = np.max(mag[mask]) / (np.mean(mag[mask]) + 1e-12)
        else:
            dom = 0.0
            prom = 1.0

        ent = spectral_entropy(mag[mask] if np.any(mask) else mag)

        features.append([rms, p2p, zcr, dom, prom, ent])

    return np.array(features), np.array(t_centers)
