import numpy as np
from src.config import c, ref_cof, TX_power_dBm, K

def compute_phase(d, f):
    lam = c / f
    return (2 * np.pi / lam) * d

def free_space_path_loss(d, f):

    """Calculate free-space path loss in dB.
    Args:
        d (1D array): Path length over time.
        f (float): Frequency.
    Returns:
        1D array: Path loss in dB over time."""
    
    L = (4 * np.pi / c)
    L_dB = 20 * np.log10(d) + 20 * np.log10(f) + 20 * np.log10(L) + 20 * np.log10(ref_cof) # reflection coefficient from human included here
    return L_dB

def compute_path_loss(d, f):
    L = (4 * np.pi / c)
    return 20*np.log10(d) + 20*np.log10(f) + 20*np.log10(L) + 20*np.log10(ref_cof)


def pw_recvd_dBm(PL_dB):
    return TX_power_dBm - PL_dB


def pw_recvd_w(pw_r_dBm):
    return 1e-3 * 10**(pw_r_dBm / 10)

def amp(d, f):
    d = np.asarray(d).reshape(-1, 1)   # distances
    f = np.asarray(f).reshape(1, -1)   # frequencies
    # print("amp() d shape:", d.shape)
    # print("amp() f shape:", f.shape)
    PL_dB = free_space_path_loss(d, f)
    pw_r_dBm = pw_recvd_dBm(PL_dB)
    pw_r_w = pw_recvd_w(pw_r_dBm)
    amp = np.sqrt(pw_r_w)
    return amp


def add_noise(pw_r_w, snr_db):
    sig_power = np.mean(pw_r_w)
    noise_power = sig_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(pw_r_w))
    return np.sqrt(pw_r_w + noise)


def reshape_for_subcarriers(signal):
    n_full = (len(signal) // K) * K
    return signal[:n_full].reshape(-1, K)


def dsp(tx, rx):
    RX = np.fft.fft(rx, axis=1)
    TX = np.fft.fft(tx, axis=1)
    H = RX / TX
    h = np.fft.ifft(H, axis=1)
    h_mag = np.abs(h)
    max_bin = np.argmax(np.mean(h_mag, axis=0))
    h_max = h[:, max_bin]
    changes = np.fft.fftshift(np.fft.fft(np.angle(h_max)))
    return H, h, h_max, changes
