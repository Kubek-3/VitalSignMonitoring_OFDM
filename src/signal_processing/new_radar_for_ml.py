import os
from matplotlib.mlab import detrend
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.fft import fft, fftshift, ifft
from scipy.signal import butter, filtfilt
from src.config import C, K, freqs, ups_factor, cf, TX_power_dBm, b, M, Fs_high, Nfft_time, NF, FS_SLOW
from src.signal_processing.filters import bp_filter, lowpass_filter
from src.signal_processing.radar_model import amp
from src.visualisation.plot_phase_signals import plot_phase_signals
from src.visualisation.range_profile import plot_range_profile
from src.visualisation.plot_avg_profile import plot_avg_profile



TX_power_W = 10 ** ((TX_power_dBm - 30) / 10.0)


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def qam16_constellation():

    m_vals = np.arange(M)
    I = 2 * ((m_vals % 4) - 1.5)
    Q = 2 * ((m_vals // 4) - 1.5)
    return (I + 1j * Q) / np.sqrt(10)

def generate_ofdm_pilot():

    constellation = qam16_constellation()
    data_idx = np.random.randint(0, M, K)
    qamSymbols = constellation[data_idx]          # shape (K,)

    # Map to oversampled IFFT bins (centered)
    X = np.zeros(Nfft_time, dtype=complex)
    startIdx = Nfft_time // 2 - K//2
    X[startIdx:startIdx + K] = qamSymbols

    return qamSymbols


def add_thermal_noise(signal, B, NF_dB=NF, T0=290.0):
    """
    Add complex AWGN based on thermal noise and noise figure.
    signal shape: (N_slow, K) or similar.
    """
    # 290K = 16.85 °C
    kB = 1.380649e-23   # Boltzmann constant
    NF_lin = 10**(NF_dB / 10.0)

    noise_power = kB * T0 * B * NF_lin
    noise_std = np.sqrt(noise_power / 2)

    noise = noise_std * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise, noise


# ------------------------------------------------------------
# MAIN END-TO-END RADAR SIMULATION
# ------------------------------------------------------------
def simulate_ofdm_radar_end_to_end_for_ml(d_tot, Fs_slow):
    """
    End-to-end model:
      RF (conceptual) -> baseband-equivalent channel -> radar processing -> vitals.

    Inputs:
      d_tot  : distances vs slow-time (meters), shape (N_slow,)
      Fs_slow: slow-time sampling frequency (Hz) i.e. pulse repetition frequency
      amp    : complex amplitudes vs time, shape (N_slow,)

    Returns:
      h_slow, avg_profile, r_bin,
      phase_resp, phase_heart, p_coeff, t_slow
    """

    N_slow = len(d_tot)       # number of slow-time samples
    # print(f"N_slow = {N_slow}, Fs_slow = {Fs_slow} Hz")

    # 1) Generate one OFDM pilot + RF view
    qamSymbols = generate_ofdm_pilot()

    # Frequency-domain pilot vector repeated each pulse
    TX = np.tile(qamSymbols, (N_slow, 1))     # shape (N_slow, K)

    # 2) Build baseband-equivalent radar channel H[n,k]
    #    τ[n] = 2 d[n] / c  (two-way delay)
    tau_1 = d_tot / C                     # (N_slow,)
    freqs_2d = freqs[np.newaxis, :]          # (1, K)
    tau_2d_1 = tau_1[:, np.newaxis]              # (N_slow, 1)  
    amp_2d_1 = amp(d_tot, freqs_2d)             # (N_slow, 1)

    # Phase = -2π f_k τ[n]  (baseband-equivalent)
    phase_2d_1 = -2.0 * np.pi * freqs_2d * tau_2d_1
    H = amp_2d_1 * np.exp(1j * phase_2d_1)       # (N_slow, K)

    #  Conceptual RF chain:
    #  - DAC: ofdm_bb(t) (complex baseband) -> upconvert to RF
    #  - Channel: delay & attenuation at RF
    #  - LNA + mixer: downconvert to complex baseband
    #  -> mathematically equivalent to applying H[n,k] to subcarriers.

    RX_ideal = TX * H                         # (N_slow, K)

    # 3) Add noise (thermal + NF)
    RX_noisy, noise = add_thermal_noise(RX_ideal, b, NF_dB=10.0)

    # Measured SNR
    signal_power = np.mean(np.abs(RX_ideal)**2)
    noise_power_meas = np.mean(np.abs(noise)**2)
    snr_linear = signal_power / noise_power_meas
    snr_db_meas = 10 * np.log10(snr_linear)
    #print(f"Measured SNR ≈ {snr_db_meas:.2f} dB")

    # 4) Range processing: IFFT across subcarriers (fast-time)
    H_range = np.fft.ifft(RX_noisy / TX, axis=1)   # (N_slow, K)
    h_mag = np.abs(H_range)

    avg_profile = np.mean(h_mag, axis=0)
    r_bin = np.argmax(avg_profile)

    # 5) Extract slow-time complex signal at that range bin
    h_slow = H_range[:, r_bin]      # shape (N_slow,), one person

    # 6) Phase vs slow time
    phase_slow = np.unwrap(np.angle(h_slow))

    t_slow = np.arange(N_slow) / Fs_slow
    p_coeff = np.polyfit(t_slow, phase_slow, 1)
    phase_detr = phase_slow - np.polyval(p_coeff, t_slow)

    # 7) Bandpass for respiration and heart
    # Respiration: 0.1–0.5 Hz (6–30 bpm)
    phase_for_resp = detrend(phase_detr, "linear")
    phase_for_resp = phase_for_resp - np.mean(phase_for_resp)
    phase_resp = lowpass_filter(phase_for_resp, Fs_slow, 0.5)

    # Heart: 0.8–2.0 Hz (48–120 bpm)
    phase_heart = bp_filter(phase_detr, Fs_slow, 0.8, 2.0)
    
    return (
        h_slow,
        avg_profile,
        r_bin,
        phase_detr,
        phase_resp,
        phase_heart,
        p_coeff,
        t_slow,
    )
