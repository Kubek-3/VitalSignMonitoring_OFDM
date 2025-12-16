import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.fft import fft, fftshift, ifft
from scipy.signal import butter, filtfilt
from src.config import c, K, freqs, ups_factor, cf, TX_power_dBm, b, M, Fs_high, Nfft_time
from src.signal_processing.filters import bp_filter
from src.signal_processing.radar_model import amp



TX_power_W = 10 ** ((TX_power_dBm - 30) / 10.0)


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def qam16_constellation():
    """Returns 16-QAM unit-average-power constellation points."""
    m_vals = np.arange(M)
    I = 2 * ((m_vals % 4) - 1.5)
    Q = 2 * ((m_vals // 4) - 1.5)
    return (I + 1j * Q) / np.sqrt(10)

def generate_ofdm_pilot():
    """
    Generate:
      - a 16-QAM OFDM pilot in the frequency domain,
      - its oversampled baseband time-domain signal,
      - RF upconversion and PSD for visualization.
    """
    constellation = qam16_constellation()
    data_idx = np.random.randint(0, M, K)
    qamSymbols = constellation[data_idx]          # shape (K,)

    # Map to oversampled IFFT bins (centered)
    X = np.zeros(Nfft_time, dtype=complex)
    startIdx = Nfft_time // 2 - K//2
    X[startIdx:startIdx + K] = qamSymbols

    # Time-domain oversampled baseband OFDM
    ofdm_bb = ifft(fftshift(X))                  # complex baseband, one symbol

    # Normalize to 1 W average power
    power_signal = np.mean(np.abs(ofdm_bb)**2)
    ofdm_bb = ofdm_bb / np.sqrt(power_signal)

    # Scale to desired TX power
    ofdm_bb = ofdm_bb * np.sqrt(TX_power_W)

    # Time vector for one OFDM symbol
    t = np.arange(Nfft_time) / Fs_high

    # --- Baseband PSD ---
    Nfft_spec = 16384
    spec_base = fftshift(fft(ofdm_bb, Nfft_spec))
    Pxx_base = (np.abs(spec_base)**2) / (Nfft_spec * Fs_high)
    Pxx_base_dBHz = 10 * np.log10(Pxx_base + 1e-15)
    f_axis_base = np.linspace(-Fs_high/2, Fs_high/2, Nfft_spec)

    # --- RF Upconversion & PSD (for visualization) ---
    ofdm_rf = np.real(ofdm_bb * np.exp(1j * 2*np.pi*cf*t))

    spec_pass = fftshift(fft(ofdm_rf, Nfft_spec))
    Pxx_pass = (np.abs(spec_pass)**2) / (Nfft_spec * Fs_high)
    Pxx_pass_dBHz = 10 * np.log10(Pxx_pass + 1e-15)
    f_axis_pass = np.linspace(-Fs_high/2, Fs_high/2, Nfft_spec) + cf

    # Plot PSDs

    f_welch, Pxx_welch = welch(
        ofdm_bb,
        fs=Fs_high,
        nperseg=1024,
        return_onesided=False
    )

    plt.figure(figsize=(8, 4))
    plt.plot(fftshift(f_welch)/1e6, 10*np.log10(fftshift(Pxx_welch)))
    plt.title("Baseband OFDM PSD (Welch Averaged)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.grid(True)

    # plt.figure(figsize=(8, 4))
    # plt.plot(f_axis_base / 1e6, Pxx_base_dBHz)
    # plt.xlabel("Frequency (MHz)")
    # plt.ylabel("PSD (dB/Hz)")
    # plt.title("Baseband OFDM PSD")
    # plt.grid(True)

    f_welch_rf, Pxx_welch_rf = welch(
        ofdm_rf,
        fs=Fs_high,
        nperseg=1024,
        return_onesided=False
    )

    f_welch_rf = np.fft.fftshift(f_welch_rf)
    Pxx_welch_rf = np.fft.fftshift(Pxx_welch_rf)
    Pxx_welch_rf_dBHz = 10*np.log10(Pxx_welch_rf + 1e-15)

    # Option A: discrete-time axis (offset from DC)
    plt.figure()
    plt.plot(f_welch_rf/1e6, Pxx_welch_rf_dBHz)
    plt.xlabel("Discrete-time frequency (MHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("Passband OFDM PSD (Welch, aliased)")
    plt.grid(True)

    # # Option B: RF axis around 26.5 GHz
    # f_welch_rf_phys = f_welch_rf + cf
    # plt.figure()
    # plt.plot(f_welch_rf_phys/1e9, Pxx_welch_rf_dBHz)
    # plt.xlabel("Frequency (GHz)")
    # plt.ylabel("PSD (dB/Hz)")
    # plt.title("Passband OFDM PSD (Welch, absolute RF)")
    # plt.grid(True)


    # # Also show single-symbol time-domain magnitude
    # ofdm_time_baseband = np.fft.ifft(qamSymbols)
    # plt.figure()
    # plt.plot(np.abs(ofdm_time_baseband))
    # plt.title("Single OFDM Symbol (K samples) – Baseband Magnitude")
    # plt.xlabel("Sample index")
    # plt.ylabel("Magnitude")
    # plt.grid(True)

    return qamSymbols, ofdm_bb


def add_thermal_noise(signal, B, NF_dB=8.0, T0=290.0):
    """
    Add complex AWGN based on thermal noise and noise figure.
    signal shape: (N_slow, K) or similar.
    """
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
def simulate_ofdm_radar_end_to_end(d_tot, Fs_slow):
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

    N_slow = len(d_tot)
    # print(f"N_slow = {N_slow}, Fs_slow = {Fs_slow} Hz")

    # 1) Generate one OFDM pilot + RF view
    qamSymbols, ofdm_bb = generate_ofdm_pilot()

    # Frequency-domain pilot vector repeated each pulse
    TX = np.tile(qamSymbols, (N_slow, 1))     # shape (N_slow, K)

    # 2) Build baseband-equivalent radar channel H[n,k]
    #    τ[n] = 2 d[n] / c  (two-way delay)
    tau_1 = 2.0 * d_tot / c                     # (N_slow,)
    #tau_2 = 2.0 * d_tot2 / c                   # (N_slow,)
    freqs_2d = freqs[np.newaxis, :]          # (1, K)
    tau_2d_1 = tau_1[:, np.newaxis]              # (N_slow, 1)  
    #tau_2d_2 = tau_2[:, np.newaxis]              # (N_slow, 1)
    amp_2d_1 = amp(d_tot, freqs_2d)             # (N_slow, 1)
    #amp_2d_2 = amp(d_tot2, freqs_2d)             # (N_slow, 1)


    # Phase = -2π f_k τ[n]  (baseband-equivalent)
    phase_2d_1 = -2.0 * np.pi * freqs_2d * tau_2d_1
    # phase_2d_2 = -2.0 * np.pi * freqs_2d * tau_2d_2
    H_1 = amp_2d_1 * np.exp(1j * phase_2d_1)       # (N_slow, K)
    #H_2 = amp_2d_2 * np.exp(1j * phase_2d_2)       # (N_slow, K)
    H = H_1                            # superposition from two targets

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
    print(f"Measured SNR ≈ {snr_db_meas:.2f} dB")

    # 4) Range processing: IFFT across subcarriers (fast-time)
    H_range = np.fft.ifft(RX_noisy / TX, axis=1)   # (N_slow, K)
    h_mag = np.abs(H_range)

    avg_profile = np.mean(h_mag, axis=0)
    r_bin = np.argmax(avg_profile)
    # peaks, props = find_peaks(
    #     avg_profile,
    #     height=np.max(avg_profile) * 0.3,   # threshold
    #     distance=3                           # bins separation
    # )

    # Sort strongest first
    #peaks = peaks[np.argsort(props["peak_heights"])[::-1]]
    # print("Strongest range bin index:", r_bin)

    # Plot average range profile (magnitude)
    # plt.figure()
    # plt.plot(avg_profile)
    # plt.title("Średni profil zasięgu |H_range|")
    # plt.xlabel("Indeks binu zasięgu")
    # plt.ylabel("Magnituda")
    # plt.grid(True)
    # plt.show()

    # 5) Extract slow-time complex signal at that range bin
    h_slow = H_range[:, r_bin]      # shape (N_slow,), one person
    # h1 = H_range[:, peaks[0]]
    # h2 = H_range[:, peaks[1]]

    # phase1 = np.unwrap(np.angle(h1))
    # phase2 = np.unwrap(np.angle(h2))


    # 6) Phase vs slow time
    phase_slow = np.unwrap(np.angle(h_slow))

    t_slow = np.arange(N_slow) / Fs_slow
    print("Slow-time observation duration (s):", t_slow[-1])
    print("Slow-time samples (N_slow):", N_slow)
    p_coeff = np.polyfit(t_slow, phase_slow, 1)
    phase_detr = phase_slow - np.polyval(p_coeff, t_slow)

    # 7) Bandpass for respiration and heart
    # Respiration: 0.1–0.5 Hz (6–30 bpm)
    phase_resp = bp_filter(phase_detr, Fs_slow, 0.1, 0.5)

    # Heart: 0.8–2.0 Hz (48–120 bpm)
    phase_heart = bp_filter(phase_detr, Fs_slow, 0.8, 2.0)

    # Plot phase signals
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t_slow, phase_detr)
    plt.title("Faza na docelowym zakresie")
    plt.ylabel("Faza (rad)")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_slow, phase_heart)
    plt.title("Faza w paśmie sercowym (0.8–2.0 Hz)")
    plt.xlabel("Czas (s)")
    plt.ylabel("Faza (rad)")
    plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    return (
        h_slow,
        avg_profile,
        r_bin,
        phase_resp,
        phase_heart,
        p_coeff,
        t_slow,
    )


# ------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------
if __name__ == "__main__":
    # Slow-time sampling (PRF)
    Fs_slow_example = 50.0       # 50 Hz
    T_obs = 20.0                 # observe 20 seconds
    N_slow_example = int(Fs_slow_example * T_obs)
    t_slow_example = np.arange(N_slow_example) / Fs_slow_example

    # Target distance:
    # 1.0 m average + respiration-like motion (0.25 Hz)
    d_tot_example = 1.0 + 0.005 * np.sin(2*np.pi*0.25 * t_slow_example)

    # For simplicity: constant amplitude (no fading)
    amp_example = np.ones_like(d_tot_example, dtype=float)

    simulate_ofdm_radar_end_to_end(d_tot_example, Fs_slow_example, amp_example)
