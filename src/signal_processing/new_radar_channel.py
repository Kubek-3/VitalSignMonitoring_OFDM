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
    Nfft_spec = 10384
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

    # plt.figure()
    # plt.plot(f_axis_pass/1e9, Pxx_pass_dBHz)
    # plt.xlabel("Frequency (GHz)")
    # plt.ylabel("PSD (dB/Hz)")
    # plt.title("Passband OFDM PSD centered at 26.5 GHz")
    # plt.grid(True)

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
def simulate_ofdm_radar_end_to_end(d_tot, Fs_slow, filename, output_folder):
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
    qamSymbols, ofdm_bb = generate_ofdm_pilot()

    plt.figure()
    plt.scatter(np.real(qamSymbols), np.imag(qamSymbols), s=20)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join(output_folder, filename.replace(".mat", "_6_qam_pilot_constellation.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


    original_time = np.arange(len(ofdm_bb)) / FS_SLOW

    # Frequency-domain pilot vector repeated each pulse
    TX = np.tile(qamSymbols, (N_slow, 1))     # shape (N_slow, K)

    # 2) Build baseband-equivalent radar channel H[n,k]
    #    τ[n] = 2 d[n] / c  (two-way delay)
    tau_1 = d_tot / C                     # (N_slow,)
    freqs_2d = freqs[np.newaxis, :]          # (1, K)
    tau_2d_1 = tau_1[:, np.newaxis]              # (N_slow, 1)  
    amp_2d_1 = amp(d_tot, freqs_2d)             # (N_slow, 1)

    tau_1_pico = tau_1 * 1e9

    original_time = np.arange(len(tau_1)) / Fs_slow
    plt.plot(original_time, tau_1_pico)
    plt.xlabel("Time [s]")
    plt.ylabel("Two-way Delay [ns]")
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    out_png = os.path.join(output_folder, filename.replace(".mat", "_2_two_way_delay_vs_slow_time.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    sum_amp = np.sum(amp_2d_1, axis=1) * 1e3
    plt.plot(original_time, sum_amp)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [√μW]")
    out_png = os.path.join(output_folder, filename.replace(".mat", "_3_channel_amplitude_vs_slow_time.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


    # Phase = -2π f_k τ[n]  (baseband-equivalent)
    phase_2d_1 = -2.0 * np.pi * freqs_2d * tau_2d_1

    
    plt.plot(original_time, phase_2d_1[:, K//2])
    plt.xlabel("Time [s]")
    plt.ylabel("Phase [rad]")
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    out_png = os.path.join(output_folder, filename.replace(".mat", "_4_channel_phase_vs_slow_time.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    H = amp_2d_1 * np.exp(1j * phase_2d_1)       # (N_slow, K)
    # h_sum = np.sum(H, axis=1)
    plt.figure()
    plt.plot(original_time,H)
    plt.xlabel("Time [s]")
    plt.ylabel("Phase [rad]")
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    out_png = os.path.join(output_folder, filename.replace(".mat", "_5_channel_frequency_responseNOTSUMMED_vs_slow_time.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    h_sum = np.sum(H, axis=1)
    plt.figure()
    plt.plot(original_time,h_sum)
    plt.xlabel("Time [s]")
    plt.ylabel("Phase [rad]")
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    out_png = os.path.join(output_folder, filename.replace(".mat", "_5_channel_frequency_response_vs_slow_time.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    #  Conceptual RF chain:
    #  - DAC: ofdm_bb(t) (complex baseband) -> upconvert to RF
    #  - Channel: delay & attenuation at RF
    #  - LNA + mixer: downconvert to complex baseband
    #  -> mathematically equivalent to applying H[n,k] to subcarriers.
    
    RX_ideal = TX * H                         # (N_slow, K)

    plt.figure()
    plt.scatter(np.real(RX_ideal), np.imag(RX_ideal), s=20)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    out_png = os.path.join(output_folder, filename.replace(".mat", "_7_rx_constellation_ideal.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # 3) Add noise (thermal + NF)
    RX_noisy, noise = add_thermal_noise(RX_ideal, b, NF_dB=10.0)
    TX_noisy, noise_tx = add_thermal_noise(TX, b, NF_dB=10.0)

    plt.figure()
    plt.scatter(np.real(RX_noisy), np.imag(RX_noisy), s=20)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    plt.axis("equal")
    out_png = os.path.join(output_folder, filename.replace(".mat", "_9_rx_constellation_noisy.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    #rx correcrion for plot
    h_est = H
    eps = 1e-12  # numerical stability
    rx_corrected_noisy = RX_noisy / (h_est + eps)
    rx_corrected = RX_ideal / (h_est + eps)

    plt.figure()
    plt.scatter(
        np.real(rx_corrected),
        np.imag(rx_corrected),
        s=2,
        alpha=0.5)
    plt.xlabel("IQ")
    plt.ylabel("Q")
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    out_png = os.path.join(output_folder, filename.replace(".mat", "_10_rx_ideal_corrected.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(
    np.real(rx_corrected_noisy),
    np.imag(rx_corrected_noisy),
    s=2,
    alpha=0.5)
    plt.xlabel("IQ")
    plt.ylabel("Q")
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    out_png = os.path.join(output_folder, filename.replace(".mat", "_10_rx_noisy_corrected.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    # Measured SNR
    signal_power = np.mean(np.abs(RX_ideal)**2)
    noise_power_meas = np.mean(np.abs(noise)**2)
    snr_linear = signal_power / noise_power_meas
    snr_db_meas = 10 * np.log10(snr_linear)
    #print(f"Measured SNR ≈ {snr_db_meas:.2f} dB")

    # 4) Range processing: IFFT across subcarriers (fast-time)
    H_range = np.fft.ifft(RX_noisy / TX, axis=1)   # (N_slow, K)
    h_mag = np.abs(H_range)

    # plt.figure()
    # plt.imshow(np.abs(H_range), aspect="auto")
    # plt.colorbar(label="Magnitude")
    # plt.xlabel("Range Bin")
    # plt.ylabel("Slow Time Index")
    # plt.title("Range Map |H_range[n, r]|")
    # out_png = os.path.join(output_folder, filename.replace(".mat", "_8_range_map.png"))
    # plt.tight_layout()
    # plt.savefig(out_png, dpi=200)
    # plt.close()


    avg_profile = np.mean(h_mag, axis=0)
    r_bin = np.argmax(avg_profile)

    # Plot range profile
    plot_range_profile(avg_profile, r_bin, filename, output_folder)

    # Plot average range profile (magnitude)
    plot_avg_profile(avg_profile, filename, output_folder)

    # 5) Extract slow-time complex signal at that range bin
    h_slow = H_range[:, r_bin]      # shape (N_slow,), one person

    h_micro = np.abs(h_slow) * 1e3
    plt.figure()
    plt.plot(original_time, h_micro)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [√μW]")
    plt.title("Slow-Time Signal at Target Range Bin")
    out_png = os.path.join(output_folder, filename.replace(".mat", "_11_slow_time_signal_amplitude.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # 6) Phase vs slow time
    phase_slow = np.unwrap(np.angle(h_slow))

    plt.figure()
    plt.plot(original_time, phase_slow)
    plt.xlabel("Time [s]")
    plt.ylabel("Phase (rad)") 
    plt.title("Phase vs Slow Time")
    out_png = os.path.join(output_folder, filename.replace(".mat", "_12_phase_vs_slow_time_raw.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


    t_slow = np.arange(N_slow) / Fs_slow
    p_coeff = np.polyfit(t_slow, phase_slow, 1)
    phase_detr = phase_slow - np.polyval(p_coeff, t_slow)

    plt.figure()
    plt.plot(t_slow, np.real(h_slow) * 1e3, label="Real")
    plt.plot(t_slow, np.imag(h_slow) * 1e3, label="Imag")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [√μW]")
    plt.title("Slow-Time Signal at Target Range Bin")
    plt.legend()
    out_png = os.path.join(output_folder, filename.replace(".mat", "_13_slow_time_signal.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


    plt.figure()
    plt.plot(t_slow, phase_slow, label="Raw Phase")
    plt.plot(t_slow, phase_detr, label="Detrended Phase")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.title("Phase vs Slow Time")
    plt.legend()
    plt.grid()
    out_png = os.path.join(output_folder, filename.replace(".mat", "_14_phase_vs_slow_time.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()



    # 7) Bandpass for respiration and heart
    # Respiration: 0.1–0.5 Hz (6–30 bpm)
    phase_for_resp = detrend(phase_detr, "linear")
    phase_for_resp = phase_for_resp - np.mean(phase_for_resp)
    phase_resp = lowpass_filter(phase_for_resp, Fs_slow, 0.5)

    # Heart: 0.8–2.0 Hz (48–120 bpm)
    phase_heart = bp_filter(phase_detr, Fs_slow, 0.8, 2.0)

    plot_phase_signals(t_slow, phase_detr, phase_resp, phase_heart, filename, output_folder)
    
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
