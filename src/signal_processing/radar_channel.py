import numpy as np
import matplotlib.pyplot as plt
from src.config import c, K, t_snr_db, freqs
from src.signal_processing.filters import bp_filter

def radar_channel(d_tot, Fs_slow, amp):
    # 1) Slow-time definition
    N_slow = len(d_tot)   # number of pulses / OFDM frames

    print("Slow-time samples:", N_slow, "Fs_slow:", Fs_slow, "Hz")

    # 2) Build TX pilot on each subcarrier (complex baseband)
    M = 16
    m_vals = np.arange(M)
    I = 2*((m_vals % 4) - 1.5)
    Q = 2*((m_vals // 4) - 1.5)
    constellation = (I + 1j*Q) / np.sqrt(10)

    # Random QAM per subcarrier (you could also use all-ones pilot)
    data_idx = np.random.randint(0, M, K)
    qamSymbols = constellation[data_idx]     # shape (K,)

    # 3) Build channel H[n,k] for a single point target whose distance varies with time
    #    τ[n] = 2 d_tot[n] / c  (two-way delay)
    tau = 2.0 * d_tot / c   # shape (N_slow,)

    # convert freqs vector into shape (1,K) for broadcasting
    freqs_2d = freqs[np.newaxis, :]          # (1, K)
    tau_2d = tau[:, np.newaxis]              # (N_slow, 1)

    # Static amplitude from path loss, already in pw_r_w; we used amp = sqrt(pw_r_w)
    amp_2d = amp[:, np.newaxis]              # (N_slow, 1)

    # Frequency-dependent phase: -2π f_k τ[n]
    phase_2d = -2.0 * np.pi * freqs_2d * tau_2d   # (N_slow, K)

    H = amp_2d * np.exp(1j * phase_2d)            # (N_slow, K), complex channel

    # 4) Form TX and RX matrices: TX is same pilot each pulse
    TX = np.tile(qamSymbols, (N_slow, 1))                  # (N_slow, K)
    RX = TX * H                                   # (N_slow, K) ideal, no noise

  
    # ----------- THERMAL NOISE + NOISE FIGURE MODEL -----------

    kB = 1.380649e-23   # Boltzmann constant
    T0 = 290.0          # Room temperature (Kelvin)
    NF_db = 8.0        # Noise Figure in dB
    NF_lin = 10**(NF_db / 10.0)

    # Signal bandwidth from OFDM
    B = np.max(freqs) - np.min(freqs)

    # Thermal noise power (Watts)
    noise_power = kB * T0 * B * NF_lin

    # Complex AWGN (per complex dimension)
    noise_std = np.sqrt(noise_power / 2)

    noise = noise_std * (
        np.random.randn(*RX.shape) + 1j * np.random.randn(*RX.shape)
    )

    RX_noisy = RX + noise

    # ----------- MEASURED SNR CALCULATION -----------

    signal_power = np.mean(np.abs(RX)**2)
    noise_power_meas = np.mean(np.abs(noise)**2)

    snr_linear = signal_power / noise_power_meas
    snr_db_measured = 10 * np.log10(snr_linear)

    print(f"Measured SNR: {snr_db_measured:.2f} dB")
    # ------------------------------------------------------------


    # 5) Range processing: FFT across subcarriers (fast-time) for each slow-time pulse
    H_range = np.fft.ifft(RX_noisy / TX, axis=1)    # (N_slow, K), similar to your dsp() idea
    h_mag = np.abs(H_range)

    # Find strongest range bin (averaged over slow-time)
    avg_profile = np.mean(h_mag, axis=0)
    r_bin = np.argmax(avg_profile)
    print("Strongest range bin index:", r_bin)

    # Extract complex slow-time signal at that bin
    h_slow = H_range[:, r_bin]          # shape (N_slow,)

    # ---------------- BR / HR estimation from phase -----------------

    # 6) Phase over time
    phase_slow = np.unwrap(np.angle(h_slow))

    # Remove linear trend (distance offset) – crude detrend
    t_slow = np.arange(N_slow) / Fs_slow
    p_coeff = np.polyfit(t_slow, phase_slow, 1)
    phase_detr = phase_slow - np.polyval(p_coeff, t_slow)
    
    # Respiration: ~0.1–0.5 Hz (6–30 bpm)
    phase_resp = bp_filter(phase_detr, Fs_slow, 0.1, 0.5)

    # Heart: ~0.8–2.0 Hz (48–120 bpm)
    phase_heart = bp_filter(phase_detr, Fs_slow, 0.8, 2.0)

    qamSymbols = constellation[data_idx]

    ofdm_time = np.fft.ifft(qamSymbols)
    ofdm_mag = np.abs(ofdm_time)

    plt.figure()
    plt.plot(ofdm_mag)
    plt.title("OFDM Domena Czasowa Sygnału - Magnituda")
    plt.xlabel("Indeks Próbki")
    plt.ylabel("Magnituda")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(t_slow, phase_resp, label="Sygnał Fazy Oddychania", color='blue')
    plt.plot(t_slow, phase_heart, label="Sygnał Fazy Serca", color='orange')
    plt.title("Sygnały Fazy w Czasie Wolnym")
    plt.xlabel("Czas (s)")
    plt.ylabel("Faza (radiany)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    
    return h_slow, avg_profile, r_bin, phase_resp, phase_heart, p_coeff, t_slow