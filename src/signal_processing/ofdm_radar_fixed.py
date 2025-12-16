import numpy as np
from numpy.fft import fftshift, fft, ifft
from src.config import (
    c, cf, K, b, M, Nfft_time, Fs_high, TX_power_dBm
)
from src.signal_processing.filters import bp_filter
import matplotlib.pyplot as plt
from scipy.signal import welch


# ------------------------------------------------------------
# 1) Constellation + pilot generation
# ------------------------------------------------------------
def qam16_constellation():
    m = np.arange(M)
    I = 2 * ((m % 4) - 1.5)
    Q = 2 * ((m // 4) - 1.5)
    return (I + 1j * Q) / np.sqrt(10)


def generate_pilot_symbol():
    """Generate ONE clean OFDM pilot symbol (all subcarriers known)."""
    const = qam16_constellation()
    pilot = const[np.random.randint(0, M, K)]

    return pilot

def plot_ofdm_psd(ofdm_bb):
    """
    Computes and plots:
      • Baseband OFDM PSD
      • Passband PSD after upconversion to cf

    Input: ofdm_bb — oversampled baseband OFDM signal (complex)
    """

    TX_power_W = 10 ** ((TX_power_dBm - 30)/10)

    # Normalize to 1 watt average power
    power = np.mean(np.abs(ofdm_bb)**2)
    ofdm_bb = ofdm_bb / np.sqrt(power)
    ofdm_bb *= np.sqrt(TX_power_W)

    N = len(ofdm_bb)
    t = np.arange(N) / Fs_high

    # -------------------------------
    # BASEBAND PSD
    # -------------------------------
    Nfft = 32768
    spec_base = fftshift(fft(ofdm_bb, Nfft))
    Pxx_base = (np.abs(spec_base)**2) / (Nfft * Fs_high)
    Pxx_base_dB = 10*np.log10(Pxx_base + 1e-15)

    f_axis_base = np.linspace(-Fs_high/2, Fs_high/2, Nfft)

    # -------------------------------
    # PASSBAND UP-CONVERSION
    # -------------------------------
    ofdm_pass = np.real(ofdm_bb * np.exp(1j*2*np.pi*cf*t))

    spec_pass = fftshift(fft(ofdm_pass, Nfft))
    Pxx_pass = (np.abs(spec_pass)**2) / (Nfft * Fs_high)
    Pxx_pass_dB = 10*np.log10(Pxx_pass + 1e-15)

    f_axis_pass = np.linspace(-Fs_high/2, Fs_high/2, Nfft)

    # -------------------------------
    # PLOTS
    # -------------------------------
    plt.figure(figsize=(10,4))
    plt.plot(f_axis_base/1e6, Pxx_base_dB)
    plt.title("Baseband OFDM PSD")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.grid(True)

    plt.figure(figsize=(10,4))
    plt.plot(f_axis_pass/1e6, Pxx_pass_dB)
    plt.title(f"Passband PSD centered at {cf/1e6:.2f} MHz")
    plt.xlabel("Frequency (MHz offset from carrier)")
    plt.ylabel("PSD (dB/Hz)")
    plt.grid(True)

    # Zoom around OFDM band
    bw_MHz = b/1e6
    plt.figure(figsize=(10,4))
    plt.plot(f_axis_pass/1e6, Pxx_pass_dB)
    plt.xlim([-bw_MHz, bw_MHz])
    plt.title("Passband PSD (zoom around OFDM band)")
    plt.xlabel("Frequency (MHz offset)")
    plt.ylabel("PSD (dB/Hz)")
    plt.grid(True)

    plt.show()

def build_ofdm_symbol(Xk):
    X = np.zeros(Nfft_time, dtype=complex)
    start = Nfft_time // 2 - K // 2
    X[start:start + K] = Xk

    x = ifft(fftshift(X))
    x /= np.sqrt(np.mean(np.abs(x)**2))

    TX_power_W = 10 ** ((TX_power_dBm - 30) / 10)
    x *= np.sqrt(TX_power_W)

    return x


# ------------------------------------------------------------
# 2) Noise model
# ------------------------------------------------------------
def add_noise(signal, B, NF_dB=6, T0=290):
    kB = 1.380649e-23
    NF = 10 ** (NF_dB / 10)
    noise_power = kB * T0 * B * NF
    sigma = np.sqrt(noise_power / 2)

    noise = sigma * (
        np.random.randn(*signal.shape)
        + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


# ------------------------------------------------------------
# 3) OFDM Radar Channel Simulation
# ------------------------------------------------------------
def simulate_ofdm_radar_fixed(d_tot, Fs_slow):
    """
    A correct OFDM radar simulation pipeline:
    - pilot symbol
    - per-pulse channel H[n, k]
    - noise
    - range FFT
    - strongest bin selection
    - slow-time phase extraction
    
    """
    N = len(d_tot)
    tau = 2 * d_tot / c

    # Subcarrier frequency offsets (BASEBAND!)
    delta_f = b / K
    f_offsets = delta_f * (np.arange(K) - K//2)

    # Generate pilot
    pilot = generate_pilot_symbol()
    TX = np.tile(pilot, (N, 1))

    # Build channel matrix
    tau_2d = tau[:, None]
    f_2d = f_offsets[None, :]

    # Phase model: baseband-equivalent
    phase = -2 * np.pi * f_2d * tau_2d
    H = np.exp(1j * phase)

    # Received subcarriers
    RX = TX * H

    # Add noise
    RX_noisy = add_noise(RX, b)

    # Range FFT (fast-time)
    H_range = ifft(RX_noisy, axis=1)
    h_mag = np.abs(H_range)

    # strongest range bin
    r_bin = np.argmax(np.mean(h_mag, axis=0))

    # slow-time signal at strongest bin
    h_slow = H_range[:, r_bin]

    t_slow = np.arange(N) / Fs_slow

    # unwrap phase
    phase_slow = np.unwrap(np.angle(h_slow))

    # detrend
    p = np.polyfit(t_slow, phase_slow, 1)
    phase_detr = phase_slow - np.polyval(p, t_slow)

    # respiration band (0.1–0.5 Hz)
    phase_resp = bp_filter(phase_detr, Fs_slow, 0.1, 0.5)

    # heart band (0.8–2.0 Hz)
    phase_heart = bp_filter(phase_detr, Fs_slow, 0.8, 2.0)

    return (
        h_slow,
        r_bin,
        phase_detr,
        phase_resp,
        phase_heart,
        t_slow
    )
