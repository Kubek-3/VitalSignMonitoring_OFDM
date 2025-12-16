import numpy as np
import matplotlib.pyplot as plt

def plot_breathing_spectrum(phase_signal, fs, fmax=1.3, label="Phase FFT"):
    """
    Plot frequency spectrum of respiratory phase signal with harmonics.
    Matches style of Fig. 17 (normalized amplitude, dashed harmonic lines).
    """
    N = len(phase_signal)
    
    # FFT
    win = np.hanning(N)
    spec = np.fft.rfft(phase_signal * win)
    freqs_fft = np.fft.rfftfreq(N, 1/fs)
    
    # Limit frequency axis
    mask = freqs_fft <= fmax
    freqs_use = freqs_fft[mask]
    spec_use = np.abs(spec[mask])
    
    # Normalize amplitude
    spec_norm = spec_use / np.max(spec_use)
    
    # Find main breathing peak automatically within 0.1-0.5 Hz
    br_mask = (freqs_fft >= 0.1) & (freqs_fft <= 0.5)
    br_idx = np.argmax(np.abs(spec[br_mask]))
    f_br = freqs_fft[br_mask][br_idx]

    # Find main breathing peak automatically within 0.1-0.5 Hz
    hr_mask = (freqs_fft >= 0.8) & (freqs_fft <= 2.0)
    hr_idx = np.argmax(np.abs(spec[hr_mask]))
    f_hr = freqs_fft[hr_mask][hr_idx]

    print("Estimated breathing frequency:", f_br)
    print("Estimated breathing rate:", 60*f_br, "bpm")
    print("Estimated heart frequency:", f_hr)
    print("Estimated heart rate:", 60*f_hr, "bpm")
    
    # Harmonics
    harmonics = [f_br, 2*f_br, 3*f_br]
    
    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(freqs_use, spec_norm, label="Filtered FFT", linewidth=2)
    
    # Vertical dashed lines for harmonics
    for h in harmonics:
        if h <= fmax:
            plt.axvline(h, color='red', linestyle='--', alpha=0.8,
                        label=f"Harmonic {h:.2f} Hz")
    
    plt.xlim(0, fmax)
    plt.ylim(0, 1.1)
    plt.xlabel("Częstotliwość (Hz)")
    plt.ylabel("Znormalizowana Amplituda")
    plt.title("Składniki Częstotliwościowe Zmian Fazy (Oddychanie i Harmoniczne)")
    plt.grid(True)
    
    # Avoid repeated labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="upper right")
    
    plt.tight_layout()
    plt.show()
    
    return f_br, f_hr