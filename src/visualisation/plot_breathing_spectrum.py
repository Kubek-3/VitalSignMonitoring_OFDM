import numpy as np
import matplotlib.pyplot as plt
import os

def plot_breathing_spectrum(phase_signal, fs, filename, output_folder, fmax=1.3, label="Phase FFT"):
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
    
    fig, (ax_br, ax_hr) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=False
    )
    fig.suptitle("Frequency Spectrum of Phase Variations - " + filename)

    ax_br.plot(freqs_use, spec_norm, label="Filtered FFT", linewidth=2)
    
    # Vertical dashed lines for harmonics
    for h in harmonics:
        if h <= fmax:
            ax_br.axvline(h, color='red', linestyle='--', alpha=0.8,
                        label=f"Harmonic {h:.2f} Hz")
    
    ax_br.set_xlim(0, fmax)
    ax_br.set_ylim(0, 1.1)
    ax_br.set_xlabel("Frequency (Hz)")
    ax_br.set_ylabel("Normalized Amplitude")
    ax_br.set_title("Frequency Components of Phase Variations (Breathing and Harmonics)")
    ax_br.grid(True)
    
    # Avoid repeated labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_br.legend(unique.values(), unique.keys(), loc="upper right")

    # ====================
    # BOTTOM: HR SPECTRUM
    # ====================

    hr_band = (0.8, 2.0)
    hr_offset = 0.3
    hr_mask = (freqs_fft >= hr_band[0] - hr_offset) & (freqs_fft <= hr_band[1] + hr_offset)

    freqs_hr = freqs_fft[hr_mask]
    spec_hr = np.abs(spec[hr_mask])
    spec_hr_norm = spec_hr / np.max(spec_hr)

    ax_hr.plot(freqs_hr, spec_hr_norm, linewidth=2, label="Filtered HR FFT")

    # HR band shading
    ax_hr.axvspan(
        hr_band[0], hr_band[1],
        color="gray", alpha=0.2, label="HR Filter Band"
    )

    # Detected HR line
    ax_hr.axvline(
        f_hr, color="red", linestyle="--", linewidth=2,
        label=f"Estimated HR: {60*f_hr:.1f} bpm"
    )

    ax_hr.set_xlim(hr_band[0] - hr_offset, hr_band[1] + hr_offset)
    ax_hr.set_ylim(0, 1.1)
    ax_hr.set_xlabel("Frequency (Hz)")
    ax_hr.set_ylabel("Normalized Amplitude")
    ax_hr.set_title("Heart Rate Spectrum")
    ax_hr.grid(True)
    ax_hr.legend(loc="upper right")
    
    out_png = os.path.join(output_folder, filename.replace(".mat", "_breath_and_hearth_spectrum.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
    return f_br, f_hr