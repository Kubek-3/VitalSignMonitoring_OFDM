import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from src.config import c, Fs_slow, freqs, cf, data_fs, ups_factor


# -----------------------------------------------
# Helpers
# -----------------------------------------------
def bp_filter(sig, fs, low, high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)


def spectral_entropy(mag):
    p = mag / (np.sum(mag) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

def merge_regions(indices, win_s, hop_s, min_dur=3.0):
    """
    Merge boolean window flags into continuous time regions.
    
    indices : array of bool (length = n_windows)
    win_s   : window length in seconds
    hop_s   : hop size in seconds
    min_dur : minimum region duration in seconds
    """
    regions = []
    in_region = False
    start_idx = 0

    for i, flag in enumerate(indices):
        if flag and not in_region:
            in_region = True
            start_idx = i
        elif not flag and in_region:
            end_idx = i - 1
            start_t = start_idx * hop_s
            end_t   = end_idx * hop_s + win_s
            if end_t - start_t >= min_dur:
                regions.append((start_t, end_t))
            in_region = False

    # handle case where region reaches the end
    if in_region:
        end_idx = len(indices) - 1
        start_t = start_idx * hop_s
        end_t   = end_idx * hop_s + win_s
        if end_t - start_t >= min_dur:
            regions.append((start_t, end_t))

    return regions



def phase_to_displacement(phase):
    lam = c / cf
    return (lam / (2 * np.pi)) * phase


# -----------------------------------------------
# MAIN VISUALIZATION FUNCTION
# -----------------------------------------------
def analyze_and_plot_received(
    phase_detr,
    t_slow,
    win_s=10.0,
    step_s=2.0,
    flat_factor=0.25,
    entropy_pct=80,
    show=True,
):
    """
    Converts received radar slow-time signal into displacement, detects anomalies,
    and plots respiration, heart, and displacement with highlighted regions.
    """


    # Convert to displacement
    disp = phase_to_displacement(phase_detr)

    # -------------------------
    # 2. Respiration & heart signals
    # -------------------------
    resp = bp_filter(phase_detr, Fs_slow, 0.1, 0.5)
    heart = bp_filter(phase_detr, Fs_slow, 0.8, 2.5)

    # -------------------------
    # 3. Sliding windows
    # -------------------------
    win = int(win_s * Fs_slow)
    step = int(step_s * Fs_slow)

    centers = []
    p2p_list = []
    entropy_list = []

    for i in range(0, len(disp) - win, step):
        seg = disp[i:i+win]
        centers.append((i + win/2) / Fs_slow)

        # Time-domain flatness (breath-hold)
        p2p = np.ptp(seg)
        p2p_list.append(p2p)

        # Frequency-domain entropy
        spec = np.abs(rfft(seg * np.hanning(len(seg))))
        entropy = spectral_entropy(spec)
        entropy_list.append(entropy)

    centers = np.array(centers)
    p2p_list = np.array(p2p_list)
    entropy_list = np.array(entropy_list)

    # -------------------------
    # 4. Detect anomalies
    # -------------------------
    median_p2p = np.median(p2p_list) + 1e-12
    flat_thresh = flat_factor * median_p2p

    is_flat = p2p_list < flat_thresh
    flat_regions = merge_regions(is_flat, 10, 2, min_dur=3.0)

    entropy_cut = np.percentile(entropy_list, entropy_pct)
    is_irregular = entropy_list > entropy_cut
    irregular_regions = merge_regions(is_irregular, 10, 2, min_dur=3.0)
    original_time = np.arange(len(phase_detr)) / data_fs / ups_factor
    print(f"Original signal duration: {original_time[-1]:.2f} s")
    # -------------------------
    # 5. Plot
    # -------------------------
    if show:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        # ---- 1. Displacement ----
        ax1.plot(original_time, disp * 1000, color="black")
        ax1.set_title("Chest displacement (mm, derived from OFDM radar)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("mm")
        ax1.tick_params(axis="x", which="both", labelbottom=True)
        ax1.grid(True)

        for s, e in flat_regions:
            ax1.axvspan(s, e, color="blue", alpha=0.25, label="Breath-hold")
        for s, e in irregular_regions:
            ax1.axvspan(s, e, color="red", alpha=0.25, label="Irregular BR")

        # Avoid duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(dict(zip(labels, handles)).values(),
                       dict(zip(labels, handles)).keys())

        # ---- 2. Respiration ----
        ax2.plot(original_time, resp, color="tab:orange")
        ax2.set_title("Respiration-band (0.1–0.5 Hz)")
        ax2.set_ylabel("phase (rad)")
        ax2.tick_params(axis="x", which="both", labelbottom=True)
        ax2.grid(True)

        for s, e in irregular_regions:
            ax2.axvspan(s, e, color="red", alpha=0.25)

        # ---- 3. Heart ----
        ax3.plot(original_time, heart, color="tab:green")
        ax3.set_title("Heart-band (0.8–2.5 Hz)")
        ax3.set_ylabel("phase (rad)")
        ax3.set_xlabel("Time (s)")
        ax3.tick_params(axis="x", which="both", labelbottom=True)
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

    # -------------------------
    # 6. Return data
    # -------------------------
    return {
        "t_slow": t_slow,
        "disp": disp,
        "resp": resp,
        "heart": heart,
        "centers": centers,
        "p2p": p2p_list,
        "entropy": entropy_list,
        "flat_regions": flat_regions,
        "irregular_regions": irregular_regions,
    }
