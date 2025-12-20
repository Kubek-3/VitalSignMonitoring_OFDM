import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from src.config import c, Fs_slow, freqs, cf, data_fs, ups_factor, window_sec, step_sec


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

def merge_regions(indices, win_s, hop_s, min_dur=5.0):
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
# def analyze_and_plot_received(
#     phase_detr,
#     t_slow,
#     flat_regions=None,
#     irregular_regions=None,
#     show=True,
# ):
#     """
#     Plot chest displacement with detected anomaly regions.
#     """

#     disp = phase_to_displacement(phase_detr)

#     if show:
#         fig, ax = plt.subplots(1, 1, figsize=(14, 3), sharex=True)

#         # ---- Signal ----
#         ax.plot(t_slow, disp * 1000, color="black", linewidth=1.2)
#         ax.set_title("Chest displacement (mm, derived from OFDM radar)")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("mm")
#         ax.grid(True)

#         # ---- Flat (breath-hold) regions ----
#         if flat_regions is not None:
#             for s, e in flat_regions:
#                 ax.axvspan(s, e, color="blue", alpha=0.25, label="Breath-hold")

#         # ---- Irregular breathing regions ----
#         if irregular_regions is not None:
#             for s, e in irregular_regions:
#                 ax.axvspan(s, e, color="red", alpha=0.25, label="Irregular BR")

#         # ---- Legend (deduplicated) ----
#         handles, labels = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         ax.legend(by_label.values(), by_label.keys())

#         plt.tight_layout()
#         plt.show()

def analyze_and_plot_received(
    phase_detr,
    t_slow,
    irregular_regions,
    show=True,
):

    """
    Converts received radar slow-time signal into displacement, detects anomalies,
    and plots respiration, heart, and displacement with highlighted regions.
    """


    # Convert to displacement
    disp = phase_to_displacement(phase_detr)
    original_time = np.arange(len(phase_detr)) / data_fs / ups_factor

    # -------------------------
    # 5. Plot
    # -------------------------
    if show:
        fig, (ax1) = plt.subplots(1, 1, figsize=(14, 3), sharex=True)

        # ---- 1. Displacement ----
        ax1.plot(original_time, disp * 100, color="black")
        ax1.set_title("Chest displacement (mm, derived from OFDM radar)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("mm")
        ax1.tick_params(axis="x", which="both", labelbottom=True)
        ax1.grid(True)

        for s, e in irregular_regions:
            ax1.axvspan(s, e, color="red", alpha=0.25, label="Irregular BR")

        # Avoid duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(dict(zip(labels, handles)).values(),
                       dict(zip(labels, handles)).keys())

        # # ---- 2. Respiration ----
        # ax2.plot(original_time, resp, color="tab:orange")
        # ax2.set_title("Respiration-band (0.1–0.5 Hz)")
        # ax2.set_ylabel("phase (rad)")
        # ax2.tick_params(axis="x", which="both", labelbottom=True)
        # ax2.grid(True)

        # for s, e in irregular_regions:
        #     ax2.axvspan(s, e, color="red", alpha=0.25)

        # # ---- 3. Heart ----
        # ax3.plot(original_time, heart, color="tab:green")
        # ax3.set_title("Heart-band (0.8–2.5 Hz)")
        # ax3.set_ylabel("phase (rad)")
        # ax3.set_xlabel("Time (s)")
        # ax3.tick_params(axis="x", which="both", labelbottom=True)
        # ax3.grid(True)

        plt.tight_layout()
        plt.show()