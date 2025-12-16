import os
import numpy as np
import matplotlib.pyplot as plt

from src.signal_processing.ofdm_radar_fixed import simulate_ofdm_radar_fixed
from src.visualisation.plot_first_graphs import plot_first_graphs

from .config import (
    xh, yh, xtx, ytx, xrx, yrx,
    xh_2, yh_2,
    data_fs, ups_factor, Fs_slow,
    cf,
    resp_low, resp_high,
    first_samples
)

from .signal_processing.chest_motion import (
    dist,
    load_chest_motion,
    upsample_signal,
    chest_displacement,
)

from .signal_processing.filters import bp_filter


from .signal_processing.chest_motion import (
    load_chest_motion,
    upsample_signal,
    chest_displacement
)
from .signal_processing.radar_channel import radar_channel
from .signal_processing.new_radar_channel import simulate_ofdm_radar_end_to_end
from .signal_processing.radar_model import compute_phase, free_space_path_loss, pw_recvd_dBm, pw_recvd_w
from .visualisation.range_profile import plot_range_profile
from src.visualisation.plot_breathing_spectrum import plot_breathing_spectrum



# -----------------------------
# LOAD ONE FILE (FOR NOW)
# -----------------------------

FILE_PATH = "data/irregular/IR003.mat"   # change later for batch training
FILE_PATH2 = "data/normal/N011.mat"   # change later for batch training


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():

    bed_dist_tx = dist(xh, yh, xtx, ytx)
    bed_dist_rx = dist(xh, yh, xrx, yrx)
    bed_tot = bed_dist_tx + bed_dist_rx
    print("Total distance to bed (m):", bed_tot)

    print("✅ Loading chest motion...")
    disp = load_chest_motion(FILE_PATH)
    disp2 = load_chest_motion(FILE_PATH2)
    min_len = min(len(disp), len(disp2))
    disp = disp[:min_len]  # make same length
    disp2 = disp2[:min_len]  # make same length

    print("✅ Upsampling...")
    disp_m = upsample_signal(first_samples, disp, data_fs, ups_factor)
    disp2_m = upsample_signal(first_samples, disp2, data_fs, ups_factor)

    original_time = np.arange(len(disp_m)) / data_fs / ups_factor        # time in seconds
    
    print("✅ Computing chest displacement...")
    d_tot = chest_displacement(disp_m, xh, yh, xtx, ytx, xrx, yrx)
    d_tot2 = chest_displacement(disp2_m, xh_2, yh_2, xtx, ytx, xrx, yrx)

    print("✅ Computing phase...")
    phase = compute_phase(d_tot, cf)
    wrapped_phase = np.angle(np.exp(1j*phase))  # wrap to [-pi, pi]

    print("✅ Radar channel simulation...")
    PL_dB = free_space_path_loss(d_tot, cf)                             # path loss due to chest motion
    pw_r_dBm = pw_recvd_dBm(PL_dB)                               # received power in dBm
    pw_r_w = pw_recvd_w(pw_r_dBm)                                       # received power in watts
    amp = np.sqrt(pw_r_w)                                               # received signal amplitude [sqrt(W)]

    plot_first_graphs(original_time, disp_m, wrapped_phase, pw_r_w)

    h_slow, avg_profile, r_bin, phase_resp, phase_heart, p_coeff, t_slow  = simulate_ofdm_radar_end_to_end(d_tot, Fs_slow)   # radar channel slow-time signal
    
    plot_range_profile(avg_profile, r_bin)

    phase_slow = np.unwrap(np.angle(h_slow))
    phase_detr = phase_slow - np.polyval(p_coeff, t_slow)

    plot_breathing_spectrum(phase_detr, Fs_slow)

# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    main()
