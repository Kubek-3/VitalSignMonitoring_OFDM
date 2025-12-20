import os
import numpy as np
import matplotlib.pyplot as plt
import glob

from src.signal_processing.new_radar_channel_2_persons import simulate_ofdm_radar_end_to_end_2
from src.signal_processing.ofdm_radar_fixed import simulate_ofdm_radar_fixed
from src.visualisation.plot_first_graphs import plot_first_graphs

from .config import (
    xh, yh, xtx, ytx, xrx, yrx,
    xh_2, yh_2,
    xh_3, yh_3,
    data_fs, ups_factor, Fs_slow,
    cf,
    resp_low, resp_high,
    first_samples,
    FILE_PATHS
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
from .old_ml.radar_channel import radar_channel
from .signal_processing.new_radar_channel import simulate_ofdm_radar_end_to_end
from .signal_processing.radar_model import compute_phase, free_space_path_loss, pw_recvd_dBm, pw_recvd_w
from .visualisation.range_profile import plot_range_profile
from src.visualisation.plot_breathing_spectrum import plot_breathing_spectrum

# -----------------------------
# MAIN PIPELINE
# -----------------------------
# USER SETTINGS

input_folder = "data/post_exercise"
output_csv = "BR_HR_results_post_exercise.csv"
output_folder = "results/post_exercise/"


def main():

    bed_dist_tx = dist(xh, yh, xtx, ytx)
    bed_dist_rx = dist(xh, yh, xrx, yrx)
    bed_tot = bed_dist_tx + bed_dist_rx
    print("Total distance to bed (m):", bed_tot)

    files = sorted(glob.glob(os.path.join(input_folder, "*.mat")))

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")

        #print(f"✅ Loading chest motion from file {os.path.basename(filepath)}")
        disp = load_chest_motion(filepath)
        #disp2 = load_chest_motion(FILE_PATHS[1])
        #disp3 = load_chest_motion(FILE_PATHS[2])
        #min_len = min(len(disp), len(disp2), len(disp3))
        #disp = disp[:min_len]  # make same length
        #disp2 = disp2[:min_len]  # make same length
        #disp3 = disp3[:min_len]  # make same length

        #print("✅ Upsampling...")
        disp_m = upsample_signal(first_samples, disp, data_fs, ups_factor)
        #disp2_m = upsample_signal(first_samples, disp2, data_fs, ups_factor)
        #disp3_m = upsample_signal(first_samples, disp3, data_fs, ups_factor)

        #original_time = np.arange(len(disp_m)) / data_fs / ups_factor        # time in seconds
        
        #print("✅ Computing chest displacement...")
        d_tot = chest_displacement(disp_m, xh, yh, xtx, ytx, xrx, yrx)
        #d_tot2 = chest_displacement(disp2_m, xh_2, yh_2, xtx, ytx, xrx, yrx)
        #d_tot3 = chest_displacement(disp3_m, xh_3, yh_3, xtx, ytx, xrx, yrx)

        # print("✅ Computing phase...")
        # phase = compute_phase(d_tot, cf)
        # wrapped_phase = np.angle(np.exp(1j*phase))  # wrap to [-pi, pi]

        # print("✅ Radar channel simulation...")
        # PL_dB = free_space_path_loss(d_tot, cf)                             # path loss due to chest motion
        # pw_r_dBm = pw_recvd_dBm(PL_dB)                               # received power in dBm
        # pw_r_w = pw_recvd_w(pw_r_dBm)                                       # received power in watts
        # amp = np.sqrt(pw_r_w)                                               # received signal amplitude [sqrt(W)]

        # plot_first_graphs(original_time, disp_m, wrapped_phase, pw_r_w, amp)

        h_slow, avg_profile, r_bin, phase_resp, phase_heart, p_coeff, t_slow  = simulate_ofdm_radar_end_to_end(d_tot, Fs_slow, filename, output_folder)   # radar channel slow-time signal

        # (
        #     h1,
        #     h2,
        #     #h3,
        #     avg_profile,
        #     r_bin,
        #     phase_resp_1,
        #     phase_heart_1,
        #     phase_resp_2,
        #     phase_heart_2,
        #     #phase_resp_3,
        #     #phase_heart_3,
        #     p_coeff_1,
        #     p_coeff_2,
        #     #p_coeff_3,
        #     t_slow
        # ) = simulate_ofdm_radar_end_to_end_2(d_tot, d_tot2, d_tot3, Fs_slow)   # radar channel slow-time signal
        
        plot_range_profile(avg_profile, r_bin, filename, output_folder)

        phase_slow = np.unwrap(np.angle(h_slow))
        phase_detr = phase_slow - np.polyval(p_coeff, t_slow)

    # phase1 = np.unwrap(np.angle(h1))
    # phase_detr_1 = phase1 - np.polyval(p_coeff_1, t_slow)

    # phase2 = np.unwrap(np.angle(h2))
    # phase_detr_2 = phase2 - np.polyval(p_coeff_2, t_slow)  

        #phase3 = np.unwrap(np.angle(h3))
        #phase_detr_3 = phase3 - np.polyval(p_coeff_3, t_slow)

        #print("✅ Plotting breathing spectrum...")
        plot_breathing_spectrum(phase_detr, Fs_slow, filename, output_folder)
        #plot_breathing_spectrum(phase_detr_1, Fs_slow)
        #plot_breathing_spectrum(phase_detr_2, Fs_slow)
        #plot_breathing_spectrum(phase_detr_3, Fs_slow)

        #plot_breathing_spectrum(phase_detr, Fs_slow)

        #plot_range_profile(avg_profile, r_bin)

# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    main()
