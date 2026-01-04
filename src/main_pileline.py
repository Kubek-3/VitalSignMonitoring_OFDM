import os
import numpy as np
import matplotlib.pyplot as plt
import glob

from src.signal_processing.new_radar_channel_2_persons import simulate_ofdm_radar_end_to_end_2
from src.signal_processing.ofdm_radar_fixed import simulate_ofdm_radar_fixed
from src.visualisation.plot_first_graphs import plot_first_graphs
from src.visualisation.range_profile_two_persosn import plot_range_profile_two

from .config import (
    xh, yh, xtx, ytx, xrx, yrx,
    xh_2, yh_2,
    xh_3, yh_3,
    x_chair, y_chair,
    DATA_FS, ups_factor, FS_SLOW,
    cf,
    RESP_LOW, RESP_HIGH,
    FIRST_SAMPLES,
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

input_folder = "data/normal/"
#output_csv = "BR_HR_results_post_exercise.csv"
output_folder = "results/new/two_persons/"


def main():

    bed_dist_tx = dist(xh, yh, xtx, ytx)
    bed_dist_rx = dist(xh, yh, xrx, yrx)
    bed_tot = bed_dist_tx + bed_dist_rx
    bed_tot_2 = dist(xh_2, yh_2, xtx, ytx) + dist(xh_2, yh_2, xrx, yrx)
    chair_dist_tx = dist(x_chair, y_chair, xtx, ytx)
    chair_dist_rx = dist(x_chair, y_chair, xrx, yrx)
    chair_tot = chair_dist_tx + chair_dist_rx
    print("Total distance to bed (m):", bed_tot/2)
    print("Total distance to second bed (m):", bed_tot_2/2)
    print("Total distance to chair (m):", chair_tot/2)

    files = sorted(glob.glob(os.path.join(input_folder, "*.mat")))

    for filepath in files[38:39]:  # process only one file for testing
        filename = os.path.basename(filepath)
        filenumber2 = int(filename.rstrip(".mat")[-3:]) + 1
        filenumber3 = filenumber2 + 1
        if filenumber3 == 6:
            filenumber3 = 7
        if filenumber2 == 6:
            filenumber2 = 7
        filetype2 = filename.rstrip(".mat")[:-3]
        filepath2 = filepath.replace(filename, f"{filetype2}{filenumber2:03d}.mat")
        filepath3 = filepath.replace(filename, f"{filetype2}{filenumber3:03d}.mat")
        filename2 = os.path.basename(filepath2)
        filename3 = os.path.basename(filepath3)
        print(f"Processing: {filename}, {filename2}, {filename3}")

        #print(f"✅ Loading chest motion from file {os.path.basename(filepath)}")
        disp = load_chest_motion(filepath)
        disp2 = load_chest_motion(filepath2)
        disp3 = load_chest_motion(filepath3)
        min_len = min(len(disp), len(disp2), len(disp3))
        disp = disp[:min_len]  # make same length
        disp2 = disp2[:min_len]  # make same length
        disp3 = disp3[:min_len]  # make same length

        original_time = np.arange(len(disp)) / DATA_FS  # time in seconds   

        disp_mm = disp * 10000  # convert to mm for plotting
        plt.plot(original_time, disp_mm)
        plt.xlabel("Time [s]")
        plt.ylabel("Chest Displacement [mm]")
        plt.title("Original Chest Displacement Signal")
        out_png = os.path.join(output_folder, filename.replace(".mat", "_1_original_chest_displacement.png"))
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        arr = np.ones_like(disp)
        chair_dist_tx = dist(x_chair, y_chair, xtx, ytx)
        chair_dist_rx = dist(x_chair, y_chair, xrx, yrx)
        chair_tot = arr * (chair_dist_tx + chair_dist_rx)

        #print("✅ Upsampling...")
        disp_m = upsample_signal(FIRST_SAMPLES, disp, DATA_FS, ups_factor)
        disp2_m = upsample_signal(FIRST_SAMPLES, disp2, DATA_FS, ups_factor)
        disp3_m = upsample_signal(FIRST_SAMPLES, disp3, DATA_FS, ups_factor)

        #original_time = np.arange(len(disp_m)) / DATA_FS / ups_factor        # time in seconds
        
        #print("✅ Computing chest displacement...")
        d_tot = chest_displacement(disp_m, xh, yh, xtx, ytx, xrx, yrx)
        d_tot2 = chest_displacement(disp2_m, xh_2, yh_2, xtx, ytx, xrx, yrx)
        d_tot3 = chest_displacement(disp3_m, xh_3, yh_3, xtx, ytx, xrx, yrx)

        # print("✅ Computing phase...")
        # phase = compute_phase(d_tot, cf)
        # wrapped_phase = np.angle(np.exp(1j*phase))  # wrap to [-pi, pi]

        # print("✅ Radar channel simulation...")
        # PL_dB = free_space_path_loss(d_tot, cf)                             # path loss due to chest motion
        # pw_r_dBm = pw_recvd_dBm(PL_dB)                               # received power in dBm
        # pw_r_w = pw_recvd_w(pw_r_dBm)                                       # received power in watts
        # amp = np.sqrt(pw_r_w)                                               # received signal amplitude [sqrt(W)]

        # plot_first_graphs(original_time, disp_m, wrapped_phase, pw_r_w, amp)

        #h_slow, avg_profile, r_bin, phase_detr, phase_resp, phase_heart, p_coeff, t_slow  = simulate_ofdm_radar_end_to_end(d_tot, FS_SLOW, filename, output_folder)   # radar channel slow-time signal

        simulate_ofdm_radar_end_to_end_2(d_tot, d_tot2, d_tot3, chair_tot, FS_SLOW, filename, filename2, filename3, output_folder)   # radar channel slow-time signal
        
        # plot_range_profile(avg_profile, r_bin, filename, output_folder)
        # phase_slow = np.unwrap(np.angle(h_slow))
        # phase_detr = phase_slow - np.polyval(p_coeff, t_slow)

        #plot_breathing_spectrum(phase_detr, FS_SLOW, filename, output_folder)

        

# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    main()
