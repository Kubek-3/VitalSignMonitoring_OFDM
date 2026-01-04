import numpy as np
from src.signal_processing.chest_motion import load_chest_motion, upsample_signal, chest_displacement
from src.signal_processing.new_radar_for_ml import simulate_ofdm_radar_end_to_end_for_ml
from src.config import DATA_FS, xh, yh, xtx, ytx, xrx, yrx, FS_SLOW, cf
from src.signal_processing.ofdm_radar_fixed import simulate_ofdm_radar_fixed
from src.signal_processing.phase_processing import compute_phase
from src.signal_processing.radar_model import free_space_path_loss, pw_recvd_dBm, pw_recvd_w
from src.visualisation.plot_breathing_spectrum import plot_breathing_spectrum
from src.visualisation.plot_first_graphs import plot_first_graphs
from src.visualisation.range_profile import plot_range_profile


def extract_phase_from_radar_file(path):
    """
    Load .mat chest motion → create target distance → run OFDM simulation → extract stable phase.
    """

    # 1. Load chest motion (displacement)
    disp = load_chest_motion(path)

    # 2. Compute total radar path distance
    d_tot = chest_displacement(disp, xh, yh, xtx, ytx, xrx, yrx)

    original_time = np.arange(len(disp)) / (DATA_FS)
    phase = compute_phase(d_tot, cf)
    wrapped_phase = np.angle(np.exp(1j*phase))  # wrap to [-pi, pi]
    PL_dB = free_space_path_loss(d_tot, cf)                             # path loss due to chest motion
    pw_r_dBm = pw_recvd_dBm(PL_dB)                               # received power in dBm
    pw_r_w = pw_recvd_w(pw_r_dBm)                                       # received power in watts

    #plot_first_graphs(original_time, disp, wrapped_phase, pw_r_w)

    # 5. Simulate OFDM radar
    h_slow, avg_profile, r_bin, phase_detr, phase_resp, phase_heart, p_coeff, t_slow = simulate_ofdm_radar_end_to_end_for_ml(d_tot, FS_SLOW)

    #plot_range_profile(avg_profile, r_bin)

   #plot_breathing_spectrum(phase_detr, Fs_slow)

    return phase_detr, t_slow
