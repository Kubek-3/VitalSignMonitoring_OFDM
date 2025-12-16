import numpy as np
from src.signal_processing.chest_motion import load_chest_motion, upsample_signal, chest_displacement
from src.signal_processing.new_radar_channel import simulate_ofdm_radar_end_to_end
from src.config import first_samples, data_fs, ups_factor, xh, yh, xtx, ytx, xrx, yrx, Fs_slow, cf
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
    disp = load_chest_motion(path)[:first_samples]
    disp = disp - np.mean(disp)

    # 2. Upsample to match simulation density
    disp_ups = upsample_signal(len(disp), disp, data_fs, ups_factor)

    # 3. Compute total radar path distance
    d_tot = chest_displacement(disp_ups, xh, yh, xtx, ytx, xrx, yrx)

    original_time = np.arange(len(disp_ups)) / (data_fs * ups_factor)
    phase = compute_phase(d_tot, cf)
    wrapped_phase = np.angle(np.exp(1j*phase))  # wrap to [-pi, pi]
    PL_dB = free_space_path_loss(d_tot, cf)                             # path loss due to chest motion
    pw_r_dBm = pw_recvd_dBm(PL_dB)                               # received power in dBm
    pw_r_w = pw_recvd_w(pw_r_dBm)                                       # received power in watts

    plot_first_graphs(original_time, disp, wrapped_phase, pw_r_w)

    # 5. Simulate OFDM radar
    h_slow, avg_profile, r_bin, phase_resp, phase_heart, p_coeff, t_slow = simulate_ofdm_radar_end_to_end(d_tot, Fs_slow)

    plot_range_profile(avg_profile, r_bin)

    # 6. Extract and detrend phase
    phase = np.unwrap(np.angle(h_slow))
    p = np.polyfit(t_slow, phase, 1)
    phase_detr = phase - np.polyval(p, t_slow)

    plot_breathing_spectrum(phase_detr, Fs_slow)

    return phase_detr, t_slow
