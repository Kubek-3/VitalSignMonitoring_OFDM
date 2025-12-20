import os
import numpy as np
import matplotlib.pyplot as plt
from src.config import c, K, Delta_f


def plot_range_profile(avg_profile, r_bin, filename, output_folder):
    """
    Plot range profile with detected person, matching style of Fig. 14.
    avg_profile : 1D array (K,) - averaged magnitude over slow-time
    r_bin       : strongest range bin
    K           : number of subcarriers
    delta_f     : subcarrier spacing
    """
    
    # ----- Compute distance axis -----
    # Standard OFDM radar: R = c / (2B) * k, where B = K * delta_f
    B = K * Delta_f
    dist_axis = c / (2 * B) * np.arange(K)

    # ----- Compute time-of-flight axis for top x-axis -----
    tof_ns = (2 * dist_axis / c) * 1e9   # round-trip ToF [ns]

    # ----- Normalize profile -----
    prof_norm = avg_profile / np.max(avg_profile)

    # ----- Distance of strongest peak -----
    dist_detected = dist_axis[r_bin]
    tof_detected = tof_ns[r_bin]
    # dist_detected2 = dist_axis[r_bin[1]]
    # tof_detected2 = tof_ns[r_bin[1]]
    #dist_detected3 = dist_axis[r_bin[2]]
    #tof_detected3 = tof_ns[r_bin[2]]

    # ----- Plot -----
    fig, ax1 = plt.subplots(figsize=(10,5))

    fig.suptitle("Measured Distance to a Single Person - " + filename)

    ax1.plot(dist_axis, prof_norm, label="Power Profile", linewidth=2)
    ax1.set_xlabel("Distance[m]")
    ax1.set_ylabel("Normalized Amplitude")
    ax1.set_xlim(0, 2 * dist_detected)
    ax1.grid(True)

    # Vertical dashed red line
    ax1.axvline(dist_detected, color='red', linestyle='--',
                label=f"Distance: {dist_detected:.2f} m")
    # ax1.axvline(dist_detected2, color='orange', linestyle='--',
    #             label=f"Distance 2: {dist_detected2:.2f} m")
   # ax1.axvline(dist_detected3, color='green', linestyle='--',
   #             label=f"Distance 3: {dist_detected3:.2f} m")

    # Red marker at the peak
    ax1.plot(dist_detected, prof_norm[r_bin], 'ro', markersize=10)
    #ax1.plot(dist_detected2, prof_norm[r_bin[1]], 'o', color='orange', markersize=10)
    #ax1.plot(dist_detected3, prof_norm[r_bin[2]], 'o', color='green', markersize=10)

    # ----- Add top x-axis (time of flight) -----
        # ---- Time-of-Flight axis ----
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    dist_ticks = ax1.get_xticks()
    tof_ticks_ns = (2 * dist_ticks / c) * 1e9

    ax2.set_xticks(dist_ticks)
    ax2.set_xticklabels([f"{t:.2f}" for t in tof_ticks_ns])
    ax2.set_xlabel("Time of Flight [ns]")

    # Legend
    ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.xlim(0, dist_detected*2)
    out_png = os.path.join(output_folder, filename.replace(".mat", "_range_profile_single_person.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    #print(f"Measured distance: {dist_detected:.3f} m")
    #print(f"Measured distance 2: {dist_detected2:.3f} m")
    #print(f"Measured distance 3: {dist_detected3:.3f} m")
    #print(f"Measured time of flight: {tof_detected:.2f} ns")
    #print(f"Measured time of flight 2: {tof_detected2:.2f} ns")
    #print(f"Measured time of flight 3: {tof_detected3:.2f} ns")