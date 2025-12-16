import numpy as np
import matplotlib.pyplot as plt
from src.config import c, K, Delta_f


def plot_range_profile(avg_profile, r_bin):
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

    # ----- Plot -----
    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(dist_axis, prof_norm, label="Profil Mocy", linewidth=2)
    ax1.set_xlabel("Dystans [m]")
    ax1.set_ylabel("Znormalizowana Amplituda")
    ax1.set_xlim(0, 2 * dist_detected)
    ax1.grid(True)

    # Vertical dashed red line
    ax1.axvline(dist_detected, color='red', linestyle='--',
                label=f"Dystans: {dist_detected:.2f} m")

    # Red marker at the peak
    ax1.plot(dist_detected, prof_norm[r_bin], 'ro', markersize=10)

    # ----- Add top x-axis (time of flight) -----
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(dist_axis[::K//6])   # 6 labels
    ax2.set_xticklabels([f"{t:.1f}" for t in tof_ns[::K//6]])
    ax2.set_xlabel("Czas Przelotu [ns]")

    # Legend
    ax1.legend(loc="upper right")

    plt.title("Zmierzony Dystans do Pojedynczej Osoby")
    plt.tight_layout()
    plt.xlim(0, dist_detected*2)
    plt.show()

    print(f"Zmierzony dystans: {dist_detected:.3f} m")
    print(f"Zmierzony czas przelotu    : {tof_detected:.2f} ns")