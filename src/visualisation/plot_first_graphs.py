import matplotlib.pyplot as plt
import numpy as np

def plot_first_graphs(time, chest_disp, radar_phase, received_power):
    """
    Plot chest displacement and radar phase over time.

    time          : 1D array - time axis in seconds
    chest_disp    : 1D array - chest displacement in meters
    radar_phase   : 1D array - radar phase in radians
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot chest displacement
    ax1.plot(time, chest_disp * 1000, color='blue', linewidth=2)  # convert to mm
    ax1.set_title("Chest Displacement Over Time")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Displacement [mm]")
    ax1.grid(True)

    # Plot received power
    ax2.plot(time, 10 * np.log10(received_power + 1e-12), color='red', linewidth=2)  # convert to dBm
    ax2.set_title("Received Power Over Time")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Received Power [W]")
    ax2.grid(True)


    plt.tight_layout()
    plt.show()