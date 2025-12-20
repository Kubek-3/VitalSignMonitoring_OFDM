import matplotlib.pyplot as plt
import numpy as np

def plot_first_graphs(time, chest_disp, radar_phase, received_power, amp):
    """
    Plot chest displacement and radar phase over time.

    time          : 1D array - time axis in seconds
    chest_disp    : 1D array - chest displacement in meters
    radar_phase   : 1D array - radar phase in radians
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Plot chest displacement
    ax1.plot(time, chest_disp * 1000, color='blue', linewidth=2)  # convert to mm
    ax1.set_title("Chest Displacement Over Time (Data from File)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Displacement [mm]")
    ax1.grid(True)

    # Plot radar phase
    ax2.plot(time, radar_phase, color='green', linewidth=2)
    ax2.set_title("Radar Phase Over Time (Wrapped)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Phase [radians]")
    ax2.grid(True)

    # Plot received power
    ax3.plot(time, 10 * np.log10(received_power + 1e-12), color='red', linewidth=2)  # convert to dBm
    ax3.set_title("Received Power Over Time ")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Received Power [W]")
    ax3.grid(True)

    # Plot amplitude
    ax4.plot(time, amp, color='purple', linewidth=2)
    ax4.set_title("Received Signal Amplitude Over Time")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Amplitude [sqrt(W)]")
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()