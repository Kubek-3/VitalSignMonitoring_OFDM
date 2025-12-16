import matplotlib.pyplot as plt
import numpy as np


def analyze_and_plot_received(t, phase, t_mid, anomaly_flags):
    plt.figure(figsize=(12, 4))
    plt.plot(t, phase, label="Radar phase (detrended)", lw=1)

    # Highlight anomaly windows
    for tm, is_anom in zip(t_mid, anomaly_flags):
        if is_anom:
            plt.axvspan(tm - 1.0, tm + 1.0, color="red", alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.title("Radar Phase with Detected Breathing Irregularities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
