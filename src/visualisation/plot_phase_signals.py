import matplotlib.pyplot as plt
import numpy as np
import os

def plot_phase_signals(t_slow, phase_detr, phase_resp, phase_heart, filename, output_folder):
        
    plt.figure()
    plt.suptitle("Phase Signals over Slow Time - " + filename)

    plt.subplot(3, 1, 1)
    plt.plot(t_slow, phase_detr)
    plt.title("Phase at target range bin")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t_slow, phase_resp)
    plt.title("Respiration band phase (0.1–0.5 Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t_slow, phase_heart)
    plt.title("Heart band phase (0.8–2.0 Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (rad)")
    plt.grid(True)

    plt.tight_layout()

    out_png = os.path.join(output_folder, filename.replace(".mat", "_phase_of_signals_at_detected_range_bin.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()