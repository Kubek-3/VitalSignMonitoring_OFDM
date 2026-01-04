import matplotlib.pyplot as plt
import os


def plot_avg_profile(avg_profile, filename, output_folder):
    # Plot average range profile (magnitude)
    plt.figure()
    plt.plot(avg_profile)
    plt.xlabel("Range bin index")
    plt.ylabel("Magnitude")
    plt.xlim(0, 200)
    plt.grid(True)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    out_png = os.path.join(output_folder, filename.replace(".mat", "_10_average_profile_range.png"))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()