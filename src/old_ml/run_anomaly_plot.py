import argparse
import os
import joblib
import numpy as np

from src.ml.inference_isoforest_OLD import detect_anomalies_from_radar_file
from src.signal_processing.phase_for_ml import extract_phase_from_radar_file
 
from src.config import cf, Fs_slow
from src.visualisation.plot_received_anomalies_OLD import analyze_and_plot_received


def main():

    parser = argparse.ArgumentParser(description="Run anomaly detection + plot for radar .mat file")
    parser.add_argument("filepath", type=str, help="Path to .mat radar file")
    args = parser.parse_args()

    file = args.filepath
    if not os.path.exists(file):
        print(f"ERROR: File not found: {file}")
        return

    print("\n================= ANOMALY DETECTION (PLOT) =================")
    print("File:", file)

    # ---------- 1. Run Isolation Forest ----------
    try:
        result = detect_anomalies_from_radar_file(file)
        if result is None:
            print("ERROR: No features extracted from this file.")
            return
    except Exception as e:
        print("ERROR during anomaly detection:", e)
        return

    print("Number of anomaly windows:", result["num_anomalies"])
    print("Anomaly array:", result["anomaly_windows"])
    print("Window centers:", result["time_centers"])

    # ---------- 2. Extract phase for visualization ----------
    try:
        phase_detr, t_slow = extract_phase_from_radar_file(file)
    except Exception as e:
        print("ERROR extracting phase:", e)
        return

    # ---------- 3. Plot using the visualization module ----------
    print("\nGenerating anomaly plot...")
    analyze_and_plot_received(
        h_slow=phase_detr,               # We use phase directly
        t_slow=t_slow,
        show=True,                 # Display interactive plot
    )

    print("\n===================== DONE =====================\n")


if __name__ == "__main__":
    main()
