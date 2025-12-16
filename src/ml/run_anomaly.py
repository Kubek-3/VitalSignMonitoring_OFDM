from src.ml.inference_isoforest import detect_anomalies_from_radar_file
from src.visualisation.plot_received_anomalies import analyze_and_plot_received
from src.config import freqs, Fs_slow


if __name__ == "__main__":
    test_file = "data/normal/N002.mat"

    t_mid, anomaly_flags, scores, phase_detr, t_slow = detect_anomalies_from_radar_file(
        test_file,
        window_sec=10.0,
        step_sec=2.0
    )

    analyze_and_plot_received(phase_detr, t_slow)
