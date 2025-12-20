from src.ml.inference_isoforest_OLD import detect_anomalies_from_radar_file
from src.visualisation.plot_received_anomalies import analyze_and_plot_received
from src.config import freqs, Fs_slow


if __name__ == "__main__":
    test_file = "data/irregular/IR001.mat"

    irregular_regions, anomaly_flags, scores, phase_detr, t_slow = detect_anomalies_from_radar_file(test_file)

    print("anomaly flags", anomaly_flags)
    # print("anomaly scores", scores)


    analyze_and_plot_received(phase_detr, t_slow, irregular_regions)

