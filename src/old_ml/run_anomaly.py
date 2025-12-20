import argparse
from src.ml.inference_isoforest_OLD import detect_anomalies_from_radar_file

def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection on a radar .mat file")
    parser.add_argument("filepath", type=str, help="Path to .mat file")
    args = parser.parse_args()

    result = detect_anomalies_from_radar_file(args.filepath)

    if result is None:
        print("No features extracted. File too short or invalid.")
        return

    print("\n================= ANOMALY DETECTION =================")
    print("File:", result["file"])
    print("Number of anomaly windows:", result["num_anomalies"])
    print("Anomaly windows (bool array):")
    print(result["anomaly_windows"])
    print("Window times (seconds):")
    print(result["time_centers"])
    print("======================================================\n")

if __name__ == "__main__":
    main()
