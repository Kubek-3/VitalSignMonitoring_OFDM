from src.ml.inference_isoforest_OLD import detect_anomalies_from_radar_file
from src.visualisation.plot_received_anomalies import analyze_and_plot_received
from src.config import freqs, FS_SLOW, WINDOW_SEC, STEP_SEC, DATA_FS
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def csv_labels_from_dataframe(csv_df):
    """
    Returns ground-truth labels aligned by window index.
    """
    return csv_df["flag"].astype(int).values


def evaluate_file_ml_vs_csv(
    anomaly_flags,
    csv_df,
    STEP_SEC,
    WINDOW_SEC
):
    y_true = []
    y_pred = []

    for i, flag in enumerate(anomaly_flags):
        t_start = i * STEP_SEC
        t_end   = t_start + WINDOW_SEC

        gt = csv_labels_from_dataframe(csv_df)

        y_true.append(gt)
        y_pred.append(int(flag))

    return np.array(y_true), np.array(y_pred)


def load_csv_labels(csv_path):
    """
    Load window-level labels from CSV.
    Expected columns:
      start_time, end_time, flag
    """
    df = pd.read_csv(csv_path)
    return df

def csv_to_window_labels(csv_df, file_name, n_windows):
    """
    Returns y_true array aligned with ML windows
    """
    df_file = csv_df[csv_df["file"] == file_name].reset_index(drop=True)

    if len(df_file) < n_windows:
        print(f"CSV has fewer windows than ML for {file_name}")

    y_true = (
        df_file["anomaly_label"]
        .astype(str)
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0})
        .values
    )

    # Safety crop
    return y_true[:n_windows]

def plot_and_save_confusion_matrix(cm, folder, out_dir):
    """
    Plot and save confusion matrix as PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"Confusion Matrix – {folder}")

    out_path = os.path.join(out_dir, f"confusion_matrix_{folder}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix → {out_path}")


all_y_true = []
all_y_pred = []

models = [
    'isoforest',
    'elliptic',
    'knn'
    ]

if __name__ == "__main__":
    input_folders = os.listdir("data/")
    
    for model in models:
        print(f"\n=== Evaluating model: {model} ===\n")
        model_path = f"models/{model}.pkl"
        for folder in input_folders[0:3]:  
            files = sorted([f for f in os.listdir("data/" + folder) if f.endswith(".mat")])
            all_y_true = []
            all_y_pred = []
            for f in files:
                test_file = os.path.join("data/", folder, f)
                #print("Processing file:", test_file)

                # --- ML detection ---
                irregular_regions, anomaly_flags, scores, phase_detr, t_slow = \
                    detect_anomalies_from_radar_file(test_file, model_path)

                # --- Load CSV reference ---
                csv_path = f"results/baseline/short_window/{folder}/BR_HR_results_window_15_{folder}_labeled.csv"
                csv_df = pd.read_csv(csv_path)

                anomaly_flags = anomaly_flags.astype(int)
                n_windows = len(anomaly_flags)

                # --- CSV ground truth ---
                y_true = csv_to_window_labels(csv_df, f, n_windows)

                # --- Accumulate ---
                all_y_true.extend(y_true)
                all_y_pred.extend(anomaly_flags)

                analyze_and_plot_received(
                    phase_detr,
                    t_slow,
                    irregular_regions,
                    f,
                    f"results/ML/" + model + f"/normal",
                )
    
            cm = confusion_matrix(all_y_true, all_y_pred)
            print("Confusion matrix:")
            print(cm)
            print("number of Anomalies detected for model", model, " for folder", folder, ":\n",  np.sum(all_y_pred))
            print("\nClassification report:")
            print(classification_report(all_y_true, all_y_pred, digits=3))
            plot_and_save_confusion_matrix(cm, folder, f"results/ML/{model}")

    # --- FINAL CONFUSION MATRIX ---
    # cm = confusion_matrix(all_y_true, all_y_pred)
    # print("Confusion matrix:")
    # print(cm)

    # print("\nClassification report:")
    # print(classification_report(all_y_true, all_y_pred, digits=3))
