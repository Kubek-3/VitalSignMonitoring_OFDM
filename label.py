import pandas as pd
import os


input_folders = os.listdir("data/")
for folder in input_folders:
    # Load original CSV
    input_file = f"results/baseline/{folder}/BR_HR_results_window_15_{folder}.csv"
    df = pd.read_csv(input_file)

    # Apply the rule
    # 10 < BR_bpm < 22  → FALSE
    # otherwise        → TRUE
    df["anomaly_label"] = ~df["BR_bpm"].between(10, 22)

    # Optional: use TRUE/FALSE instead of True/False
    df["anomaly_label"] = df["anomaly_label"].map({True: "TRUE", False: "FALSE"})

    # Save to a NEW file
    output_file = f"results/baseline/{folder}/BR_HR_results_window_15_{folder}_labeled.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved labeled file to: {output_file}")
