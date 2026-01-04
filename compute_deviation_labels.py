import pandas as pd
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Configuration
# =========================

# =========================
# Configuration
# =========================
NORMAL_CSV = f"results/baseline/short_window/BR_HR_results_window_15_normal.csv"
IRREGULAR_CSV = f"results/baseline/short_window/BR_HR_results_window_15_irregular.csv"
BREATHHOLD_CSV = f"results/baseline/short_window/BR_HR_results_window_15_breath_hold.csv"

OUTPUT_NORMAL = f"results/baseline/short_window/BR_HR_results_window_15_normal_labeled_deviation.csv"
OUTPUT_IRREGULAR = f"results/baseline/short_window/BR_HR_results_window_15_irregular_labeled_deviation.csv"
OUTPUT_BREATHHOLD = f"results/baseline/short_window/BR_HR_results_window_15_breath_hold_labeled_deviation.csv"
DEVIATION_THRESHOLD = 0.20  # ±20%
TAIL_FRACTION = 0.0001       # 20% tails for Gaussian labeling

# =========================
# Helper function
# =========================
def estimate_std(df):
    return (df["BR_p2p_max_s"] - df["BR_p2p_min_s"]) / 4.0




# Convert MAD to robust sigma

# =========================
# 1. Load NORMAL data and compute baseline
# =========================
normal_df = pd.read_csv(NORMAL_CSV)

# Per-sample deviation
normal_df["BR_p2p_std_est"] = estimate_std(normal_df)

# # File-level baseline (normal breathing reference)
# baseline_df = (
#     normal_df
#     .groupby("file")["BR_p2p_std_est"]
#     .agg(
#         BR_mu="mean",
#         BR_sigma="std"
#     )
#     .reset_index()
# )

baseline_df = (
    normal_df
    .groupby("file")
    .agg(
        BR_var_q_low=("BR_p2p_std_est", lambda x: np.quantile(x, 0.05)),
        BR_var_q_high=("BR_p2p_std_est", lambda x: np.quantile(x, 0.95)),
        BR_amp_q_low=("BR_amp", lambda x: np.quantile(x, 0.05)),
        BR_amp_q_high=("BR_amp", lambda x: np.quantile(x, 0.95)),
    )
    .reset_index()
)


def label_with_quantile_baseline(df, baseline_df):
    df = df.copy()

    # Deviation feature
    df["BR_p2p_std_est"] = (
        df["BR_p2p_max_s"] - df["BR_p2p_min_s"]
    ) / 4.0

    # Merge baseline
    df = df.merge(baseline_df, on="file", how="left")

    # Individual anomaly flags
    df["BR_var_anomaly"] = ~df["BR_p2p_std_est"].between(
        df["BR_var_q_low"], df["BR_var_q_high"]
    )

    df["BR_amp_anomaly"] = ~df["BR_amp"].between(
        df["BR_amp_q_low"], df["BR_amp_q_high"]
    )

    # Final anomaly label (logical AND)
    df["anomaly_label"] = (
        df["BR_var_anomaly"] & df["BR_amp_anomaly"]
    )

    return df

# =========================
# 3. Load and label IRREGULAR data
# =========================
irregular_df = pd.read_csv(IRREGULAR_CSV)
irregular_labeled = label_with_quantile_baseline(irregular_df, baseline_df)
irregular_labeled.to_csv(OUTPUT_IRREGULAR, index=False)

# =========================
# 4. Load and label BREATHHOLD data
# =========================
breathhold_df = pd.read_csv(BREATHHOLD_CSV)
breathhold_labeled = label_with_quantile_baseline(breathhold_df, baseline_df)
breathhold_labeled.to_csv(OUTPUT_BREATHHOLD, index=False)

normal_df__ = pd.read_csv(NORMAL_CSV)
normal_labeled = label_with_quantile_baseline(normal_df__, baseline_df)
normal_labeled.to_csv(OUTPUT_NORMAL, index=False)


plt.figure()

plt.hist(
    normal_df["BR_p2p_std_est"],
    bins=30,
    density=True,
    alpha=0.6,
    label="Normal"
)

plt.axvline(
    baseline_df["BR_var_q_low"].mean(),
    linestyle="-.",
    label="5% quantile"
)
plt.axvline(
    baseline_df["BR_var_q_high"].mean(),
    linestyle="--",
    label="95% quantile"
)

plt.xlabel("Estimated BR Peak-to-Peak Std [s]")
plt.ylabel("Probability Density")
plt.title("Quantile-Based Baseline for Normal Breathing Variability")
plt.legend()
plt.show()


# =========================
# Done
# =========================
print("Labeling complete.")
print(f"Saved: {OUTPUT_IRREGULAR}")
print(f"Saved: {OUTPUT_BREATHHOLD}")
print(f"Saved: {OUTPUT_NORMAL}")


