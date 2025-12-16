"""
src/ml/br_hr_ml.py

Train / test / inference utilities to detect irregularities in
Breathing (BR) and Heart (HR) based on phase signals produced by the OFDM radar pipeline.

Usage:
    # Train (uses all files under DATA_NORMAL from config)
    python -m src.ml.br_hr_ml train

    # Test on irregular + breath-hold
    python -m src.ml.br_hr_ml test

    # Run inference on single file
    python -m src.ml.br_hr_ml infer path/to/file.mat
"""

import os
import sys
import numpy as np
from glob import glob
import joblib
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# imports from your pipeline
from src.config import (
    DATA_NORMAL, DATA_IRREG, DATA_HOLD, MODELS_PATH,
    data_fs, ups_factor, Fs_slow, cf,
    resp_low, resp_high, first_samples,
    xh, yh, xtx, ytx, xrx, yrx
)

from src.signal_processing.chest_motion import (
    load_chest_motion, upsample_signal, chest_displacement
)

from src.signal_processing.filters import bp_filter

# -------------- Feature extraction --------------

def spectral_entropy(mag):
    p = mag / (np.sum(mag) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

def detect_peaks_rate(sig, fs, height=None, distance_seconds=0.3):
    """
    Detect peaks in a (bandpassed) phase signal and return instantaneous rate series (Hz)
    distance_seconds: minimum spacing between peaks in seconds
    """
    distance = int(distance_seconds * fs)
    # find_peaks works on amplitude; phase may have positive/negative; use absolute value
    peaks, props = find_peaks(np.abs(sig), height=height, distance=distance)
    if len(peaks) < 2:
        return np.array([]), peaks
    # convert peak indices to times
    times = peaks / fs
    inst_periods = np.diff(times)  # seconds between beats/breaths
    inst_rate_hz = 1.0 / (inst_periods + 1e-12)
    return inst_rate_hz, peaks

def extract_features_from_windows(phase_resp, phase_heart, fs, win_s=10.0, step_s=2.0):
    """
    Slide window and extract features for BR and HR for each window.
    Returns:
      X_br: (N_windows, n_br_features)
      X_hr: (N_windows, n_hr_features)
      t_centers: window centers (s)
    BR features: [rms, p2p, dom_freq, dom_prom, spec_entropy, brv]
    HR features: [rms, p2p, dom_freq, dom_prom, spec_entropy, sdnn, rmssd]
    """
    win = int(win_s * fs)
    step = int(step_s * fs)
    X_br = []
    X_hr = []
    t_centers = []

    N = len(phase_resp)
    for start in range(0, N - win + 1, step):
        seg_r = phase_resp[start:start+win]
        seg_h = phase_heart[start:start+win]

        # ------- Resp features -------
        # time-domain
        rms_r = np.sqrt(np.mean(seg_r**2))
        p2p_r = np.ptp(seg_r)
        # frequency-domain
        win_hann = np.hanning(len(seg_r))
        spec_r = np.abs(rfft(seg_r * win_hann))
        freqs_r = rfftfreq(len(seg_r), 1/fs)
        br_mask = (freqs_r >= 0.05) & (freqs_r <= 0.7)
        if np.any(br_mask):
            mag_br = spec_r[br_mask]
            freqs_br = freqs_r[br_mask]
            idx = np.argmax(mag_br)
            dom_freq_r = freqs_br[idx]
            dom_prom_r = mag_br[idx] / (np.mean(mag_br) + 1e-12)
        else:
            dom_freq_r = 0.0
            dom_prom_r = 1.0
        spec_ent_r = spectral_entropy(spec_r)
        # breathing-rate variability proxy: std of derivative / short-term peak intervals
        diff_r = np.diff(seg_r)
        brv = np.std(diff_r)

        X_br.append([rms_r, p2p_r, dom_freq_r, dom_prom_r, spec_ent_r, brv])

        # ------- Heart features -------
        rms_h = np.sqrt(np.mean(seg_h**2))
        p2p_h = np.ptp(seg_h)
        win_hann_h = np.hanning(len(seg_h))
        spec_h = np.abs(rfft(seg_h * win_hann_h))
        freqs_h = rfftfreq(len(seg_h), 1/fs)
        hr_mask = (freqs_h >= 0.8) & (freqs_h <= 3.0)  # 48-180 bpm
        if np.any(hr_mask):
            mag_hr = spec_h[hr_mask]
            freqs_hr = freqs_h[hr_mask]
            idxh = np.argmax(mag_hr)
            dom_freq_h = freqs_hr[idxh]
            dom_prom_h = mag_hr[idxh] / (np.mean(mag_hr) + 1e-12)
        else:
            dom_freq_h = 0.0
            dom_prom_h = 1.0
        spec_ent_h = spectral_entropy(spec_h)
        # HRV: detect peaks in heart band
        inst_rates, peaks = detect_peaks_rate(seg_h, fs, height=None, distance_seconds=0.25)
        if len(inst_rates) >= 2:
            sdnn = np.std(inst_rates)
            rmssd = np.sqrt(np.mean(np.diff(inst_rates)**2))
        else:
            sdnn = 0.0
            rmssd = 0.0

        X_hr.append([rms_h, p2p_h, dom_freq_h, dom_prom_h, spec_ent_h, sdnn, rmssd])

        # time center
        center = (start + start + win) / 2.0 / fs
        t_centers.append(center)

    return np.array(X_br), np.array(X_hr), np.array(t_centers)


# -------------- Training / saving --------------

def train_models(save=True,
                 br_win=10.0, hr_win=10.0, step=2.0,
                 contamination_br=0.2, contamination_hr=0.2):
    """
    Train one-class IsolationForest separately for BR and HR using DATA_NORMAL.
    Saves: models/br_model.pkl, models/hr_model.pkl, models/scaler_br.pkl, scaler_hr.pkl, rms_ref.pkl
    """
    file_list = sorted(glob(os.path.join(DATA_NORMAL, "*.mat")))
    if len(file_list) == 0:
        raise RuntimeError(f"No normal files found in {DATA_NORMAL}")

    all_br = []
    all_hr = []
    print(f"[train_models] Found {len(file_list)} normal files. Extracting features...")

    for fpath in file_list:
        print(" Processing:", os.path.basename(fpath))
        disp = load_chest_motion(fpath)
        disp = upsample_signal(first_samples, disp, data_fs, ups_factor)
        d_tot = chest_displacement(disp, xh, yh, xtx, ytx, xrx, yrx)
        phi = compute_phase(d_tot, cf)
        phase_resp = bp_filter(phi, Fs_slow, resp_low, resp_high)
        phase_heart = bp_filter(phi, Fs_slow, 0.8, 2.5)

        Xbr, Xhr, _ = extract_features_from_windows(phase_resp, phase_heart, Fs_slow,
                                                    win_s=br_win, step_s=step)
        if Xbr.size > 0:
            all_br.append(Xbr)
        if Xhr.size > 0:
            all_hr.append(Xhr)

    X_br = np.vstack(all_br)
    X_hr = np.vstack(all_hr)
    print("[train_models] Feature matrices shapes:", X_br.shape, X_hr.shape)

    # scalers
    scaler_br = StandardScaler().fit(X_br)
    scaler_hr = StandardScaler().fit(X_hr)
    Xb_s = scaler_br.transform(X_br)
    Xh_s = scaler_hr.transform(X_hr)

    # Isolation Forests
    br_model = IsolationForest(n_estimators=500, contamination=contamination_br, random_state=0,
                               max_samples=0.8)
    hr_model = IsolationForest(n_estimators=500, contamination=contamination_hr, random_state=0,
                               max_samples=0.8)

    br_model.fit(Xb_s)
    hr_model.fit(Xh_s)

    os.makedirs(MODELS_PATH, exist_ok=True)
    if save:
        joblib.dump(br_model, os.path.join(MODELS_PATH, "br_model.pkl"))
        joblib.dump(hr_model, os.path.join(MODELS_PATH, "hr_model.pkl"))
        joblib.dump(scaler_br, os.path.join(MODELS_PATH, "scaler_br.pkl"))
        joblib.dump(scaler_hr, os.path.join(MODELS_PATH, "scaler_hr.pkl"))
        # save RMS reference (median RMS across training windows for breath/heart)
        rms_ref_br = np.median(X_br[:, 0])
        rms_ref_hr = np.median(X_hr[:, 0])
        joblib.dump(rms_ref_br, os.path.join(MODELS_PATH, "rms_ref_br.pkl"))
        joblib.dump(rms_ref_hr, os.path.join(MODELS_PATH, "rms_ref_hr.pkl"))
        print("[train_models] Models and scalers saved under", MODELS_PATH)

    return br_model, hr_model, scaler_br, scaler_hr


# ----------- Test / evaluation --------------

def run_inference_on_file(filepath,
                          br_model=None, hr_model=None,
                          scaler_br=None, scaler_hr=None,
                          rms_ref_br=None, rms_ref_hr=None,
                          br_win=10.0, step=2.0,
                          br_hold_factor=0.25):
    """
    Run detection on a single .mat file.
    Returns dict with results for BR and HR and additional metadata.
    """
    # load models if not provided
    if br_model is None:
        br_model = joblib.load(os.path.join(MODELS_PATH, "br_model.pkl"))
    if hr_model is None:
        hr_model = joblib.load(os.path.join(MODELS_PATH, "hr_model.pkl"))
    if scaler_br is None:
        scaler_br = joblib.load(os.path.join(MODELS_PATH, "scaler_br.pkl"))
    if scaler_hr is None:
        scaler_hr = joblib.load(os.path.join(MODELS_PATH, "scaler_hr.pkl"))
    if rms_ref_br is None:
        rms_ref_br = joblib.load(os.path.join(MODELS_PATH, "rms_ref_br.pkl"))
    if rms_ref_hr is None:
        rms_ref_hr = joblib.load(os.path.join(MODELS_PATH, "rms_ref_hr.pkl"))

    disp = load_chest_motion(filepath)
    disp = upsample_signal(first_samples, disp, data_fs, ups_factor)
    d_tot = chest_displacement(disp, xh, yh, xtx, ytx, xrx, yrx)
    phi = compute_phase(d_tot, cf)
    phase_resp = bp_filter(phi, Fs_slow, resp_low, resp_high)
    phase_heart = bp_filter(phi, Fs_slow, 0.8, 2.5)

    Xbr, Xhr, t_centers = extract_features_from_windows(phase_resp, phase_heart, Fs_slow,
                                                       win_s=br_win, step_s=step)

    result = {
        "file": filepath,
        "br_windows": len(Xbr),
        "hr_windows": len(Xhr),
        "br_anomaly_ratio": None,
        "hr_anomaly_ratio": None,
        "br_state": None,
        "hr_state": None,
        "t_centers": t_centers
    }

    # Breath-hold detection (global amplitude check)
    rms_now = np.median(Xbr[:, 0]) if Xbr.shape[0] > 0 else np.sqrt(np.mean(phase_resp**2))
    if rms_now < br_hold_factor * rms_ref_br:
        result["br_state"] = "BREATH_HOLD"
    else:
        # predict windows with model (1 normal, -1 anomaly)
        if Xbr.shape[0] > 0:
            Xb_s = scaler_br.transform(Xbr)
            preds_b = br_model.predict(Xb_s)
            anomaly_ratio_b = np.mean(preds_b == -1)
            result["br_anomaly_ratio"] = float(anomaly_ratio_b)
            result["br_state"] = "IRREGULAR" if anomaly_ratio_b > 0.4 else "NORMAL"
        else:
            result["br_state"] = "INSUFFICIENT_DATA"

    # Heart
    rms_now_h = np.median(Xhr[:, 0]) if Xhr.shape[0] > 0 else np.sqrt(np.mean(phase_heart**2))
    if rms_now_h < 0.25 * rms_ref_hr:
        result["hr_state"] = "HR_SIGNAL_TOO_LOW"
    else:
        if Xhr.shape[0] > 0:
            Xh_s = scaler_hr.transform(Xhr)
            preds_h = hr_model.predict(Xh_s)
            anomaly_ratio_h = np.mean(preds_h == -1)
            result["hr_anomaly_ratio"] = float(anomaly_ratio_h)
            result["hr_state"] = "IRREGULAR" if anomaly_ratio_h > 0.3 else "NORMAL"
        else:
            result["hr_state"] = "INSUFFICIENT_DATA"
    
    # after predicting for BR
    Xb_s = scaler_br.transform(Xbr)
    preds_b = br_model.predict(Xb_s)
    anomaly_ratio_b = np.mean(preds_b == -1)
    anomaly_br = (preds_b == -1)
    result["anomaly_br"] = anomaly_br
    anomaly_hr = (preds_h == -1)
    result["anomaly_hr"] = anomaly_hr
    result["phase_resp"] = phase_resp
    result["phase_heart"] = phase_heart
    result["t_slow"] = np.arange(len(phase_resp)) / Fs_slow


    # plot_anomalies(
    #     t_slow=result["t_slow"],
    #     phase_resp=result["phase_resp"],
    #     phase_heart=result["phase_heart"],
    #     t_centers=result["t_centers"],
    #     anomaly_br=result["anomaly_br"],
    #     anomaly_hr=result["anomaly_hr"],
    #     win_s=10.0,
    #     title=f"Anomaly Detection for {filepath}"
    # )


    return result


def test_models_on_dirs(irreg_dir=None, hold_dir=None):
    if irreg_dir is None:
        irreg_dir = DATA_IRREG
    if hold_dir is None:
        hold_dir = DATA_HOLD

    br_model = joblib.load(os.path.join(MODELS_PATH, "br_model.pkl"))
    hr_model = joblib.load(os.path.join(MODELS_PATH, "hr_model.pkl"))
    scaler_br = joblib.load(os.path.join(MODELS_PATH, "scaler_br.pkl"))
    scaler_hr = joblib.load(os.path.join(MODELS_PATH, "scaler_hr.pkl"))
    rms_ref_br = joblib.load(os.path.join(MODELS_PATH, "rms_ref_br.pkl"))
    rms_ref_hr = joblib.load(os.path.join(MODELS_PATH, "rms_ref_hr.pkl"))

    print("Testing irregular breathing files:")
    for f in sorted(glob(os.path.join(irreg_dir, "*.mat"))):
        res = run_inference_on_file(f, br_model, hr_model, scaler_br, scaler_hr, rms_ref_br, rms_ref_hr)
        print(os.path.basename(f), "→ BR:", res["br_state"], f"(anomaly_ratio={res['br_anomaly_ratio']})",
              "| HR:", res["hr_state"], f"(anomaly_ratio={res['hr_anomaly_ratio']})")

    print("\nTesting breath-hold files:")
    for f in sorted(glob(os.path.join(hold_dir, "*.mat"))):
        res = run_inference_on_file(f, br_model, hr_model, scaler_br, scaler_hr, rms_ref_br, rms_ref_hr)
        print(os.path.basename(f), "→ BR:", res["br_state"], f"(anomaly_ratio={res['br_anomaly_ratio']})",
              "| HR:", res["hr_state"], f"(anomaly_ratio={res['hr_anomaly_ratio']})")


# -------------- CLI --------------
def _usage():
    print("Usage:")
    print(" python -m src.ml.br_hr_ml train")
    print(" python -m src.ml.br_hr_ml test")
    print(" python -m src.ml.br_hr_ml infer path/to/file.mat")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "train":
        train_models(save=True)
    elif cmd == "test":
        test_models_on_dirs()
    elif cmd == "infer":
        if len(sys.argv) < 3:
            print("Please provide a .mat file to infer.")
            sys.exit(1)
        fpath = sys.argv[2]
        res = run_inference_on_file(fpath)
        print("Inference result:", res)
    else:
        _usage()
