import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# --------------- User / radar params ----------------
data_fs = 100                            # original sampling rate of my dataset [Hz]
K = 1024                            # number of OFDM subcarriers
t_snr_db = 20                       # signal to noise-ratio
ref_cof = 0.685                       # reflection coefficient
a_gain = 13                         # antenna gain in dBi
t_pow = 20                          # transmit power in dBm
samp_rate = 2.5 * int(m.pow(10, 6)) # 2.4576 MHz sampling rate
c = 299792458                       # speed of light in m/s
cf = 26.5 * int(m.pow(10, 9))       # 26.5 GHz
b = 1 * int(m.pow(10, 9))           # 1 GHz total occupied bandwidth
lam = c / cf                        # carrier wavelength in m
delta_f = b / K                     # Subcarrier spacing (assuming contiguous K subcarriers over bandwidth b)
start_f = cf - delta_f * (K // 2)   # Start frequency of the first subcarrier
freqs = start_f + delta_f * np.arange(K)  # Frequencies of each subcarrier
first_samples = 28000                 # number of first samples
sym_fs = 2500                       # sampling of each symbol at 2500 Hz
data_fs = 100.0                     # original data sampling frequency 100Hz
des_ups_factor = samp_rate / data_fs    # upsampling factor to match symbol sampling
des_ups_factor = int(des_ups_factor)        # integer upsampling factor
ups_factor = 1                     # used upsampling factor
print("Desired Upsampling factor:", des_ups_factor)
print("Used Upsampling factor:", ups_factor)
print("number of first samples", first_samples)
print("number of subcarriers", K)
print("freqs shape:", freqs.shape)
# print(f"wavelength = {lam:.6e} m")
# print(f"delta_f = {delta_f:.3f} Hz")

start = time.perf_counter()
h_pos = [xh, yh] = [10, 2]           # human position
tx_pos = [xtx, ytx] = [3, 0]        # transmitter position
rx_pos = [xrx, yrx] = [3, 0]        # receiver position
chair_pos = [xch, ych] = [2, 0]     # chair position


# --------------- Load chest motion ---------------
def load_chest_motion():

    """Load chest surface motion data from .mat file and preprocess it.
    Returns:
        1D array: Preprocessed chest displacement in meters."""
    
    # mat = sio.loadmat('Multimodal chest surface motion\Free_T2'
    # '.mat')
    mat = sio.loadmat('Breath Hold\Hold_T2.mat')
    # Assume the chest motion is in second row; adapt key/structure as needed
    # E.g., if mat['Free_T1'] exists: disp = mat['Free_T1'][1,:].squeeze()
    # Here try common patterns:
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    raw = mat[key]
    if raw.ndim == 2 and raw.shape[0] >= 2:
        disp_m = np.array(raw[1, :]).astype(float).squeeze()
    else:
        disp_m = np.array(raw).astype(float).squeeze()
    if np.nanmax(np.abs(disp_m)) > 0.02:
        disp_m = disp_m * 1e-3  # convert mm to m if needed
    disp_m = disp_m - np.mean(disp_m)   # remove DC
    return disp_m
# ---------------------------------------------------

#--------------- interpolation ---------------
def upsample_signal(first_samples, signal, factor):

    """Upsample the input signal by the given factor using cubic interpolation.
    Args:
        first_samples (int): Number of initial samples to consider from the input signal.
        signal (1D array): Input signal to be upsampled.
        factor (int): Upsampling factor.
    Returns:
        1D array: Upsampled signal."""
    
    # Given: disp_m (1D array of length N), sampled at Fs = 100 Hz
    dt = 1.0 / data_fs
    short_disp = signal[:first_samples]
    N = len(short_disp)  # use first 500 samples for testing
    t = np.arange(N) * dt # Original time grid
    t_dense = np.linspace(t[0], t[-1], factor*N, endpoint=True) # New dense time grid
    f_cub = interp1d(t, short_disp, kind='cubic',  bounds_error=False, fill_value='extrapolate')
    new_disp_m = f_cub(t_dense)
    print("new shape:", new_disp_m.shape)
    return new_disp_m
# ---------------------------------------------------
# --------------- Calculate distance ---------------
def dist(x1, y1, x2, y2):

    """Calculate Euclidean distance between two points."""

    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
# ---------------------------------------------------
# --------------- Calculate path length ---------------
# Assume the chest motion is along x direction
def chest_displacement(disp, chest_pos_x, chest_pos_y, xtx, ytx, xrx, yrx):

    """Calculate time-varying path length due to chest motion.
    Args:
        disp (1D array): Chest displacement over time.
        chest_pos_x (float): Initial x position of the chest.
        chest_pos_y (float): y position of the chest.
        xtx (float): x position of the transmitter.
        ytx (float): y position of the transmitter.
        xrx (float): x position of the receiver.
        yrx (float): y position of the receiver.
    Returns:
        1D array: Total path length over time."""
    
    t_disp = disp + chest_pos_x                     #h_pos[0], absolute position of chest surface
    # Time-varying path length due to chest motion
    d_tx = dist(t_disp, chest_pos_y, xtx, ytx)           # time-varying distance from tx to chest
    d_rx = dist(t_disp, chest_pos_y, xrx, yrx)           # time-varying distance from chest to rx
    d_tot = d_tx + d_rx                         # total path length
    print("Max, min path length change (m):", np.max(d_tot), np.min(d_tot))
    return d_tot
# ---------------------------------------------------
# ----------- calculate phase change --------------
def compute_phase(d, f):

    """Compute phase change due to path length d at frequency f.
    Args:
        d (1D array): Path length over time.
        f (float): Frequency.
    Returns:
        1D array: Phase change over time."""
    
    lam = c / f
    return np.float64((2 * np.pi / lam) * d)
# ---------------------------------------------------
# ----------- calculate path loss (dB) --------------
def free_space_path_loss(d, f):

    """Calculate free-space path loss in dB.
    Args:
        d (1D array): Path length over time.
        f (float): Frequency.
    Returns:
        1D array: Path loss in dB over time."""
    
    L = (4 * np.pi / c)
    L_dB = 20 * np.log10(d) + 20 * np.log10(f) + 20 * np.log10(L) + 20 * np.log10(ref_cof) # reflection coefficient from human included here
    return L_dB
# ---------------------------------------------------
# ----------- calculate received power --------------
def pw_recvd_dBm(PL_dB, t_pow):           # [W]

    """Calculate received power in dBm given path loss and transmit power.
    Args:
        PL_dB (1D array): Path loss in dB over time.
        t_pow (float): Transmit power in dBm.
    Returns:
        1D array: Received power in dBm over time."""
    
    return t_pow - PL_dB

def pw_recvd_w(pw_r_dBm):

    """Convert received power from dBm to watts.
    Args:
        pw_r_dBm (1D array): Received power in dBm over time.
    Returns:
        1D array: Received power in watts over time."""
    
    return 1e-3 * np.power(10, pw_r_dBm / 10)         # W
# ---------------------------------------------------
# ---------------- noise ---------------- TO BE REPLACED WITH NOISE FACTOR AND SYSTEM TEMPERATURE   -----------
def add_noise(pw_r_dBm, pw_r_w, t_snr_db):
    sig_avg_dB= np.mean(pw_r_dBm)               # mean singal power in dB
    noise_avg_db = sig_avg_dB - t_snr_db        # noise power in dB
    noise_avg_watts = 10 ** (noise_avg_db / 10) # noise power in watts
    mean_noise = 0                              # Generate white Gaussian noise
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(pw_r_w)) # noise samples
    sig_noise_w = pw_r_w + noise                # received signal + noise power in watts
    amp_noise = np.sqrt(sig_noise_w)            # signal amplitude [sqrt(W)]
    return amp_noise
# ---------------------------------------------------

def reshape_for_subcarriers(signal, fs_sym):
    
    """Reshape the signal into rows of length fs_sym.
    Args:
        signal (1D array): Input signal to be reshaped.
        fs_sym (int): Length of each row (symbol length).
    Returns:
        2D array: Reshaped signal with rows of length fs_sym."""
    
    n_full = (len(signal) // fs_sym) * fs_sym       # Trim tail to a multiple of K
    a_trim = signal[:n_full]
    reshaped = a_trim.reshape(-1, fs_sym)      # shape: (n_full//K, K)
    return reshaped

# ------------------ DSP ---------------------
def dsp(tx, rx):
    RX = np.fft.fft(rx, axis=1)
    TX = np.fft.fft(tx, axis=1)
    H = RX / TX
    h = np.fft.ifft(H, axis=1)
    h_mag = np.abs(h)
    max_bin = np.argmax(np.mean(h_mag, axis=0))  # strongest return averaged across pulses
    h_max = h[:, max_bin]  # this is a complex signal varying over slow-time (i.e., per pulse)
    # Step 7: Phase & motion analysis
    changes = np.fft.fftshift(np.fft.fft(np.angle(h_max)))  # Doppler / phase spectrum
 
    return H, h, h_max, changes
# ----------------------------------------------

# ---------------- PLOTTING ---------------------
bed_dist_tx = dist(xh, yh, xtx, ytx)
bed_dist_rx = dist(xh, yh, xrx, yrx)
bed_tot = bed_dist_tx + bed_dist_rx
print("Total distance to bed (m):", bed_tot)
disp = load_chest_motion()
print("disp shape:", disp.shape)
samples = len(disp)
disp_m = upsample_signal(samples, disp, ups_factor)
print("disp_m shape:", disp_m.shape)
Fs_slow = data_fs * ups_factor                     # slow-time is 100 Hz
print("Slow-time sampling Fs_slow (Hz):", Fs_slow)
d_tot = chest_displacement(disp_m, h_pos[0], h_pos[1], tx_pos[0], tx_pos[1], rx_pos[0], rx_pos[1])  # total path length with chest motion
original_time = np.arange(len(d_tot)) / data_fs / ups_factor        # time in seconds
print("Original time shape:", original_time.shape)
phi_t = compute_phase(d_tot, cf)                                    # phase change due to chest motion
PL_dB = free_space_path_loss(d_tot, cf)                             # path loss due to chest motion
pw_r_dBm = pw_recvd_dBm(PL_dB, t_pow)                               # received power in dBm
pw_r_w = pw_recvd_w(pw_r_dBm)                                       # received power in watts
amp = np.sqrt(pw_r_w)                                               # received signal amplitude [sqrt(W)]

# ---------------- OFDM radar channel simulation in subcarrier domain -----------------
def channel_simulation(d_tot, Fs_slow, amp):
    
    """Simulate OFDM radar channel with given path length and amplitude.
    Args:
        d_tot (1D array): Total path length over time.
        Fs_slow (float): Slow-time sampling frequency.
        amp (1D array): Received signal amplitude over time.
    Returns:
        tuple: (h_slow, avg_profile, r_bin)
            h_slow (1D array): Complex slow-time signal at strongest range bin.
            avg_profile (1D array): Average range profile over slow-time.
            r_bin (int): Index of strongest range bin."""
    # 1) Slow-time definition
    N_slow = len(d_tot)   # number of pulses / OFDM frames

    print("Slow-time samples:", N_slow, "Fs_slow:", Fs_slow, "Hz")

    # 2) Build TX pilot on each subcarrier (complex baseband)
    M = 16
    m_vals = np.arange(M)
    I = 2*((m_vals % 4) - 1.5)
    Q = 2*((m_vals // 4) - 1.5)
    constellation = (I + 1j*Q) / np.sqrt(10)

    # Random QAM per subcarrier (you could also use all-ones pilot)
    data_idx = np.random.randint(0, M, K)
    qamSymbols = constellation[data_idx]     # shape (K,)

    # 3) Build channel H[n,k] for a single point target whose distance varies with time
    #    τ[n] = 2 d_tot[n] / c  (two-way delay)
    tau = 2.0 * d_tot / c   # shape (N_slow,)

    # convert freqs vector into shape (1,K) for broadcasting
    freqs_2d = freqs[np.newaxis, :]          # (1, K)
    tau_2d = tau[:, np.newaxis]              # (N_slow, 1)

    # Static amplitude from path loss, already in pw_r_w; we used amp = sqrt(pw_r_w)
    amp_2d = amp[:, np.newaxis]              # (N_slow, 1)

    # Frequency-dependent phase: -2π f_k τ[n]
    phase_2d = -2.0 * np.pi * freqs_2d * tau_2d   # (N_slow, K)

    H = amp_2d * np.exp(1j * phase_2d)            # (N_slow, K), complex channel

    # 4) Form TX and RX matrices: TX is same pilot each pulse
    TX = np.tile(qamSymbols, (N_slow, 1))                  # (N_slow, K)
    RX = TX * H                                   # (N_slow, K) ideal, no noise

    # Optional: add complex AWGN in RF baseband domain
    snr_db = t_snr_db
    sig_power = np.mean(np.abs(RX)**2)
    noise_power = sig_power / (10**(snr_db/10.0))
    noise = np.sqrt(noise_power/2) * (np.random.randn(*RX.shape) + 1j*np.random.randn(*RX.shape))
    RX_noisy = RX + noise

    # 5) Range processing: FFT across subcarriers (fast-time) for each slow-time pulse
    H_range = np.fft.ifft(RX_noisy / TX, axis=1)    # (N_slow, K), similar to your dsp() idea
    h_mag = np.abs(H_range)

    # Find strongest range bin (averaged over slow-time)
    avg_profile = np.mean(h_mag, axis=0)
    r_bin = np.argmax(avg_profile)
    print("Strongest range bin index:", r_bin)

    # Extract complex slow-time signal at that bin
    h_slow = H_range[:, r_bin]          # shape (N_slow,)

    # ---------------- BR / HR estimation from phase -----------------

    # 6) Phase over time
    phase_slow = np.unwrap(np.angle(h_slow))

    # Remove linear trend (distance offset) – crude detrend
    t_slow = np.arange(N_slow) / Fs_slow
    p_coeff = np.polyfit(t_slow, phase_slow, 1)
    phase_detr = phase_slow - np.polyval(p_coeff, t_slow)
    
    # 7) Design band-pass filters for respiration and heart
    def bp_filter(sig, fs, f_low, f_high, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [f_low/nyq, f_high/nyq], btype='band')
        return filtfilt(b, a, sig)

    # Respiration: ~0.1–0.5 Hz (6–30 bpm)
    phase_resp = bp_filter(phase_detr, Fs_slow, 0.1, 0.5)

    # Heart: ~0.8–2.0 Hz (48–120 bpm)
    phase_heart = bp_filter(phase_detr, Fs_slow, 0.8, 2.0)
    return h_slow, avg_profile, r_bin, phase_resp, phase_heart, p_coeff, t_slow
# ---------------- PLOTTING FUNCTIONS ---------------------

h_slow, avg_profile, r_bin, phase_resp, phase_heart, p_coeff, t_slow  = channel_simulation(d_tot, Fs_slow, amp)

def plot_breathing_spectrum(phase_signal, fs, fmax=1.3, label="Phase FFT"):
    """
    Plot frequency spectrum of respiratory phase signal with harmonics.
    Matches style of Fig. 17 (normalized amplitude, dashed harmonic lines).
    """
    N = len(phase_signal)
    
    # FFT
    win = np.hanning(N)
    spec = np.fft.rfft(phase_signal * win)
    freqs_fft = np.fft.rfftfreq(N, 1/fs)
    
    # Limit frequency axis
    mask = freqs_fft <= fmax
    freqs_use = freqs_fft[mask]
    spec_use = np.abs(spec[mask])
    
    # Normalize amplitude
    spec_norm = spec_use / np.max(spec_use)
    
    # Find main breathing peak automatically within 0.1-0.5 Hz
    br_mask = (freqs_fft >= 0.1) & (freqs_fft <= 0.5)
    br_idx = np.argmax(np.abs(spec[br_mask]))
    f_br = freqs_fft[br_mask][br_idx]

    # Find main breathing peak automatically within 0.1-0.5 Hz
    hr_mask = (freqs_fft >= 0.8) & (freqs_fft <= 2.0)
    hr_idx = np.argmax(np.abs(spec[hr_mask]))
    f_hr = freqs_fft[hr_mask][hr_idx]
    
    # Harmonics
    harmonics = [f_br, 2*f_br, 3*f_br]
    
    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(freqs_use, spec_norm, label="Filtered FFT", linewidth=2)
    
    # Vertical dashed lines for harmonics
    for h in harmonics:
        if h <= fmax:
            plt.axvline(h, color='red', linestyle='--', alpha=0.8,
                        label=f"Harmonic {h:.2f} Hz")
    
    plt.xlim(0, fmax)
    plt.ylim(0, 1.1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amplitude")
    plt.title("Frequency Components of Phase Variations (Breathing & Harmonics)")
    plt.grid(True)
    
    # Avoid repeated labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="upper right")
    
    plt.tight_layout()
    plt.show()
    
    print("Estimated breathing frequency:", f_br_est)
    print("Estimated breathing rate:", 60*f_br_est, "bpm")
    print("Estimated heart frequency:", f_hr_est)
    print("Estimated heart rate:", 60*f_hr_est, "bpm")


    return f_br, f_hr

phase_slow = np.unwrap(np.angle(h_slow))
phase_detr = phase_slow - np.polyval(p_coeff, t_slow)

f_br_est, f_hr_est = plot_breathing_spectrum(phase_detr, Fs_slow)

def plot_range_profile(avg_profile, r_bin, K, delta_f):
    """
    Plot range profile with detected person, matching style of Fig. 14.
    avg_profile : 1D array (K,) - averaged magnitude over slow-time
    r_bin       : strongest range bin
    K           : number of subcarriers
    delta_f     : subcarrier spacing
    """
    c = 299792458.0
    
    # ----- Compute distance axis -----
    # Standard OFDM radar: R = c / (2B) * k, where B = K * delta_f
    B = K * delta_f
    dist_axis = c / (2 * B) * np.arange(K)

    # ----- Compute time-of-flight axis for top x-axis -----
    tof_ns = (2 * dist_axis / c) * 1e9   # round-trip ToF [ns]

    # ----- Normalize profile -----
    prof_norm = avg_profile / np.max(avg_profile)

    # ----- Distance of strongest peak -----
    dist_detected = dist_axis[r_bin]
    tof_detected = tof_ns[r_bin]

    # ----- Plot -----
    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(dist_axis, prof_norm, label="Power Profile", linewidth=2)
    ax1.set_xlabel("Distance [m]")
    ax1.set_ylabel("Normalized Magnitude")
    ax1.set_xlim(0, 2 * dist_detected)
    ax1.grid(True)

    # Vertical dashed red line
    ax1.axvline(dist_detected, color='red', linestyle='--',
                label=f"Distance: {dist_detected:.2f} m")

    # Red marker at the peak
    ax1.plot(dist_detected, prof_norm[r_bin], 'ro', markersize=10)

    # ----- Add top x-axis (time of flight) -----
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(dist_axis[::K//6])   # 6 labels
    ax2.set_xticklabels([f"{t:.1f}" for t in tof_ns[::K//6]])
    ax2.set_xlabel("Time of Flight [ns]")

    # Legend
    ax1.legend(loc="upper right")

    plt.title("Measured Distance to a Single Person")
    plt.tight_layout()
    plt.xlim(0, dist_detected*2)
    plt.show()

    print(f"Detected distance: {dist_detected:.3f} m")
    print(f"Detected ToF    : {tof_detected:.2f} ns")

plot_range_profile(avg_profile, r_bin, K, delta_f)

# ================== ML-BASED BREATH-PATTERN DETECTION ==================
def compute_resp_features(sig, fs, win_s=10.0, step_s=2.0):
    """
    Slide a window over respiration signal and compute simple features.

    Parameters
    ----------
    sig : 1D array
        Respiration-band phase signal (e.g. phase_resp).
    fs : float
        Sampling frequency (Fs_slow).
    win_s : float
        Window length in seconds.
    step_s : float
        Step (hop) between windows in seconds.

    Returns
    -------
    feats : 2D array, shape (n_windows, n_features)
        Feature matrix.
    t_centers : 1D array, shape (n_windows,)
        Time (s) at center of each window.
    extra : dict
        Extra info (per-window RMS, etc) for later interpretation.
    """
    win = int(win_s * fs)
    step = int(step_s * fs)
    N = len(sig)

    features = []
    t_centers = []
    rms_list = []
    peak2peak_list = []

    # Precompute freq axis for breathing frequency feature
    # Limit to respiration band ~[0.05, 0.7] Hz
    for start in range(0, N - win, step):
        seg = sig[start:start+win]

        # Basic amplitude features
        rms = np.sqrt(np.mean(seg**2))
        var = np.var(seg)
        p2p = np.ptp(seg)

        # Spectral feature: dominant breathing frequency & peak prominence
        win_hann = np.hanning(len(seg))
        seg_win = seg * win_hann
        spec = np.fft.rfft(seg_win)
        freqs = np.fft.rfftfreq(len(seg), 1/fs)
        mag = np.abs(spec)

        # Restrict to breathing range
        br_mask = (freqs >= 0.05) & (freqs <= 0.7)
        if np.any(br_mask):
            mag_br = mag[br_mask]
            freqs_br = freqs[br_mask]
            idx_max = np.argmax(mag_br)
            f_dom = freqs_br[idx_max]
            dom_amp = mag_br[idx_max]
            mean_amp = np.mean(mag_br)
            peak_prom = dom_amp / (mean_amp + 1e-12)  # how strong the main peak is
        else:
            f_dom = 0.0
            peak_prom = 1.0

        # Zero-crossing rate (kind of “activity” measure)
        zc = np.mean(np.diff(np.sign(seg)) != 0)

        # Collect features
        features.append([
            rms,        # 0
            var,        # 1
            p2p,        # 2
            f_dom,      # 3
            peak_prom,  # 4
            zc          # 5
        ])

        center = (start + start + win) / 2.0 / fs
        t_centers.append(center)
        rms_list.append(rms)
        peak2peak_list.append(p2p)

    feats = np.array(features)
    t_centers = np.array(t_centers)
    extra = {
        "rms": np.array(rms_list),
        "peak2peak": np.array(peak2peak_list)
    }
    return feats, t_centers, extra


def train_normal_breath_model(features, t_centers, normal_duration_s=60.0,
                              contamination=0.1, random_state=0):
    """
    Train an IsolationForest on an initial segment assumed to be 'normal breathing'.

    Parameters
    ----------
    features : 2D array (n_windows, n_features)
    t_centers : 1D array, times of windows
    normal_duration_s : float
        Time length from t=0 used as 'normal' training data.
    contamination : float
        Expected fraction of anomalies in all data (for IsolationForest).
    random_state : int

    Returns
    -------
    model : IsolationForest
    scaler : StandardScaler
    normal_idx : indices of windows used for training
    """
    normal_idx = np.where(t_centers <= normal_duration_s)[0]
    if len(normal_idx) < 5:
        print("WARNING: Not enough normal windows for training, using all as normal.")
        normal_idx = np.arange(features.shape[0])

    X_train = features[normal_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(X_train_scaled)

    return model, scaler, normal_idx


def classify_windows(features, t_centers, model, scaler, extra,
                     hold_rms_factor=0.4):
    """
    Classify each window as:
        0: normal breathing
        1: breath hold (very low amplitude, anomalous)
        2: irregular breathing (anomalous, but not too low amplitude)

    Parameters
    ----------
    features : 2D array (n_windows, n_features)
    t_centers : 1D array
    model : IsolationForest
    scaler : StandardScaler
    extra : dict with "rms" and "peak2peak"
    hold_rms_factor : float
        Threshold as fraction of median normal RMS below which we call 'breath hold'.

    Returns
    -------
    labels : 1D array of ints in {0,1,2}
    scores : anomaly scores (higher = more anomalous)
    """
    X_scaled = scaler.transform(features)
    # IsolationForest: 1 = normal, -1 = anomaly
    raw_pred = model.predict(X_scaled)
    scores = -model.decision_function(X_scaled)  # larger -> more anomalous

    rms = extra["rms"]

    # Estimate "normal" RMS from windows model marked as normal
    normal_mask = (raw_pred == 1)
    if np.any(normal_mask):
        rms_normal_med = np.median(rms[normal_mask])
    else:
        rms_normal_med = np.median(rms)

    hold_thresh = hold_rms_factor * rms_normal_med

    labels = np.zeros_like(raw_pred, dtype=int)
    for i in range(len(raw_pred)):
        if raw_pred[i] == 1:
            labels[i] = 0  # normal
        else:
            # anomaly -> classify as hold vs irregular by amplitude
            if rms[i] < hold_thresh:
                labels[i] = 1  # breath hold
            else:
                labels[i] = 2  # irregular

    return labels, scores


def plot_breath_states(t_slow, phase_resp, t_centers, labels, win_s=10.0):
    """
    Visualize respiration signal and overlay detected states.

    labels: 0=normal, 1=hold, 2=irregular
    """
    plt.figure(figsize=(10,5))
    plt.plot(t_slow, phase_resp, label="Respiration phase", alpha=0.6)

    # Color map for states
    colors = {0: 'green', 1: 'red', 2: 'orange'}
    names = {0: 'normal', 1: 'hold', 2: 'irregular'}

    for state in [1, 2]:
        mask = labels == state
        for tc in t_centers[mask]:
            start = tc - win_s/2
            end = tc + win_s/2
            plt.axvspan(start, end, color=colors[state], alpha=0.2,
                        label=names[state])

    # Avoid repeated labels in legend
    handles, labs = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labs, handles))
    plt.legend(unique.values(), unique.keys(), loc="upper right")
    plt.xlabel("Time [s]")
    plt.ylabel("Phase (rad)")
    plt.title("Breath pattern classification")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------- ML-based detection of breath hold / irregular breathing ----------
# 1) Extract features from respiration-band phase
win_s = 10.0   # window length [s]
step_s = 2.0   # window hop [s]
features, t_centers, extra = compute_resp_features(phase_resp, Fs_slow,
                                                   win_s=win_s, step_s=step_s)

print("Feature matrix shape:", features.shape)

# 2) Train model on an initial segment assumed to be normal breathing
#    Adjust normal_duration_s depending on your recording.
normal_duration_s = 60.0
model, scaler, normal_idx = train_normal_breath_model(
    features, t_centers,
    normal_duration_s=normal_duration_s,
    contamination=0.1
)

print(f"Trained on {len(normal_idx)} windows as 'normal'")

# 3) Classify all windows
labels, scores = classify_windows(features, t_centers, model, scaler, extra,
                                  hold_rms_factor=0.4)

# labels: 0=normal, 1=breath hold, 2=irregular breathing
unique, counts = np.unique(labels, return_counts=True)
print("Breath state counts (0=normal,1=hold,2=irregular):")
for u, c in zip(unique, counts):
    print(f"  state {u}: {c} windows")

# 4) Optional visualization
plot_breath_states(t_slow, phase_resp, t_centers, labels, win_s=win_s)


# ------------------------------------------------------------
# OVERSAMPLING
# ------------------------------------------------------------
Fs_base = 1e6        # 1 MHz
desired_Fs = 10e6   # 10 MHz
L = int(np.ceil(desired_Fs / Fs_base))
Fs_high = L * Fs_base
print(f"Fs_base = {Fs_base/1e3:.1f} kHz, Fs_high = {Fs_high/1e3:.1f} kHz, L = {L}")

# ------------------------------------------------------------
# GENERATE 16-QAM SYMBOLS
# (unit-average-power mapping)
# ------------------------------------------------------------
# MATLAB qammod(...,'UnitAveragePower',true)
M = 16
m_vals = np.arange(M)
# Gray-coded 16-QAM
I = 2*((m_vals % 4) - 1.5)
Q = 2*((m_vals // 4) - 1.5)
constellation = (I + 1j*Q) / np.sqrt(10)     # normalized to unit avg power

data = np.random.randint(0, M, K)
qamSymbols = constellation[data]

# ------------------------------------------------------------
# MAP TO IFFT BINS (CENTERED)
# ------------------------------------------------------------
Nfft_time = K * L
X = np.zeros(Nfft_time, dtype=complex)

startIdx = Nfft_time // 2 - K//2
X[startIdx:startIdx + K] = qamSymbols

# ------------------------------------------------------------
# TIME-DOMAIN OFDM SIGNAL
# ------------------------------------------------------------
ofdm_oversampled = ifft(fftshift(X))

# Scale to 1 watt average power
power_signal = np.mean(np.abs(ofdm_oversampled)**2)
ofdm_oversampled = ofdm_oversampled / np.sqrt(power_signal)
tx_pow_w = 10**((t_pow - 30)/10)  # in watts
ofdm_oversampled = ofdm_oversampled * np.sqrt(tx_pow_w)
# ------------------------------------------------------------
# TIME VECTOR
# ------------------------------------------------------------
t = np.arange(Nfft_time) / Fs_high

# ------------------------------------------------------------
# BASEBAND PSD
# ------------------------------------------------------------
Nfft_spec = 10384
spec_base = fftshift(fft(ofdm_oversampled, Nfft_spec))
Pxx_base = (np.abs(spec_base)**2) / (Nfft_spec * Fs_high)
Pxx_base_dBHz = 10*np.log10(Pxx_base + 1e-15)
f_axis_base = np.linspace(-Fs_high/2, Fs_high/2, Nfft_spec)

# ------------------------------------------------------------
# PASSBAND UP-CONVERSION
# ------------------------------------------------------------
ofdm_passband = np.real(ofdm_oversampled * np.exp(1j*2*np.pi*cf*t))

# Passband PSD
spec_pass = fftshift(fft(ofdm_passband, Nfft_spec))
Pxx_pass = (np.abs(spec_pass)**2) / (Nfft_spec * Fs_high)
Pxx_pass_dBHz = 10*np.log10(Pxx_pass + 1e-15)
f_axis_pass = np.linspace(-Fs_high/2, Fs_high/2, Nfft_spec)

# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------

# ---------------- Simple debug plots (optional) -----------------
plt.figure()
plt.subplot(3,1,1)
plt.plot(t_slow, phase_detr)
plt.title("Detrended phase at target range bin")

plt.subplot(3,1,2)
plt.plot(t_slow, phase_resp)
plt.title("Respiration-band phase")

plt.subplot(3,1,3)
plt.plot(t_slow, phase_heart)
plt.title("Heart-band phase")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(f_axis_pass/1e6, Pxx_pass_dBHz)
plt.xlabel("Frequency (MHz)")
plt.ylabel("PSD (dB/Hz)")
plt.title(f"Passband PSD centered at {cf/1e6:.2f} MHz")
plt.grid(True)
plt.show()



