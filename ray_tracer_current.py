import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

# --------------- User / radar params ----------------
data_fs = 100                            # original sampling rate of my dataset [Hz]
K = 1024                            # number of OFDM subcarriers
t_snr_db = 50                       # signal to noise-ratio
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
first_samples = 100                 # number of first samples
sym_fs = 2500                       # sampling of each symbol at 2500 Hz
data_fs = 100.0                     # original data sampling frequency 100Hz
ups_factor = samp_rate / data_fs    # upsampling factor to match symbol sampling
workers = 4                         # number of parallel workers
print("Desired Upsampling factor:", ups_factor)
ups_factor = int(ups_factor)        # integer upsampling factor
ups_factor = 2500
dec = 13                            # decimal precision for tests
print("Used Upsampling factor:", ups_factor)
print("number of first samples", first_samples)
print("number of subcarriers", K)
print("Number of workers:", workers)
print("Decimal places for assert_almost_equal:", dec)
print("freqs shape:", freqs.shape)
# print(f"wavelength = {lam:.6e} m")
# print(f"delta_f = {delta_f:.3f} Hz")

start = time.perf_counter()
h_pos = [xh, yh] = [1, 2]           # human position
tx_pos = [xtx, ytx] = [3, 0]        # transmitter position
rx_pos = [xrx, yrx] = [3, 0]        # receiver position
# chair_pos = [xch, ych] = [2, 0]     # chair position
# bed_dist_tx = dist(xh, yh, xtx, ytx)
# print("Distance to bed from tx (m):", bed_dist_tx)
# bed_dist_rx = dist(xh, yh, xrx, yrx)
# print("Distance to bed from rx (m):", bed_dist_rx)

# --------------- Load chest motion ---------------
def load_chest_motion():

    """Load chest surface motion data from .mat file and preprocess it.
    Returns:
        1D array: Preprocessed chest displacement in meters."""
    
    mat = sio.loadmat('Multimodal chest surface motion\Free_T1'
    '.mat')
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

# original_time = np.arange(len(disp_m)) / data_fs     # time in seconds

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
# ------------------ TX signal ---------------------
def tx_phi_signal(data_fs, f, data_time, ups_factor):

    """Generate transmitted phase signal for frequency f over data_time.
    Args:
        data_fs: Samplig rate od data.
        f (float): Frequency of the subcarrier.
        data_time (1D array): Time indices for the data samples.
        ups_factor: Upsampling factor.
    Returns:
        1D array: Phase signal in radians."""
    
    N = len(data_time)                  # number of samples
    fs = data_fs * ups_factor           # new sampling frequency
    t = np.arange(N, dtype=np.float64)                     # time array after upsampling
    tx_phi = np.mod(2 * np.pi * f * t, 2 * np.pi).astype(np.float64)   # phase array in radians
    return tx_phi

def tx_signal(data_fs, data_time, f, t_pow, ups_factor):

    """Generate transmitted OFDM signal for frequency f over data_time with transmit power t_pow.
    Args:
        f (float): Frequency of the subcarrier.
        data_time (1D array): Time indices for the data samples.
        t_pow (float): Transmit power in dBm.
    Returns:
        1D array: Complex baseband transmitted signal."""
    
    tx_tp_watts = 1e-3 * 10 ** (t_pow / 10)     # transmit power in watts
    amp = np.sqrt(tx_tp_watts).astype(np.float64)                  # signal amplitude [sqrt(W)]
    tx_phi = tx_phi_signal(data_fs, f, data_time, ups_factor)              # transmitted phase
    tx = np.complex128(amp * np.exp(1j * tx_phi))
    return tx

def tx_signal_subcarriers(data_fs, f_range, sym_fs, t_pow, data_time, ups_factor):
    """Transmit signal over multiple subcarriers.
    Args:
        f_range (1D array): Frequency range for subcarriers.
        sym_fs (int): Symbol length in samples.
        t_pow (float): Transmit power in dBm.
    Returns:
        2D array: Transmitted signal over multiple subcarriers.
    """
    sum_tx = np.sum(np.vectorize(lambda f: tx_signal(data_fs, data_time, f, t_pow, ups_factor), otypes=[np.ndarray])(f_range), axis=0)

    tx_subcarriers = reshape_for_subcarriers(sum_tx, sym_fs)
    return tx_subcarriers
# -------------------------------------------------------------------

# ------------ RX signal --------------
def rx_signal(data_fs, d, f, t_pow, ups_factor):
    single_tx_phi = tx_phi_signal(data_fs, f, d, ups_factor)              # transmitted phase
    phase = compute_phase(d, f)
    rx_phi = np.float32(single_tx_phi + phase)
    PL_dB = free_space_path_loss(d, f)
    pw_r_dBm = pw_recvd_dBm(PL_dB, t_pow)
    pw_r_w = pw_recvd_w(pw_r_dBm)        # W
    amp = np.sqrt(pw_r_w).astype(np.float64)                # signal amplitude [sqrt(W)]
    # amp_noise = add_noise(pw_r_dBm, pw_r_w, t_snr_db) # czystq amplituda
    single_rx = np.complex128(amp * np.exp(1j * rx_phi))
    return single_rx

def rx_signal_subcarriers(data_fs, f_range, sym_fs, t_pow, d, ups_factor):
    """Receive signal over multiple subcarriers.
    Args:
        d (1D array): Distance array.
        f_range (1D array): Frequency range for subcarriers.
        sym_fs (int): Symbol length in samples.
        t_pow (float): Transmit power in dBm.
    Returns:
        2D array: Received signal over multiple subcarriers.
    """
    sum_rx = np.sum(np.vectorize(lambda f: rx_signal(data_fs, d, f, t_pow, ups_factor), otypes=[np.ndarray])(f_range), axis=0)

    rx_subcarriers = reshape_for_subcarriers(sum_rx, sym_fs)
    return rx_subcarriers
# -------------------------------------------------------------------

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
# ------------------ PARALLEL SIGNALS ---------------------
def tx_signal_sum_parallel(data_fs, data_time, f_range, t_pow, ups_factor, n_workers=workers):
    """
    Efficient parallel transmitted signal computation across multiple subcarriers.
    Args:
        data_fs: Sampling rate of data.
        data_time (1D array): Time indices for the data samples.
        f_range (1D array): Frequency range for subcarriers.
        t_pow (float): Transmit power in dBm.
        ups_factor: Upsampling factor.
        n_workers (int): Number of parallel workers.
    Returns:
        1D array: Complex baseband transmitted signal summed over subcarriers.
    """
    def worker(f_chunk):
        s = np.zeros_like(data_time, dtype=np.complex128)
        for f in f_chunk:
            s += tx_signal(data_fs, data_time, f, t_pow, ups_factor)
        return s
    
    chunks = np.array_split(f_range, n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(worker, chunks))
    res = np.sum(results, axis=0)
    return res

def tx_vectorized_signal_sum(data_fs, data_time, f_range, t_pow, ups_factor):
    """
    Vectorized transmitted signal computation across multiple subcarriers.
    Args:
        data_fs: Sampling rate of data.
        data_time (1D array): Time indices for the data samples.
        f_range (1D array): Frequency range for subcarriers.
        t_pow (float): Transmit power in dBm.
        ups_factor: Upsampling factor.
    Returns:
        1D array: Complex baseband transmitted signal summed over subcarriers.
    """
    sum_tx = np.sum(np.vectorize(lambda f: tx_signal(data_fs, data_time, f, t_pow, ups_factor), otypes=[np.ndarray])(f_range), axis=0)
    return sum_tx

def rx_signal_sum_parallel(data_fs, data_time, f_range, t_pow, ups_factor, n_workers=workers):
    """
    Efficient parallel received signal computation across multiple subcarriers.
    Args:
        data_fs: Sampling rate of data.
        data_time (1D array): Time indices for the data samples.
        f_range (1D array): Frequency range for subcarriers.
        t_pow (float): Transmit power in dBm.
        ups_factor: Upsampling factor.
        n_workers (int): Number of parallel workers.
    Returns:
        1D array: Complex baseband received signal summed over subcarriers.
    """
    def worker(f_chunk):
        s = np.zeros_like(data_time, dtype=np.complex128)
        for f in f_chunk:
            s += rx_signal(data_fs, data_time, f, t_pow, ups_factor)
        return s
    
    chunks = np.array_split(f_range, n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(worker, chunks))
    res = np.sum(results, axis=0)
    return res

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

chest_disp = load_chest_motion()
disp_m = upsample_signal(first_samples, chest_disp, ups_factor)  # upsample to match symbol sampling rate
d_tot = chest_displacement(disp_m, h_pos[0], h_pos[1], tx_pos[0], tx_pos[1], rx_pos[0], rx_pos[1])  # total path length with chest motion

# ---------------- PLOTTING ---------------------
original_time = np.arange(len(d_tot)) / data_fs / ups_factor        # time in seconds
single_tx = tx_signal(data_fs, d_tot, cf, t_pow, ups_factor)        # transmitted signal
phi_t = compute_phase(d_tot, cf)                                    # phase change due to chest motion
PL_dB = free_space_path_loss(d_tot, cf)                             # path loss due to chest motion
pw_r_dBm = pw_recvd_dBm(PL_dB, t_pow)                               # received power in dBm
pw_r_w = pw_recvd_w(pw_r_dBm)                                       # received power in watts
single_rx = rx_signal(data_fs, d_tot, cf, t_pow, ups_factor)        # received signal with chest motion
sum_tx = tx_signal_sum_parallel(data_fs, d_tot, freqs, t_pow, ups_factor, n_workers=workers) # transmitted signal over all subcarriers
# res_sum_tx = reshape_for_subcarriers(sum_tx, sym_fs)
# sum_rx = rx_signal_sum_parallel(data_fs, d_tot, freqs, t_pow, ups_factor, n_workers=workers) # received signal over all subcarriers
# res_sum_rx = reshape_for_subcarriers(sum_rx, sym_fs)
# H, h, h_max, changes = dsp(res_sum_tx, res_sum_rx)                          # DSP processing

def generate_tx_ofdm_baseband(data_fs, ups_factor, N, freqs, t_pow, cf):
    """
    Generate baseband OFDM transmit signal for all subcarriers.

    Returns:
        tx_bb : np.ndarray, shape (K, N)
            Complex baseband signal per subcarrier.
        freqs_bb : np.ndarray, shape (K,)
            Baseband subcarrier frequencies (Hz), centered around 0.
    """
    fs_eff = data_fs * ups_factor          # effective sampling rate (Hz)
    K = len(freqs)                         # number of subcarriers
    n = np.arange(N, dtype=np.float64)     # sample index

    # Total TX power
    Ptx_W = 1e-3 * 10**(t_pow / 10.0)      # W

    # Optional: normalize per subcarrier so total power stays Ptx_W
    amp = np.sqrt(Ptx_W) / np.sqrt(K)      # sqrt(W), per-subcarrier amplitude

    # Baseband subcarrier frequencies (offset from carrier)
    freqs_bb = freqs - cf                  # Hz, centered around 0

    # Digital angular frequency (rad/sample) for each subcarrier
    omega = 2.0 * np.pi * freqs_bb / fs_eff   # shape (K,)

    # Phase matrix: (K, N)
    phase_tx = omega[:, None] * n[None, :]     # outer product

    # Complex baseband TX per subcarrier: (K, N)
    tx_bb = amp * np.exp(1j * phase_tx)

    return tx_bb, freqs_bb


def generate_rx_ofdm_baseband(d_tot, freqs, data_fs, ups_factor, t_pow, cf):
    """
    Generate baseband OFDM received signal for all subcarriers,
    including chest-motion-induced phase and free-space path loss.

    Args:
        d_tot : 1D array of length N
            Time-varying total path length (Tx -> chest -> Rx) [m].
        freqs : 1D array of length K
            Absolute RF frequencies of subcarriers [Hz].
    Returns:
        rx_bb : np.ndarray, shape (K, N)
            Complex baseband received signal per subcarrier.
    """
    N = len(d_tot)
    K = len(freqs)
    fs_eff = data_fs * ups_factor

    n = np.arange(N, dtype=np.float64)

    # ---------- Transmit baseband (same as in TX) ----------
    Ptx_W = 1e-3 * 10**(t_pow / 10.0)
    amp_tx = np.sqrt(Ptx_W) / np.sqrt(K)

    freqs_bb = freqs - cf
    omega = 2.0 * np.pi * freqs_bb / fs_eff
    phase_tx = omega[:, None] * n[None, :]       # (K, N)

    # ---------- Channel: path loss + chest motion phase ----------
    # Broadcast shapes: f -> (K,1), d -> (1,N)
    f = freqs[:, None]               # (K,1)
    d = d_tot[None, :]               # (1,N)

    lam = c / f                      # wavelength per subcarrier (K,1)

    # Chest-motion phase per subcarrier & time
    phi_chest = (2.0 * np.pi / lam) * d    # (K, N)

    # Free-space path loss with reflection coefficient
    L_const = 20.0 * np.log10(4.0 * np.pi / c) + 20.0 * np.log10(ref_cof)
    PL_dB = 20.0 * np.log10(d) + 20.0 * np.log10(f) + L_const  # (K, N)

    # Received power
    Pw_r_dBm = t_pow - PL_dB
    Pw_r_W = 1e-3 * 10.0**(Pw_r_dBm / 10.0)       # (K, N)
    amp_rx = np.sqrt(Pw_r_W)                      # sqrt(W), (K, N)

    # ---------- Combine TX phase + channel phase ----------
    phase_rx = phase_tx + phi_chest               # (K, N)
    rx_bb = amp_rx * np.exp(1j * phase_rx)        # (K, N)

    return rx_bb


fs_eff = data_fs * ups_factor
N = len(d_tot)

# --- TX: OFDM baseband ---
tx_bb, freqs_bb = generate_tx_ofdm_baseband(
    data_fs=data_fs,
    ups_factor=ups_factor,
    N=N,
    freqs=freqs,   # your absolute RF freqs array
    t_pow=t_pow,
    cf=cf
)
# Sum across subcarriers to get composite TX waveform
sum_tx = tx_bb.sum(axis=0)    # shape (N,)

# --- RX: OFDM baseband with chest motion ---
rx_bb = generate_rx_ofdm_baseband(
    d_tot=d_tot,
    freqs=freqs,
    data_fs=data_fs,
    ups_factor=ups_factor,
    t_pow=t_pow,
    cf=cf
)
sum_rx = rx_bb.sum(axis=0)    # shape (N,)

# FFT
N = len(sum_tx)
TX_f = np.fft.fft(sum_tx)
TX_mag = np.abs(TX_f)

# Frequency axis in Hz
freq_axis = np.fft.fftfreq(N, d=1/fs_eff)

# Shift for nicer centered plot
TX_mag_shift = np.fft.fftshift(TX_mag)
freq_axis_shift = np.fft.fftshift(freq_axis)

# ---- Plot (linear magnitude) ----
plt.figure(figsize=(10,4))
plt.plot(freq_axis_shift, TX_mag_shift)
plt.title("TX Signal Spectrum (Magnitude)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|FFT|")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Plot in dB ----
plt.figure(figsize=(10,4))
plt.plot(freq_axis_shift, 20*np.log10(TX_mag_shift + 1e-12))
plt.title("TX Signal Spectrum (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(np.abs(sum_tx), label='TX |sum of subcarriers|')
plt.xlabel('Sample index')
plt.ylabel('Amplitude (sqrt(W))')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(20*np.log10(np.abs(sum_tx) + 1e-12), label='TX (dB)')
plt.xlabel('Sample index')
plt.ylabel('Amplitude (dB)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(np.abs(sum_rx), label='RX |sum of subcarriers|')
plt.xlabel('Sample index')
plt.ylabel('Amplitude (sqrt(W))')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(20*np.log10(np.abs(sum_rx) + 1e-12), label='RX (dB)')
plt.xlabel('Sample index')
plt.ylabel('Amplitude (dB)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

fs_eff = data_fs * ups_factor  # 2500 Hz
n = np.arange(sym_fs)          # 2500 samples
t = n / fs_eff                 # time grid
E = np.exp(-1j * 2*np.pi * np.outer(freqs, t))   # (K, 2500)
X = reshape_for_subcarriers(sum_rx, sym_fs)  # shape (Nsym, 2500)

TXk = X @ E.T      # shape (Nsym, K)

mag = np.mean(np.abs(TXk), axis=0)
mag_db = 20*np.log10(mag + 1e-12)



plt.figure(figsize=(10,5))
plt.plot(freqs * 1e-9, mag_db, lw=1.5)
plt.title("Subcarrier Magnitude")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.show()





# TX = np.fft.fft(res_sum_tx, axis=1)

# # 4) Average magnitude across symbols and convert to dB
# tx_mag = np.mean(np.abs(TX[:, :K]), axis=0)
# tx_mag_db = 20 * np.log10(tx_mag + 1e-12)

# # 5) Plot magnitude (dB) vs subcarrier frequencies (GHz)
# plt.figure(figsize=(10,5))
# plt.plot(freqs * 1e-9, tx_mag_db, lw=1.5)
# plt.title("Subcarrier Magnitude")
# plt.xlabel("Frequency [GHz]")
# plt.ylabel("Magnitude [dB]")
# plt.grid(True)
# plt.show()

# # -----------------Plot chest motion ---------------
# plt.figure(figsize=(10, 4))
# plt.plot(original_time, disp_m, label='Chest displacement', linewidth=1.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Displacement (mm)')
# plt.title('Chest Surface Motion (Free T2)')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # ---------------------------------------------------

# # --------------- Plot path length ---------------
# plt.figure(figsize=(10, 4))
# plt.plot(original_time, d_tot, label='Total Path Length', color='orange', linewidth=1.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Path Length (m)')
# plt.title('Total Path Length Variation Due to Chest Motion')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.ylim(5.5, 5.7)
# plt.tight_layout()
# plt.show()
# # ---------------------------------------------------

# # --------------- Plot phase change ---------------
# plt.figure(figsize=(10, 4))
# plt.plot(original_time, phi_t, label='Chest Phase (wrapped)', linewidth=1.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Phase (rad)')
# plt.title('Phase Change Due to Chest Motion')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # ---------------------------------------------------

# # # --------------- Plot unwrapped phase change ---------------
# # plt.figure(figsize=(10, 4))
# # plt.plot(original_time, phi_t_unw, label='Chest Phase (unwrapped)', linewidth=1.5)
# # plt.xlabel('Time (s)')
# # plt.ylabel('Phase (rad)')
# # plt.title('Phase Change Due to Chest Motion')
# # plt.grid(True, linestyle='--', alpha=0.6)
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
# # # ---------------------------------------------------

# # ----------------- Plot path loss ---------------
# plt.figure(figsize=(10, 4))
# plt.plot(original_time, PL_dB, label='Chest Path Loss', color='blue', linewidth=1.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Path Loss (dB)')
# plt.title('Free-Space Path Loss vs Time')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # ---------------------------------------------------

# # --------------- Plot received power and amplitude ---------------
# plt.figure(figsize=(10, 4))
# plt.plot(original_time, pw_r_w, label='Received Power', color='green', linewidth=1.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Received Power (dBm)')
# plt.title('Received Power vs Time')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # ---------------------------------------------------

# # ----- plot tx signal-----
# plt.figure(figsize=(10, 4))
# plt.plot(sum_tx, label='TX Signal Amplitude', color='purple', linewidth=1.5)
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude (sqrt(W))')
# plt.title('Transmitted Signal Amplitude (First Symbol)')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # ------------------------

# # ------------------ Plot rx signal -----
# plt.figure(figsize=(10, 4))
# plt.plot(sum_rx, label='RX Signal Amplitude', color='brown', linewidth=1.5)
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude (sqrt(W))')
# plt.title('Received Signal Amplitude (First Symbol)')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # ------------------------

# # --------------- Plot fft(phase(hmax)) ---------------
# plt.figure(figsize=(10, 4))
# plt.plot(changes, label="periodic changes", color='purple', linewidth=1.5)
# plt.xlabel('f')
# plt.ylabel('Amplitude')
# plt.title('fft(phase(hmax))')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # ---------------------------------------------------


# end = time.perf_counter()
# elapsed_s = end - start
# print(f"Elapsed: {elapsed_s*1e3:.3f} ms")



# # Segment into symbols of length 2500
# X = reshape_for_subcarriers(sum_tx, sym_fs)  # (Nsym, 2500)

# # Time grid per symbol at the effective sampling rate
# fs_eff = data_fs * ups_factor
# n = np.arange(sym_fs)
# t = n / fs_eff  # seconds

# # Steering matrix for K subcarrier frequencies
# E = np.exp(-1j * 2*np.pi * np.outer(freqs, t))  # (K, 2500)

# # Per-symbol projections (DFT at chosen freqs)
# TXk = X @ E.T  # (Nsym, K)

# # Average magnitude and plot
# mag = np.mean(np.abs(TXk), axis=0)
# mag_db = 20*np.log10(mag + 1e-12)

# plt.figure(figsize=(10,5))
# plt.plot(freqs*1e-9, mag_db)
# plt.xlabel("Frequency [GHz]")
# plt.ylabel("Magnitude [dB]")
# plt.grid(True)
# plt.show()



