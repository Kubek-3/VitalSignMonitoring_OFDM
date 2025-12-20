import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

def load_chest_motion(file_path):

    """Load chest surface motion data from .mat file and preprocess it.
    Returns:
        1D array: Preprocessed chest displacement in meters."""
    
    # mat = sio.loadmat('Multimodal chest surface motion\Free_T2'
    # '.mat')
    mat = sio.loadmat(file_path)
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
        disp_m = disp_m * 1e-3     # unit is [mm]
    disp_m = disp_m - np.mean(disp_m)   # remove DC
    return disp_m


def upsample_signal(first_samples, signal, data_fs, factor):

    """Upsample the input signal by the given factor using cubic interpolation.
    Args:
        first_samples (int): Number of initial samples to consider from the input signal.
        signal (1D array): Input signal to be upsampled.
        factor (int): Upsampling factor.
    Returns:
        1D array: Upsampled signal."""
    
    # Given: disp_m (1D array of length N), sampled at Fs = 100 Hz
    dt = 1.0 / data_fs
    #short_disp = signal[:first_samples] # use first N samples for testing
    short_disp = signal
    N = len(short_disp)  # use first 500 samples for testing
    t = np.arange(N) * dt # Original time grid
    t_dense = np.linspace(t[0], t[-1], factor*N, endpoint=True) # New dense time grid
    f_cub = interp1d(t, short_disp, kind='cubic',  bounds_error=False, fill_value='extrapolate')
    new_disp_m = f_cub(t_dense)
    #print("new shape:", new_disp_m.shape)
    return new_disp_m

def dist(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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
    t_disp = disp + chest_pos_x
    d_tx = dist(t_disp, chest_pos_y, xtx, ytx)
    d_rx = dist(t_disp, chest_pos_y, xrx, yrx)
    d_tot = d_tx + d_rx                         # total path length
    #print("Max, min path length change (m):", np.max(d_tot), np.min(d_tot))
    return d_tot
