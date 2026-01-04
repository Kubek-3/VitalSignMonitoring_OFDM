import numpy as np
from src.config import WINDOW_SEC, STEP_SEC

def sliding_windows(signal, Fs):
    """
    Generate overlapping windows.

    Returns:
        windows: list of signal segments
        times:   center time of each window
    """
    win_len = int(WINDOW_SEC * Fs)
    step_len = int(STEP_SEC * Fs)

    windows = []
    times = []

    for start in range(0, len(signal) - win_len, step_len):
        end = start + win_len
        windows.append(signal[start:end])
        times.append((start + end) / 2 / Fs)
    
    # print(times)
    return windows, np.array(times)
