import numpy as np
from src.config import window_sec, step_sec

def sliding_windows(signal, Fs):
    """
    Generate overlapping windows.

    Returns:
        windows: list of signal segments
        times:   center time of each window
    """
    win_len = int(window_sec * Fs)
    step_len = int(step_sec * Fs)

    windows = []
    times = []

    for start in range(0, len(signal) - win_len, step_len):
        end = start + win_len
        windows.append(signal[start:end])
        times.append((start + end) / 2 / Fs)
    
    # print(times)
    return windows, np.array(times)
