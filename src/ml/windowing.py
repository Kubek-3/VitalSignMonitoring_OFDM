import numpy as np

def sliding_windows(signal, Fs, win_sec=10.0, step_sec=2.0):
    """
    Generate overlapping windows.

    Returns:
        windows: list of signal segments
        times:   center time of each window
    """
    win_len = int(win_sec * Fs)
    step_len = int(step_sec * Fs)

    windows = []
    times = []

    for start in range(0, len(signal) - win_len, step_len):
        end = start + win_len
        windows.append(signal[start:end])
        times.append((start + end) / 2 / Fs)

    return windows, np.array(times)
