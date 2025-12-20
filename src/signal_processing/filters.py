from scipy.signal import butter, filtfilt

def bp_filter(sig, fs, f_low, f_high, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [f_low/nyq, f_high/nyq], btype='band')
    return filtfilt(b, a, sig)

def lowpass_filter(sig, fs, f_cut, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, f_cut / nyq, btype='low')
    return filtfilt(b, a, sig)

