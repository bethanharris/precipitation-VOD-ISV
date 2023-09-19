from scipy.signal import butter, sosfiltfilt
import numpy as np
import pandas as pd


def butter_lowpass_filter(data, lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, analog=False, btype='low', output='sos')
    y = sosfiltfilt(sos, data)
    return y


def butter_highpass_filter(data, highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, analog=False, btype='high', output='sos')
    y = sosfiltfilt(sos, data)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfiltfilt(sos, data)
    return y


def bandpass_filter_missing_data(data, freq_low, freq_high, fs, order=5, min_slice_size=30):
    filtered_data = np.ones_like(data)*np.nan
    missing_data = np.isnan(data)
    if not np.any(missing_data):
        return butter_bandpass_filter(data, freq_low, freq_high, fs)
    change_missing = np.diff(missing_data.astype(float))
    start_valid = (np.where(change_missing == -1)[0] + 1).tolist()
    end_valid = (np.where(change_missing == 1)[0] + 1).tolist()
    if not missing_data[0]:
        start_valid.insert(0, 0)
    if len(end_valid) == len(start_valid) - 1:
        end_valid.append(None)
    valid_data_slices = zip(start_valid, end_valid)
    for start, end in valid_data_slices:
        slice_size = (data.size - start) if end is None else (end - start)
        if slice_size > min_slice_size+3:
            filtered_data[start:end] = butter_bandpass_filter(data[start:end], freq_low, freq_high, fs)
    return filtered_data


def lowpass_filter_missing_data(data, freq_low, fs, order=5, min_slice_size=30):
    filtered_data = np.ones_like(data)*np.nan
    missing_data = np.isnan(data)
    if not np.any(missing_data):
        return butter_lowpass_filter(data, freq_low, fs)
    change_missing = np.diff(missing_data.astype(float))
    start_valid = (np.where(change_missing == -1)[0] + 1).tolist()
    end_valid = (np.where(change_missing == 1)[0] + 1).tolist()
    if not missing_data[0]:
        start_valid.insert(0, 0)
    if len(end_valid) == len(start_valid) - 1:
        end_valid.append(None)
    valid_data_slices = zip(start_valid, end_valid)
    for start, end in valid_data_slices:
        slice_size = (data.size - start) if end is None else (end - start)
        if slice_size > min_slice_size+3:
            filtered_data[start:end] = butter_lowpass_filter(data[start:end], freq_low, fs)
    return filtered_data


def highpass_filter_missing_data(data, freq_high, fs, order=5, min_slice_size=30):
    filtered_data = np.ones_like(data)*np.nan
    missing_data = np.isnan(data)
    if not np.any(missing_data):
        return butter_highpass_filter(data, freq_high, fs)
    change_missing = np.diff(missing_data.astype(float))
    start_valid = (np.where(change_missing == -1)[0] + 1).tolist()
    end_valid = (np.where(change_missing == 1)[0] + 1).tolist()
    if not missing_data[0]:
        start_valid.insert(0, 0)
    if len(end_valid) == len(start_valid) - 1:
        end_valid.append(None)
    valid_data_slices = zip(start_valid, end_valid)
    for start, end in valid_data_slices:
        slice_size = (data.size - start) if end is None else (end - start)
        if slice_size > min_slice_size+3:
            filtered_data[start:end] = butter_highpass_filter(data[start:end], freq_high, fs)
    return filtered_data


def lanczos_lowpass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.
    Args:
    window: int
        The length of the filter window.
    cutoff: float
        The cutoff frequency in inverse time steps.
    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]


def lanczos_bandpass_weights(window, low_freq, high_freq):
    """Calculate weights for a low pass Lanczos filter.
    Args:
    window: int
        The length of the filter window.
    cutoff: float
        The cutoff frequency in inverse time steps.
    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * (high_freq - low_freq)
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * low_freq * k) / (np.pi * k)
    secondfactor = np.sin(2. * np.pi * high_freq * k) / (np.pi * k)
    w[n-1:0:-1] = (secondfactor-firstfactor) * sigma
    w[n+1:-1] = (secondfactor-firstfactor) * sigma
    #w[5:11] += 0.02*np.sin(np.pi*np.linspace(0,1,6))
    return w[1:-1]


def lanczos_highpass_weights(window, cutoff):
    """Calculate weights for a high pass Lanczos filter.
    Args:
    window: int
        The length of the filter window.
    cutoff: float
        The cutoff frequency in inverse time steps.
    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 1-(2 * cutoff)
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = -firstfactor * sigma
    w[n+1:-1] = -firstfactor * sigma
    return w[1:-1]


def lanczos_lowpass_filter(data, freq_low, window=121):
    weights = lanczos_lowpass_weights(window, freq_low)
    filtered_data = data.rolling_window('time', iris.analysis.MEAN, len(weights), weights=weights)
    _Rolling_and_Expanding.weighted_mean = weighted_mean
    filtered_data = np.array(pd.DataFrame(data).rolling(window=window).weighted_mean(weights).values.ravel().tolist())
    return filtered_data


def lanczos_filter_missing_data(data, freq_low, freq_high, window=61, min_slice_size=100):
    side_buffer = (window-1)//2
    filtered_data = np.ones_like(data)*np.nan
    missing_data = np.isnan(data)
    lanczos_weights = lanczos_bandpass_weights(window, freq_low, freq_high)
    kernel = lanczos_weights
    if not np.any(missing_data):
        filtered_data[side_buffer:-side_buffer] = np.convolve(data, kernel, mode='valid')
    elif np.all(missing_data):
        pass
    else:
        change_missing = np.diff(missing_data.astype(float))
        start_valid = (np.where(change_missing == -1)[0] + 1).tolist()
        end_valid = (np.where(change_missing == 1)[0] + 1).tolist()
        if not missing_data[0]:
            start_valid.insert(0, 0)
        if len(end_valid) == len(start_valid) - 1:
            end_valid.append(data.size)
        valid_data_slices = zip(start_valid, end_valid)
        for start, end in valid_data_slices:
            if (end - start) >= min_slice_size:
                filtered_data[start+side_buffer:end-side_buffer] = np.convolve(data[start:end], kernel, mode='valid')
    return filtered_data


def lanczos_lowpass_filter_missing_data(data, freq_high, window=61, min_slice_size=100):
    side_buffer = (window-1)//2
    filtered_data = np.ones_like(data)*np.nan
    missing_data = np.isnan(data)
    lanczos_weights = lanczos_lowpass_weights(window, freq_high)
    kernel = lanczos_weights
    if not np.any(missing_data):
        filtered_data[side_buffer:-side_buffer] = np.convolve(data, kernel, mode='valid')
    else:
        change_missing = np.diff(missing_data.astype(float))
        start_valid = (np.where(change_missing == -1)[0] + 1).tolist()
        end_valid = (np.where(change_missing == 1)[0] + 1).tolist()
        if not missing_data[0]:
            start_valid.insert(0, 0)
        if len(end_valid) == len(start_valid) - 1:
            end_valid.append(data.size)
        valid_data_slices = zip(start_valid, end_valid)
        for start, end in valid_data_slices:
            if (end - start) >= min_slice_size:
                filtered_data[start+side_buffer:end-side_buffer] = np.convolve(data[start:end], kernel, mode='valid')
    return filtered_data


def lanczos_highpass_filter_missing_data(data, freq_low, window=61, min_slice_size=100):
    side_buffer = (window-1)//2
    filtered_data = np.ones_like(data)*np.nan
    missing_data = np.isnan(data)
    lanczos_weights = lanczos_highpass_weights(window, freq_low)
    kernel = lanczos_weights
    if not np.any(missing_data):
        filtered_data[side_buffer:-side_buffer] = np.convolve(data, kernel, mode='valid')
    else:
        change_missing = np.diff(missing_data.astype(float))
        start_valid = (np.where(change_missing == -1)[0] + 1).tolist()
        end_valid = (np.where(change_missing == 1)[0] + 1).tolist()
        if not missing_data[0]:
            start_valid.insert(0, 0)
        if len(end_valid) == len(start_valid) - 1:
            end_valid.append(data.size)
        valid_data_slices = zip(start_valid, end_valid)
        for start, end in valid_data_slices:
            if (end - start) >= min_slice_size:
                filtered_data[start+side_buffer:end-side_buffer] = np.convolve(data[start:end], kernel, mode='valid')
    return filtered_data
