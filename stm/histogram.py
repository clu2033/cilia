import numpy as np

def fp(freqmap, bins, mask):
    """
    Creates a frequency histogram using the frequency map.

    Parameters
    ----------
    freqmap : array, shape (H, W)
        Spatial map of dominant frequencies.
    bins : integer
        Number of bins the histogram should have.
    mask : array, shape (H, W)
        Binary mask to indicate which pixels to keep.

    Returns
    -------
    hist : array, shape (bins,)
        Frequency histogram.
    """
    pixels = freqmap[mask]
    return _normalized_histogram(pixels, bins)

def mp(roi, bins, mask, range):
    """
    Creates a magnitude histogram using the ROI data.

    Parameters
    ----------
    roi : array, shape (F, H, W)
        ROI data.
    bins : integer
        Number of bins the histogram should have.
    mask : array, shape (H, W)
        Binary mask to indicate which pixels to keep.
    range : tuple (float, float)
        Minimum and maximum values allowed in the histogram.

    Returns
    -------
    hist : array, shape (bins,)
        Magnitude histogram.
    """
    trajectories = roi[:, mask]
    return _normalized_histogram(trajectories, bins, range)

def _normalized_histogram(data, bins, range = None):
    """
    Helper method so we don't duplicate code.
    """
    hist = 0
    edges = 0
    if range is None:
        hist, edges = np.histogram(data, bins = bins)
    else:
        hist, edges = np.histogram(data, bins = bins, range = range)
    hist = np.array(hist, dtype = np.float)
    hist /= hist.sum()
    return hist
