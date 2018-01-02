import numpy as np
import scipy.signal as signal

def prune(video, n_pixels):
    """
    Creates a mask based on the standard deviation of the pixel intensity
    variance, discarding all pixels whose standard deviations are less than
    those of the highest n_pixels.

    Parameters
    ----------
    video : array, shape (F, H, W)
        NumPy video array.
    n_pixels : integer
        Number of pixels to retain. All others are filtered out.

    Returns
    -------
    mask : array, shape (H, W)
        Binary mask over the spatial dimensions of the video, selecting
        the pixels to keep.
    """
    # Compute the pixel intensity variance at each grid location.
    stds = video.std(axis = 0)
    if n_pixels > stds.flatten().shape[0]: n_pixels = -1

    # Find the n_pixel'th standard deviation.
    threshold = np.sort(stds.flatten())[::-1][n_pixels]

    # Create a mask where everything is False except for those pixels which
    # satisfy the threshold.
    return stds > threshold

def cbfmap(video, fps, max_freq, kernel):
    """
    Calculates the ciliary beat frequency (CBF) map for a video.

    Parameters
    ----------
    video : array, shape (F, H, W)
        3D matrix with H rows, W columns, and F frames.
    fps : int
        Frames per second of the video.
    max_freq : float
        Maximum allowed frequency (clamps all frequencies that are greater).
    kernel : integer
        Size of the median 2D filter kernel.

    Returns
    -------
    cbfmap : array, shape (H, W)
        Array of dominant frequencies, one for each pixel position.
    """
    nfft = _nextpow2(video.shape[0])
    nperseg = int(nfft / 2)
    f, Pxx = signal.welch(video, axis = 0, fs = fps, window = ('gaussian', 10),
        nperseg = nperseg, nfft = nfft)

    # Find the dominant frequency at each pixel location.
    heatmap = f[Pxx.argmax(axis = 0)]

    # Clamp frequencies that are excessively large, and run a 2D median
    # filter over the resulting map to clean up the frequencies a bit.
    heatmap[heatmap > max_freq] = max_freq
    heatmap = signal.medfilt2d(heatmap, kernel_size = kernel)

    # All done.
    return heatmap

def _nextpow2(i):
    """
    Calculates the next even power of 2 which is greater than or equal to i.

    Parameters
    ----------
    i : integer
        The number which we want a power of 2 that is greater.

    Returns
    -------
    j : integer
        A power of 2, where j >= i.
    """
    n = 2
    while n < i:
        n *= 2
    return n
