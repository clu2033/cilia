"""
Cilia histogram classification, phase I.

This script takes the raw data (in numpy array format) and processes it into
frequency and magnitude histograms for each ROI.

For the frequency histograms, 1D-FFT is computed at each pixel and a heatmap
formed where each pixel is represented by its dominant frequency. A median kernel
is used to smooth this heatmap. These frequencies are then placed into a histogram.

For the magnitude histograms, the temporal signatures are smoothed with a 1D
gaussian filter and then placed into histograms.

In both cases, only the top number of pixels in an ROI (in terms of standard
deviation of motion) are retained and put in histograms.
"""

import argparse
import os.path
import pickle

from joblib import Parallel, delayed
import numpy as np
import scipy.ndimage.filters as filters

import properties as histprops
import histogram as histapi

def create_histograms(roiname, optargs):
    """
    Main method for reading in the raw numpy files and making histograms.
    """
    # Extract all the variables.
    idir, rdir, ddir, frames, sigma, mpbins, fpbins, freq, kernel, pixels, fps = optargs

    # Read the ROI numpy files.
    iroi = np.load(os.path.join(idir, roiname))
    rroi = np.load(os.path.join(rdir, roiname))
    droi = np.load(os.path.join(ddir, roiname))

    # Convert the deformation to absolute magnitudes, and truncate rotation
    # and deformation at the specified number of frames.
    droi = np.sqrt((droi[0] ** 2) + (droi[1] ** 2))[:frames]
    rroi = rroi[:frames]

    # Threshold the intensity video in terms of relative pixel-based standard
    # deviation. This will allow us to pick pixels that, relative to the ROI,
    # show the most motion. After computing the needed statistics for frequency
    # and magnitude histograms, these pixel positions will be the only values
    # retained and used to build the histograms. This is to guarantee all the
    # histograms are built using the same amount of data, but while still
    # capturing the dynamics that are unique to each ROI.
    pixel_mask = histprops.prune(iroi, pixels)

    # Compute the frequency maps and histograms.
    r_fmap = histprops.cbfmap(rroi, fps, freq, kernel)
    d_fmap = histprops.cbfmap(droi, fps, freq, kernel)
    rfp = histapi.fp(r_fmap, fpbins, pixel_mask)
    dfp = histapi.fp(d_fmap, fpbins, pixel_mask)

    # Compute the magnitude histograms.
    s_rroi = filters.gaussian_filter1d(rroi, sigma, axis = 0)
    s_droi = filters.gaussian_filter1d(droi, sigma, axis = 0)
    rmp = histapi.mp(s_rroi, mpbins, pixel_mask, (-0.3, 0.3))
    dmp = histapi.mp(s_droi, mpbins, pixel_mask, (0.0, 1.0))

    # All done!
    return [roiname, rmp, dmp, rfp, dfp]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Cilia Histogram CV: Phase I',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python 1_npy2histograms.py <args>')
    parser.add_argument('-i1', '--input1', required = True,
        help = 'Path to INTENSITY movies.')
    parser.add_argument('-i2', '--input2', required = True,
        help = 'Path to ROTATION movies.')
    parser.add_argument('-i3', '--input3', required = True,
        help = 'Path to DEFORMATION movies.')
    parser.add_argument('-f', '--files', required = True,
        help = 'List of files of interest.')
    parser.add_argument('-o', '--output', required = True,
        help = 'Path where output stuff will be written.')

    # Optional arguments.
    parser.add_argument('--pixels', default = 1000, type = int,
        help = 'Number of pixels to retain per ROI to compute features. [DEFAULT: 1000]')
    parser.add_argument('--framerate', default = 200, type = int,
        help = 'Framerate of the video data. [DEFAULT: 200]')
    parser.add_argument('--frames', default = 250, type = int,
        help = 'Maximum number of frames to retain for creating and thresholding profiles. [DEFAULT: 250]')
    parser.add_argument('--maxfreq', default = 20.0, type = float,
        help = 'Maximum allowable frequency when computing *FPs. [DEFAULT: 20.0]')
    parser.add_argument('--kernel', default = 5, choices = [3, 5, 7, 9],
        help = 'Median filter size for spatial smoothing of *FPs. [DEFAULT: 5]')
    parser.add_argument('--mpbins', default = 100, type = int,
        help = 'Number of bins to use when creating the *MPs. [DEFAULT: 100]')
    parser.add_argument('--fpbins', default = 20, type = int,
        help = 'Number of bins to use for *FPs. [DEFAULT: 20]')
    parser.add_argument('--sigma', default = 2.5, type = float,
        help = 'Sigma value for temporal smoothing for *MPs. [DEFAULT: 2.5]')

    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    # Read the list of files. These are ROI names, one name per ROI.
    files = np.loadtxt(args["files"], dtype = np.str,
        converters = {0: lambda s: str(s, encoding = 'utf-8')}).tolist()
    optargs = ([args['input1'], args['input2'], args['input3'], args['frames'],
                args['sigma'], args['mpbins'], args['fpbins'], args['maxfreq'],
                args['kernel'], args['pixels'], args['framerate']])

    # This job reads in all the ROIs, performs the specified operations,
    # and computes the requisite histograms for each ROI (4 per ROI).
    retvals = Parallel(n_jobs = -1, verbose = 10)(
        delayed(create_histograms)(f, optargs) for f in files)

    # Build a dictionary.
    outdict = {}
    for roiname, rmp, dmp, rfp, dfp in retvals:
        outdict[roiname] = {'rmp': rmp, 'dmp': dmp, 'rfp': rfp, 'dfp': dfp}

    # Serialize and quit!
    f = open(os.path.join(args['output'], "histograms.pkl"), "wb")
    pickle.dump(outdict, f)
    f.close()
