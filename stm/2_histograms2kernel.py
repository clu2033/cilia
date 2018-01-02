"""
Cilia histogram classification, phase II.

This script takes the frequency and magnitude histograms (output from previous
script) and processes them into a pairwise kernels for SVM classification.

The kernel is constructed the same way as in the STM methods. The histograms
of each type are compared via the chi-squared pairwise metric (i.e., chi-squared
distance between the RMPs of two ROIs, repeated for each of the four histograms),
and the four resulting chi-squared distances are combined in a weighted sum.

This weighted sum represents the pairwise distance between the two ROIs.
"""
import argparse
import pickle
import os.path

import numpy as np
import sklearn.metrics.pairwise as pairwise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Cilia Histogram CV: Phase II',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python 2_histograms2kernels.py <args>')
    parser.add_argument('-i', '--input', required = True,
        help = 'Path to pickle file of histograms from previous step.')
    parser.add_argument('-f', '--files', required = True,
        help = 'List of files of interest.')
    parser.add_argument('-o', '--output', required = True,
        help = 'Path where output stuff will be written.')

    # Optional arguments.
    parser.add_argument("weights", nargs = 4, type = float,
        help = "Pairs of weights for (RMP, RFP, DMP, DFP). Must sum to 1.")

    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    # Read in the dictionary file.
    f = open(args['input'], "rb")
    histograms = pickle.load(f)
    f.close()

    # Generate the pairwise kernel, using chi-squared distance.
    w = np.array(args['weights']) / np.sum(args['weights'])
    N = len(histograms.keys())
    K = np.zeros(shape = (N, N))

    # Weight each type of histogram.
    roinames = list(histograms.keys())
    for key, weight in zip(['rmp', 'rfp', 'dmp', 'dfp'], w):

        # Loop through all the keys in the dictionary, pulling out all
        # the histograms of that type.
        vectors = np.array([histograms[roiname][key] for roiname in histograms])
        chi2 = pairwise.chi2_kernel(vectors, vectors)
        mu = 1.0 / chi2.mean()
        K += (weight * np.exp(-mu * chi2))

    # ...yup, that's pretty much it.
    # We do need to save the list of ROI names, since that indicates the ordering
    # in the pairwise kernel matrix K. We'll need this in future steps.
    np.save(os.path.join(args['output'], 'K.npy'), K)
    np.savetxt(os.path.join(args['output'], 'roinames.txt'), roinames, fmt = "%s")
