"""
Cilia histogram classification, phase III.

This script reads in the kernel matrix created in the previous step, splits
it into subsets, and generates an ensemble model of SVMs.

It's not an ensemble in the formal sense; rather, each SVM learns on a slightly
different subset of the data (via cross-validation, no less) and returns predictions
on the held-out set. These predictions (and class probabilities) are retained
for each ROI; absolutely NO AVERAGING or combining is performed at this stage.

Consequently, if there are a lot of learners and a lot of iterations, these
results can become unwieldy. Go cautiously.
"""
import argparse
import os.path
import pickle

from joblib import Parallel, delayed
import numpy as np
import sklearn.cross_validation as cv
import sklearn.svm

from utils import read_labels

def _binary_labels(rois, labels):
    """
    Helper method that creates a target vector y from a list of ROIs.
    """
    return np.array([labels[r[:4]] for r in rois])

def ensemble(K, y, rois, n_learners):
    """
    Creates an ensemble of SVMs.

    Parameters
    ----------
    K : array, shape (N, N)
        Full pairwise kernel matrix.
    y : array, shape (N,)
        Target array.
    rois : list
        List of ROI names.
    n_learners : integer
        Number of learners to create (also corresponds to data splits).

    Returns
    -------
    results : dictionary
        Keyed first by ROI line ID (corresponding to the ordering), then by
        probability and prediction, this includes all the predictions made
        by the weak learners.
    """
    np.random.seed()
    indices = cv.StratifiedKFold(y, n_folds = n_learners, shuffle = True)

    # Train a whole bunch of weak learners!
    results = {}
    for train, test in indices:
        svm = sklearn.svm.NuSVC(nu = 0.35, kernel = 'precomputed', probability = True)
        Ktrain, Ktest = K[train][:, train], K[test][:, train]
        ytrain, ytest = y[train], y[test]
        svm.fit(Ktrain, ytrain)
        probas = svm.predict_proba(Ktest)[:, 1]  # Corresponds to the 1s confidences.
        y_pred = svm.predict(Ktest)

        for idx, pyi, yi in zip(test, probas, y_pred):
            key = rois[idx]
            if key not in results:
                results[key] = {'proba': [], 'pred': []}
            results[key]['proba'].append(pyi)
            results[key]['pred'].append(yi)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Cilia Histogram CV: Phase III',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python 3_kernel2ensemble.py <args>')
    parser.add_argument('-k', '--kernel', required = True,
        help = 'Path to .npy kernel matrix file from previous step.')
    parser.add_argument('-r', '--rois', required = True,
        help = 'Path to file listing the ROI names from the previous step.')
    parser.add_argument('-l', '--labels', required = True,
        help = 'Path to the text file containing patient labels.')
    parser.add_argument('-o', '--output', required = True,
        help = 'Path where output stuff will be written.')

    # Optional arguments.
    parser.add_argument('--numlearners', default = 100, type = int,
        help = 'Number of weak learners to create (also corresponds to splits of data). [DEFAULT: 100]')
    parser.add_argument('--iterations', default = 1000, type = int,
        help = 'Number of randomized cross-validation iterations each learner should perform. [DEFAULT: 1000]')

    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    # Read in the kernel, the ROI listings, and the ground-truth labels.
    K = np.load(args['kernel'])
    roilist = np.loadtxt(args['rois'], dtype = np.str,
        converters = {0: lambda s: str(s, encoding = 'utf-8')})
    sl, ll = read_labels(args['labels'])

    # Generate a vector of binary labels for each ROI.
    y = _binary_labels(roilist, sl)
    np.save(os.path.join(args['output'], "y.npy"), y)

    # Create a whole lot of weak learners that are trained over subsets of
    # the data.
    all_results = {}
    output = Parallel(n_jobs = -1, verbose = 10)(
        delayed(ensemble)(K, y, roilist, args['numlearners'])
        for i in range(args['iterations']))

    # Tally up everything.
    for d in output:
        # d is a dictionary unto itself.
        for k in d:
            if k not in all_results:
                all_results[k] = {'proba': [], 'pred': []}
            all_results[k]['proba'].extend(d[k]['proba'])
            all_results[k]['pred'].extend(d[k]['pred'])
    f = open(os.path.join(args['output'], "votes.pkl"), "wb")
    pickle.dump(all_results, f)
    f.close()
