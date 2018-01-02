"""
Cilia histogram classification, phase IV.

This script reads in the binary classification results and predictions from the
previous step and uses it to perform 4-way meta-classification.

It relies on the uncertainty in the binary classification performed in the
previous step. This can take a couple of forms:
 - changing predictions (0s and 1s)
 - varying confidences in the predictions (middling probabilities)

This script combines that information into small feature vectors that are used
to classify the patients themselves into one of four categories.
"""
import argparse
import os.path
import pickle

from joblib import Parallel, delayed
import numpy as np
import sklearn.cross_validation as cv
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
import sklearn.svm as svm

from utils import read_labels

def process_votes(patient_id, probs, preds):
    """
    Utility method for generating feature vectors from probabilities
    and predictions.
    """
    prob_avg = np.mean(probs)
    prob_std = np.std(probs)
    pred_avg = np.mean(preds)
    return [patient_id, np.array([prob_avg, prob_std, pred_avg])]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Cilia Histogram CV: Phase IV',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python 4_classifyensemble.py <args>')
    parser.add_argument('-i', '--input', required = True,
        help = 'Path to file listing the ROI names from the previous (previous) step.')
    parser.add_argument('-l', '--labels', required = True,
        help = 'Path to the text file containing patient labels.')
    parser.add_argument('-o', '--output', required = True,
        help = 'Path where output stuff will be written.')

    # Optional arguments.
    parser.add_argument('--trees', type = int, default = 42,
        help = 'Number of trees to use in the random forest classifier. [DEFAULT: 42]')
    parser.add_argument('--neighbors', type = int, default = 5,
        help = 'Size of neighborhood in KNN classifier. [DEFAULT: 5]')
    parser.add_argument('--folds', type = int, default = 4,
        help = 'Number of folds for cross-validation. [DEFAULT: 4]')
    parser.add_argument('--iterations', type = int, default = 1000,
        help = 'Number of randomized iterations of cross-validation to perform. [DEFAULT: 1000]')

    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    # Read in the prior predictions.
    f = open(args['input'], "rb")
    prev = pickle.load(f)
    f.close()
    sl, ll = read_labels(args['labels'])

    # Merge all the votes having to do with the same patient.
    keys = list(prev.keys())
    patients = {}
    for k in keys:
        patient = k[:4]
        if patient not in patients:
            patients[patient] = {'proba': [], 'pred': []}
        patients[patient]['proba'].extend(prev[k]['proba'])
        patients[patient]['pred'].extend(prev[k]['pred'])

    # Generate feature vectors for each patient.
    N = len(patients.keys())
    X = np.zeros(shape = (N, 3))
    y = np.zeros(N)
    pids = []
    vectors = Parallel(n_jobs = -1, verbose = 10)(
        delayed(process_votes)(pid, patients[pid]['proba'], patients[pid]['pred'])
        for pid in patients)
    for i, (pid, v) in enumerate(vectors):
        X[i] = v
        y[i] = ll[pid]
        pids.append(pid)
    np.save(os.path.join(args['output'], "X.npy"), X)
    np.save(os.path.join(args['output'], "y.npy"), y)
    np.savetxt(os.path.join(args['output'], "patients.txt"), pids, fmt = "%s")

    # Perform cross-validation to get a four-way accuracy!
    accuracies = {'svm': [], 'rf': [], 'knn': []}
    for i in range(args['iterations']):
        indices = cv.StratifiedKFold(y, n_folds = args['folds'], shuffle = True)
        for train, test in indices:
            Xtrain, Xtest = X[train], X[test]
            ytrain, ytest = y[train], y[test]

            # SVM
            s = svm.LinearSVC()
            s.fit(Xtrain, ytrain)
            accuracies['svm'].append(s.score(Xtest, ytest))

            # RF
            rf = ensemble.RandomForestClassifier(n_estimators = args['trees'], n_jobs = -1)
            rf.fit(Xtrain, ytrain)
            accuracies['rf'].append(rf.score(Xtest, ytest))

            # KNN
            knn = neighbors.KNeighborsClassifier(n_neighbors = args['neighbors'])
            knn.fit(Xtrain, ytrain)
            accuracies['knn'].append(knn.score(Xtest, ytest))

        if i % 100 == 0:
            print(i)

    # Whew, all done.
    print('SVM: {:.6f} (+/- {:.6f})'.format(np.mean(accuracies['svm']), np.std(accuracies['svm'])))
    print('RF: {:.6f} (+/- {:.6f})'.format(np.mean(accuracies['rf']), np.std(accuracies['rf'])))
    print('KNN: {:.6f} (+/- {:.6f})'.format(np.mean(accuracies['knn']), np.std(accuracies['knn'])))
