import argparse
import glob
import os.path

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy.ndimage as spnd
import skimage.io as skio

from utils import read_labels
import extract_patches as EP

def read_patch_data(patchpath):
    patch_data = {}
    for item in glob.glob(os.path.join(patchpath, "*.npy")):
        data = np.load(item)
        fname = item.split("/")[-1].split(".")[0]
        patch_data[fname] = data
    return patch_data

def build_folds(foldpath, patches):
    fold_ids = np.unique(np.array([int(item.split("/")[-1].split(".")[0].split("_")[0][-1]) for item in glob.glob(os.path.join(foldpath, "*.txt"))], dtype = np.int))
    patch_data = {}
    for id_num in fold_ids:
        # Which patients are we working on?
        patient_ids = parse_fold_header(os.path.join(foldpath, "fold{}_header.txt".format(id_num)))
        videos = parse_p2v(os.path.join(foldpath, "fold{}_p2v.txt".format(id_num)), patient_ids)
        predictions = np.loadtxt(os.path.join(foldpath, "fold{}.txt".format(id_num)), dtype = np.int)

        # Assuming everything is in the right order, should be a simple matter
        # of indexing against the predictions made for each patch.
        patch_index = 0
        for vid in videos:
            num_patches = patches[vid].shape[0]
            patch_data[vid] = predictions[patch_index:(patch_index + num_patches)]
            patch_index += num_patches
    return patch_data

def parse_fold_header(header_file):
    with open(header_file, "r") as f:
        s = f.read()
    return [elem.split("_")[-1] for elem in s.split(",")]

def parse_p2v(p2v_file, patient_ids):
    videos = []
    with open(p2v_file, "r") as f:
        for line in f:
            patient_name, *patient_vids = line.split()
            pid = patient_name.split("_")[-1]
            if pid not in patient_ids:
                print("{} found in p2v, but not header file!".format(pid))
                continue
            videos.extend(patient_vids)
    return videos

def run_mod_patch_selection(vid, yp, labels, optargs):
    mpath, gpath, CILIA, CELL, BG, patchsize, overlap, num_frames, outpath = optargs
    true_label = labels[vid[:4]]
    
    # Repeat... most of the things from before.
    mask_array = np.array(np.load(os.path.join(mpath, "{}_prediction.npy".format(vid))), dtype = np.uint8)
    gray_video = np.load(os.path.join(gpath, "{}.npy".format(vid)))
    mask = EP.resize_mask(mask_array, gray_video.shape)
    mask = EP.preprocess(mask, patchsize, (CILIA, BG))
    mask[mask != CILIA] = BG
    seed_indices = EP.choose_seeds(mask, overlap, patchsize)
    if seed_indices is None:
        return [vid, False, 0]
    if yp.shape[0] != seed_indices.shape[0]:
        return [vid, False, yp.shape[0] - seed_indices.shape[0]]

    annotated_frame = color_patches(seed_indices, yp, true_label, patchsize, gray_video[0])

    # Write them out.
    out_frame = os.path.join(outpath, "{}.png".format(vid))
    skio.imsave(out_frame, annotated_frame)

    # All done. Signal success.
    return [vid, True, seed_indices.shape[0]]

def color_patches(seeds, yp, yt, patchsize, gray_frame):
    rgba = np.zeros(shape = (gray_frame.shape[0], gray_frame.shape[1], 3), dtype = np.uint8)
    rgba[:, :, 0] = np.copy(gray_frame)
    rgba[:, :, 1] = np.copy(gray_frame)
    rgba[:, :, 2] = np.copy(gray_frame)

    for i, ([r, c], yi) in enumerate(zip(seeds, yp)):
        r_corner = r - int(patchsize / 2)
        c_corner = c - int(patchsize / 2)

        # Make the edges black.
        rgba[r_corner:(r_corner + patchsize), c_corner] = 0
        rgba[r_corner:(r_corner + patchsize), c_corner + patchsize - 1] = 0
        rgba[r_corner, c_corner:(c_corner + patchsize)] = 0
        rgba[r_corner + patchsize - 1, c_corner:(c_corner + patchsize)] = 0

        # Fill in with a color
        if yt == 0:  # Ground-truth is normal, use blues.
            if yi == 1:  # Normal predicted to be abnormal--FP, light blue
                color = (135, 206, 250)
            else:  # Normal predicted to be normal--TP, dark blue
                color = (0, 0, 255)
        else: # Ground-truth is abnormal, use reds.
            if yi == 0:  # Abnormal predicted to be normal--FN, orange
                color = (255, 165, 0)
            else:  # Abnormal predicted to be abnormal--TN, red
                color = (255, 0, 0)
        startcol = c_corner + 1
        endcol = c_corner + patchsize - 1
        for index in range(1, patchsize - 1):
            row = r_corner + index
            rgba[row, startcol:endcol] = color

    return rgba

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Cilia DL',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python predictions_to_patches.py <args>')
    parser.add_argument('-im', '--input_masks', required = True,
        help = 'Path to a folder containing mask files.')
    parser.add_argument("-if", "--input_folds", required = True,
        help = "Path to a folder containing fold text files.")
    parser.add_argument("-ig", "--input_gray", required = True,
        help = "Path to a folder containing grayscale video data.")
    parser.add_argument("-ip", "--input_patches", required = True,
        help = "Path to a folder containing all the patch data.")
    parser.add_argument("-l", "--labelfile", required = True,
        help = "Path to a label file.")
    parser.add_argument('-o', '--output', required = True,
        help = 'Path where output stuff will be written.')

    # Optional arguments.
    parser.add_argument("--patchsize", type = int, default = 11,
        help = "Square dimensions of the patches to be extracted. [DEFAULT: 11]")
    parser.add_argument("--random_state", type = int, default = -1,
        help = "Sets the random state for deterministic patch selection. [DEFAULT: -1]")
    parser.add_argument("--overlap", type = float, default = 0.25,
        help = "Controls the amount of allowed overlap of patches; 0 is none, 1 is complete overlap allowed. [DEFAULT: 0.25]")
    parser.add_argument("--num_frames", type = int, default = 250,
        help = "Number of frames for the patches to have. [DEFAULT: 250]")

    # Mask indexing arguments.
    parser.add_argument("--cilia_mask", type = int, default = 2,
        help = "Integer index corresponding to cilia in the mask. [DEFAULT: 2]")
    parser.add_argument("--cell_mask", type = int, default = 1,
        help = "Integer index corresponding to the cell body in the mask. [DEFAULT: 1]")
    parser.add_argument("--bg_mask", type = int, default = 0,
        help = "Integer index corresponding to the background in the mask. [DEFAULT: 0]")

    args = vars(parser.parse_args())

    if args['random_state'] != -1:
        np.random.seed(args['random_state'])

    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    labels, _ = read_labels(args['labelfile'])
    ### Step 1: Read in all the patch files that were run through the LSTM
    # predictor.
    patch_data = read_patch_data(args['input_patches'])

    ### Step 2: Align the fold data with the patches, so we know what the
    # prediction of each patch was.
    patch_predictions = build_folds(args['input_folds'], patch_data)

    ### Step 3: Rebuild the patch selection process, but jam the patch labels
    # into it and color each patch as to its FP/FN/TP/TN status.
    optargs = ([args['input_masks'], args['input_gray'], args['cilia_mask'], 
                args['cell_mask'], args['bg_mask'], args['patchsize'],
                args['overlap'], args['num_frames'], args['output']])
    patches = Parallel(n_jobs = -1, verbose = 10)(
        delayed(run_mod_patch_selection)(vid, yp, labels, optargs) for (vid, yp) in patch_predictions.items())

    # Go through the results.
    failures = {}
    for fname, outcome, count in patches:
        if not outcome:
            if count == 0: 
                continue
            else:
                failures[fname] = count

    for fname, count in failures.items():
        print("Patch count mismatch in {}: {}".format(fname, count))
