import argparse
import glob
import os.path

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy.ndimage as spnd
import skimage.io as skio
import skimage.measure as skms
import skimage.morphology as skmp

def run_patch_selection(fname, optargs):
    mpath, rpath, gpath, CILIA, CELL, BG, patchsize, overlap, n_frames, outpath = optargs

    # Load the three NumPy arrays.
    mask_array = np.array(np.load(os.path.join(mpath, "{}_prediction.npy".format(fname))), dtype = np.uint8)
    curl_video = np.load(os.path.join(rpath, "{}.npy".format(fname)))
    gray_video = np.load(os.path.join(gpath, "{}.npy".format(fname)))
    
    # Resize the mask.
    mask = resize_mask(mask_array, gray_video.shape)

    # Do some preprocessing on the mask.
    mask = preprocess(mask, patchsize, (CILIA, BG))

    # Find the seeds for the patches.
    mask[mask != CILIA] = BG
    dist = spnd.distance_transform_edt(mask)
    seed_indices = choose_seeds(mask, overlap, patchsize)
    if seed_indices is None:
        return [fname, False, 0]

    # Extract the patches.
    list_of_patches, frame_patches = extract_patches(
        curl_video, seed_indices, patchsize, n_frames, gray_frame = gray_video[0])

    # Write them out.
    out_patches = os.path.join(outpath, "{}.npy".format(fname))
    out_frame = os.path.join(outpath, "{}.png".format(fname))
    np.save(out_patches, list_of_patches)
    skio.imsave(out_frame, frame_patches)

    # All done. Signal success.
    return [fname, True, list_of_patches.shape[0]]

def resize_mask(mask, dims):
    img = PIL.Image.fromarray(mask, mode = "L")
    resized = np.asarray(img.resize((dims[2], dims[1]), resample = PIL.Image.NEAREST))
    return np.copy(resized)

def preprocess(mask, patchsize, mask_indices):
    """
    Helper function that "cleans up" the mask a bit after it's been resized.
    There are usually a lot of small pixels that can safely be removed.

    Parameters
    ----------
    mask : array, shape (H, W)
        The video mask, has the same spatial dimensions as the video.
    patchsize: integer
        Square dimensions of patches. Used to assess region sizes.
    mask_indices : tuple
        Mask labels for e.g. cilia vs background pixels.

    Returns
    -------
    mask : array, shape (H, W)
    """
    CILIA, BG = mask_indices
    
    # Do some morphological smoothing.
    mask = skmp.opening(mask, selem = skmp.square(5))

    # Find all the objects in the image and prune out the small ones.
    labels = skms.label(mask)
    props = skms.regionprops(labels)
    for p in props:
        coords = p.coords[0]  # Pull out the first x,y coordinate.
        if mask[coords[0], coords[1]] != CILIA: continue  # Not cilia.
        if p.area <= (patchsize ** 2) * 0.75:
            ### TODO: Choose the size threshold more intelligently.

            # If the number of pixels in a region of cilia is below some
            # threshold, just set them to background.
            mask[labels == p.label] = BG

            ### TODO: Instead of setting everything to BG, take a vote of
            # the pixel labels along the perimeter of the object, and set
            # the object to the label of the majority.
    return mask

def choose_seeds(mask, overlap, patchsize):
    """
    Randomly chooses a number of seeds for patch extraction.
    """
    # Create a distance transform of the mask, and restrict the results only
    # to those pixels within the mask. We'll use this as a sort of "spatial"
    # probability distribution for randomly choosing patch seeds.
    dist = spnd.distance_transform_edt(mask)

    # Need to first zero out the borders.
    width = int(patchsize / 2)
    dist[:width, :] = dist[-width:, :] = dist[:, :width] = dist[:, -width:] = 0

    # Now, proceed with sampling.
    cilia_indices = np.where(dist > 0)
    weights = dist[cilia_indices]
    indices = np.column_stack([cilia_indices[0], cilia_indices[1]])
    weights /= weights.sum()

    # How many patches do we actually extract?
    num_pixels = weights.shape[0]
    dense_patch_num = int((num_pixels / (patchsize ** 2))) # * (1 - overlap))
    if dense_patch_num == 0:
        return None
    ### TODO: More intelligently determine the number of patches.

    seeds = np.random.choice(num_pixels, size = dense_patch_num, replace = False, p = weights)
    ### TODO: Alter the probability density after each sampling to discourage
    # additional patches from being sampled right around the previous one.

    selected = indices[seeds]
    return selected

def extract_patches(video, seeds, patchsize, frames, gray_frame = None):
    patches = np.zeros(shape = (seeds.shape[0], frames, patchsize, patchsize))

    if gray_frame is not None:
        illustration = np.copy(gray_frame)

    for i, [r, c] in enumerate(seeds):
        r_corner = r - int(patchsize / 2)
        c_corner = c - int(patchsize / 2)
        patch = video[:frames, r_corner:(r_corner + patchsize), c_corner:(c_corner + patchsize)]
        patches[i] = patch

        if gray_frame is not None:
            illustration[r_corner:(r_corner + patchsize), c_corner] = 0
            illustration[r_corner:(r_corner + patchsize), c_corner + patchsize - 1] = 0
            illustration[r_corner, c_corner:(c_corner + patchsize)] = 0
            illustration[r_corner + patchsize - 1, c_corner:(c_corner + patchsize)] = 0

    if gray_frame is not None:
        return [patches, illustration]

    return patches

def list_of_files(masks, rots, grays):
    f = [item.split("/")[-1].split("_")[0] for item in glob.glob(os.path.join(masks, "*.npy"))]

    # Test to make sure the others exist, too.
    missing = []
    for index, item in enumerate(f):
        rotpath = os.path.join(rots, "{}.npy".format(item))
        graypath = os.path.join(grays, "{}.npy".format(item))
        if not os.path.exists(rotpath) or not os.path.exists(graypath):
            missing.append(index)
    for index in missing[::-1]:
        f.pop(index)
    return f

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Cilia DL',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python extract_patches.py <args>')
    parser.add_argument('-im', '--input_masks', required = True,
        help = 'Path to a folder containing mask files.')
    parser.add_argument("-ir", "--input_rot", required = True,
        help = "Path to a folder containing rotation data.")
    parser.add_argument("-ig", "--input_gray", required = True,
        help = "Path to a folder containing grayscale video data.")
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

    # Seed the RNG for anything other than a seed of -1.
    if args['random_state'] != -1:
        np.random.seed(args['random_state'])

    # Get a listing of the files.
    patient_ids = list_of_files(args['input_masks'], args['input_rot'], args['input_gray'])

    # Run the operation.
    optargs = ([args['input_masks'], args['input_rot'], args['input_gray'],
                args['cilia_mask'], args['cell_mask'], args['bg_mask'],
                args['patchsize'], args['overlap'], args['num_frames'], args['output']])
    patches = Parallel(n_jobs = -1, verbose = 10)(
        delayed(run_patch_selection)(f, optargs) for f in patient_ids)

    # Go through the results.
    total_patches = 0
    files = len(patches)
    failures = 0
    fail_files = []
    for fname, outcome, count in patches:
        if not outcome:
            failures += 1
            fail_files.append(fname)
        else:
            total_patches += count

    print("Obtained {} patches from {} files.".format(total_patches, files))
    print("{} files yielded 0 patches. They follow:".format(failures))
    print(fail_files)
