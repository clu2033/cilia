def read_labels(labelfile):
    """
    Utility method for reading the labels. Reads both the short (0-1) and
    full (0-3) labels in two lists, and the user can decide which is needed.

    Ignores any rows with -1 labels.

    Parameters
    ----------
    labelfile : string
        Path specifying the location of the label file on the filesystem.

    Returns
    -------
    shortlist : dict
        Dictionary of labels, with patient IDs as the keys and 0/1 ints as the values.
        0 = Normal, 1 = Abnormal.
    longlist : dict
        Dictionary of labels, with patient IDs as the keys and 0-3 ints as the values.
        0 = Normal, 1 = Probably normal, 2 = Probably abnormal, 3 = Abnormal.
    """
    shortlist = {}
    longlist = {}
    fi = open(labelfile, "r")
    for line in fi:
        p, f, l = line.strip().split(' ')
        l = int(l)
        f = int(f)
        if l < 0: continue
        shortlist[p] = l
        longlist[p] = f
    fi.close()
    return [shortlist, longlist]
