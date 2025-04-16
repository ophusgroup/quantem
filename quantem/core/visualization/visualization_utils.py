import numpy as np


def return_clipped_array(array, vmin=None, vmax=None, normalize=False):
    """
    Utility function for calculating vmin and vmax values for clipped array
    based on histogram distribution of pixel values

    Parameters
    ----------
    array: np.array
        array to be scaled
    vmin: float
        lower fraction cut off of pixel values
    vmax: float
        upper fraction cut off of pixel values
    normalize: bool
        if True, rescales clipped array from 0 to 1

    Returns
    ----------
    clipped_array: np.array
        array clipped outside vmin and vmax
    vmin: float
        lower value to be plotted
    vmax: float
        upper value to be plotted
    """

    if vmin is None:
        vmin = 0.02
    if vmax is None:
        vmax = 0.98

    vals = np.sort(array.ravel())
    ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
    ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
    ind_vmin = np.max([0, ind_vmin])
    ind_vmax = np.min([len(vals) - 1, ind_vmax])
    vmin = vals[ind_vmin]
    vmax = vals[ind_vmax]

    if vmax == vmin:
        vmin = vals[0]
        vmax = vals[-1]

    clipped_array = array.copy()
    clipped_array = np.where(clipped_array < vmin, vmin, clipped_array)
    clipped_array = np.where(clipped_array > vmax, vmax, clipped_array)

    if normalize:
        clipped_array -= clipped_array.min()
        clipped_array /= clipped_array.max()
        vmin = 0
        vmax = 1

    return clipped_array, vmin, vmax
