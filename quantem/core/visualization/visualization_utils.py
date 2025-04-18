import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert


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


def scalar_array_to_rgb(
    scalar_array, vmin=None, vmax=None, power=1, cmap="gray", normalization="linear"
):
    """ """

    if normalization == "linear":
        norm = mpl.colors.Normalize()
    elif normalization == "power-law":
        norm = mpl.colors.PowerNorm(gamma=power)
    elif normalization == "logarithmic":
        norm = mpl.colors.LogNorm()
    elif normalization == "centered":
        norm = mpl.colors.CenteredNorm()
    else:
        raise ValueError()

    scaled_array, vmin, vmax = return_clipped_array(scalar_array, vmin=vmin, vmax=vmax)
    scaled_array = norm(scaled_array)

    cmap = plt.get_cmap(cmap)
    rgb = cmap(scaled_array)

    return rgb


def complex_array_to_rgb(complex_array, vmin=None, vmax=None, power=1, chroma_boost=1):
    """
    Utility function for converting complex arrays to clipped rgb values
    based on histogram distribution of pixel values

    Parameters
    ----------
    complex_array: np.array
        complex_array to be scaled
    vmin: float
        lower fraction cutoff to clip amplitude values
    vmax: float
        upper fraction cutoff to clip amplitude values
    power: float
        power to raise amplitude before clipping
    chroma_boost: float
        boosts chroma amplitude for higher contrast

    Returns
    ----------
    rgb: np.array
        clipped rgb JCh array
    """

    amplitude = np.abs(complex_array) ** power
    phase = np.angle(complex_array)

    scaled_amplitude, _, _ = return_clipped_array(
        amplitude, vmin=vmin, vmax=vmax, normalize=True
    )
    scaled_amplitude = scaled_amplitude.clip(1e-16, 1)

    J = scaled_amplitude * 61.5
    C = np.minimum(chroma_boost * 98 * J / 123, 110)
    h = np.rad2deg(phase) + 180

    JCh = np.stack((J, C, h), axis=-1)
    rgb = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

    return rgb
