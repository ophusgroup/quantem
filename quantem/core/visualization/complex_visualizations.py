from colorspacious import cspace_convert
from quantem.core.visualization.visualization_utils import return_clipped_array
import numpy as np


def complex_array_to_rgb(
    complex_array, vmin=None, vmax=None, power=None, chroma_boost=None
):
    """ """
    amplitude = np.abs(complex_array)
    phase = np.angle(complex_array)

    if power is not None:
        amplitude = amplitude**power

    if chroma_boost is None:
        chroma_boost = 1

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
