import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert

from quantem.core.visualization.custom_normalizations import CustomNormalization


def array_to_rgba(
    scaled_amplitude: np.ndarray,
    scaled_angle: np.ndarray | None = None,
    *,
    cmap: str | mpl.colors.Colormap = "gray",
    chroma_boost: float = 1,
):
    """ """
    cmap = cmap if isinstance(cmap, mpl.colors.Colormap) else plt.get_cmap(cmap)
    if scaled_angle is None:
        rgba = cmap(scaled_amplitude)
    else:
        if scaled_angle.shape != scaled_amplitude.shape:
            raise ValueError()

        J = scaled_amplitude * 61.5
        C = np.minimum(chroma_boost * 98 * J / 123, 110)
        h = np.rad2deg(scaled_angle) + 180

        JCh = np.stack((J, C, h), axis=-1)
        with np.errstate(invalid="ignore"):
            rgb = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

        alpha = np.ones_like(scaled_amplitude)
        rgba = np.dstack((rgb, alpha))

    return rgba


def list_of_arrays_to_rgba(
    list_of_arrays,
    *,
    norm: CustomNormalization = CustomNormalization(),
    chroma_boost: float = 1,
):
    """Converts a list of arrays to a perceptually-uniform RGB array."""
    list_of_arrays = [norm(array) for array in list_of_arrays]
    bins = np.asarray(list_of_arrays)
    n = bins.shape[0]

    # circular encoding
    hue_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    hue_angles += np.linspace(0.0, 0.5, n) * (
        2 * np.pi / n / 2
    )  # jitter to avoid cancellation
    complex_weights = np.exp(1j * hue_angles)[:, None, None] * bins

    # weighted average direction (w/ normalization)
    complex_sum = complex_weights.sum(0)
    scaled_amplitude = np.clip(np.abs(complex_sum), 0, 1)
    scaled_angle = np.angle(complex_sum)

    return array_to_rgba(scaled_amplitude, scaled_angle, chroma_boost=chroma_boost)
