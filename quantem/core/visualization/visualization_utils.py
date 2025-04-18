import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert


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
