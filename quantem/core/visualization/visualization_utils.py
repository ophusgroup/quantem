from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.stats import binned_statistic_2d

from quantem.core.visualization.custom_normalizations import CustomNormalization


def array_to_rgba(
    scaled_amplitude: np.ndarray,
    scaled_angle: np.ndarray | None = None,
    *,
    cmap: str | mpl.colors.Colormap = "gray",
    chroma_boost: float = 1,
):
    """Convert amplitude and angle arrays to an RGBA color array.

    This function creates a color representation of data using either a simple colormap
    or a perceptually-uniform color space based on amplitude and angle information.

    Parameters
    ----------
    scaled_amplitude : np.ndarray
        Array of amplitude values, typically normalized to [0, 1].
    scaled_angle : np.ndarray, optional
        Array of angle values in radians. If provided, creates a color representation
        using the JCh color space where amplitude controls lightness and angle controls hue.
    cmap : str or mpl.colors.Colormap, default="gray"
        Colormap to use when scaled_angle is None.
    chroma_boost : float, default=1
        Factor to boost color saturation when using angle-based coloring.

    Returns
    -------
    np.ndarray
        RGBA array with shape (height, width, 4) where the last dimension contains
        (red, green, blue, alpha) values in the range [0, 1].

    Raises
    ------
    ValueError
        If scaled_angle is provided but has a different shape than scaled_amplitude.
    """
    cmap = (
        cmap if isinstance(cmap, mpl.colors.Colormap) else mpl.colormaps.get_cmap(cmap)
    )
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
    """Converts a list of arrays to a perceptually-uniform RGB array.

    This function takes multiple arrays and creates a color representation where each
    array is assigned a unique hue angle, and the amplitude of each array determines
    the contribution to the final color. The result is a perceptually-uniform color
    representation that can effectively visualize multiple data sources simultaneously.

    Parameters
    ----------
    list_of_arrays : list of np.ndarray
        List of arrays to be converted to a color representation. All arrays must have
        the same shape.
    norm : CustomNormalization, default=CustomNormalization()
        Normalization to apply to each array before processing.
    chroma_boost : float, default=1
        Factor to boost color saturation in the final output.

    Returns
    -------
    np.ndarray
        RGBA array with shape (height, width, 4) representing the combined data.
    """
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


@dataclass
class ScalebarConfig:
    """Configuration for adding a scale bar to a plot.

    Attributes
    ----------
    sampling : float, default=1.0
        Physical units per pixel.
    units : str, default="pixels"
        Units to display on the scale bar.
    length : float, optional
        Length of the scale bar in physical units. If None, an appropriate length
        will be estimated.
    width_px : float, default=1
        Width of the scale bar in pixels.
    pad_px : float, default=0.5
        Padding around the scale bar in pixels.
    color : str, default="white"
        Color of the scale bar.
    loc : str or int, default="lower right"
        Location of the scale bar on the plot. Can be a string like "lower right"
        or an integer location code.
    """

    sampling: float = 1.0
    units: str = "pixels"
    length: float | None = None
    width_px: float = 1
    pad_px: float = 0.5
    color: str = "white"
    loc: str | int = "lower right"


def _resolve_scalebar(cfg) -> ScalebarConfig:
    """Resolve various input types to a ScalebarConfig object.

    Parameters
    ----------
    cfg : None, bool, dict, or ScalebarConfig
        Configuration for the scale bar.

    Returns
    -------
    ScalebarConfig or None
        Resolved configuration object or None if cfg is None or False.

    Raises
    ------
    TypeError
        If cfg is not one of the supported types.
    """
    if cfg is None or cfg is False:
        return None
    elif cfg is True:
        return ScalebarConfig()
    elif isinstance(cfg, dict):
        return ScalebarConfig(**cfg)
    elif isinstance(cfg, ScalebarConfig):
        return cfg
    else:
        raise TypeError("scalebar must be None, dict, bool, or ScalebarConfig")


def estimate_scalebar_length(length, sampling):
    """Estimate an appropriate scale bar length based on data dimensions.

    This function calculates a "nice" scale bar length that is a multiple of
    0.5, 1.0, or 2.0 times a power of 10, depending on the data range.

    Parameters
    ----------
    length : float
        Total length of the data in physical units.
    sampling : float
        Physical units per pixel.

    Returns
    -------
    tuple
        (length_units, length_pixels) where length_units is the estimated
        scale bar length in physical units and length_pixels is the equivalent
        in pixels.
    """
    d = length * sampling / 2
    exp = np.floor(np.log10(d))
    base = d / (10**exp)
    if base >= 1 and base < 2.1:
        _spacing = 0.5
    elif base >= 2.1 and base < 4.6:
        _spacing = 1.0
    elif base >= 4.6 and base <= 10:
        _spacing = 2.0
    spacing = _spacing * 10**exp
    return spacing, spacing / sampling


def add_scalebar_to_ax(
    ax,
    array_size,
    sampling,
    length_units,
    units,
    width_px,
    pad_px,
    color,
    loc,
):
    """Add a scale bar to a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to.
    array_size : float
        Size of the data array in pixels.
    sampling : float
        Physical units per pixel.
    length_units : float, optional
        Length of the scale bar in physical units. If None, an appropriate length
        will be estimated.
    units : str
        Units to display on the scale bar.
    width_px : float
        Width of the scale bar in pixels.
    pad_px : float
        Padding around the scale bar in pixels.
    color : str
        Color of the scale bar.
    loc : str or int
        Location of the scale bar on the plot.
    """
    if length_units is None:
        length_units, length_px = estimate_scalebar_length(array_size, sampling)
    else:
        length_px = length_units / sampling

    if length_units % 1 == 0.0:
        label = f"{length_units:.0f} {units}"
    else:
        label = f"{length_units:.2f} {units}"

    if isinstance(loc, int):
        loc_codes = mpl.legend.Legend.codes
        loc_strings = {v: k for k, v in loc_codes.items()}
        loc = loc_strings[loc]

    bar = AnchoredSizeBar(
        ax.transData,
        length_px,
        label,
        loc,
        pad=pad_px,
        color=color,
        frameon=False,
        label_top=loc[:3] == "low",
        size_vertical=width_px,
    )
    ax.add_artist(bar)


def add_cbar_to_ax(
    fig,
    cax,
    norm,
    cmap,
    eps=1e-8,
):
    """Add a colorbar to a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the colorbar to.
    cax : matplotlib.axes.Axes
        The axes to place the colorbar in.
    norm : matplotlib.colors.Normalize
        The normalization for the colormap.
    cmap : matplotlib.colors.Colormap
        The colormap to use.
    eps : float, default=1e-8
        Small value to avoid floating point errors when filtering ticks.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar object.
    """
    tick_locator = mpl.ticker.AutoLocator()
    ticks = tick_locator.tick_values(norm.vmin, norm.vmax)
    ticks = ticks[(ticks >= norm.vmin - eps) & (ticks <= norm.vmax + eps)]

    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, ticks=ticks, format=formatter)
    return cb


def add_arg_cbar_to_ax(
    fig,
    cax,
    chroma_boost=1,
):
    """Add a colorbar for phase values to a matplotlib figure.

    This function creates a colorbar suitable for displaying phase values.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the colorbar to.
    cax : matplotlib.axes.Axes
        The axes to place the colorbar in.
    chroma_boost : float, default=1
        Factor to boost color saturation.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar object.
    """
    h = np.linspace(0, 360, 256, endpoint=False)
    J = np.full_like(h, 61.5)
    C = np.full_like(h, np.minimum(49 * chroma_boost, 110))
    JCh = np.stack((J, C, h), axis=-1)
    rgb_vals = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

    angle_cmap = mpl.colors.ListedColormap(rgb_vals)
    angle_norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    sm = mpl.cm.ScalarMappable(norm=angle_norm, cmap=angle_cmap)
    cb_angle = fig.colorbar(sm, cax=cax)

    cb_angle.set_label("arg", rotation=0, ha="center", va="bottom")
    cb_angle.ax.yaxis.set_label_coords(0.5, -0.05)
    cb_angle.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb_angle.set_ticklabels(
        [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
    )

    return cb_angle


def turbo_black(num_colors=256, fade_len=None):
    """Create a modified version of the 'turbo' colormap that fades to black.

    This function creates a colormap based on the 'turbo' colormap but with
    the beginning portion fading to black, which can be useful for visualizing
    data with a clear zero point.

    Parameters
    ----------
    num_colors : int, default=256
        Number of colors in the colormap.
    fade_len : int, optional
        Number of colors to fade to black at the beginning. If None, defaults
        to num_colors // 8.

    Returns
    -------
    matplotlib.colors.ListedColormap
        The modified colormap.
    """
    if fade_len is None:
        fade_len = num_colors // 8
    turbo = mpl.colormaps.get_cmap("turbo").resampled(num_colors)
    colors = turbo(np.linspace(0, 1, num_colors))
    fade = np.linspace(0, 1, fade_len)[:, None]
    colors[:fade_len, :3] *= fade
    return mpl.colors.ListedColormap(colors)


_turbo_black = turbo_black()
mpl.colormaps.register(_turbo_black, name="turbo_black")
mpl.colormaps.register(_turbo_black.reversed(), name="turbo_black_r")


def bilinear_histogram_2d(
    shape,
    x,
    y,
    weight,
    origin=(0.0, 0.0),
    sampling=(1.0, 1.0),
    statistic="sum",
):
    """Create a 2D histogram with bilinear binning.

    This function creates a 2D histogram where data points are distributed
    across bins according to their position relative to bin centers, allowing
    for smoother visualizations than standard histograms.

    Parameters
    ----------
    shape : tuple
        (Nx, Ny) shape of the output histogram.
    x : array-like
        x-coordinates of the data points.
    y : array-like
        y-coordinates of the data points.
    weight : array-like
        Weights for each data point.
    origin : tuple, default=(0.0, 0.0)
        (x0, y0) origin of the histogram grid.
    sampling : tuple, default=(1.0, 1.0)
        (dx, dy) sampling intervals.
    statistic : str, default="sum"
        Statistic to compute for each bin. Options include "sum", "mean", "count", etc.

    Returns
    -------
    np.ndarray
        2D histogram array with shape (Nx, Ny).
    """
    Nx, Ny = shape
    dx, dy = sampling
    x0, y0 = origin
    x1, y1 = x0 + Nx * dx, y0 + Ny * dy

    hist, _, _, _ = binned_statistic_2d(
        x,
        y,
        values=weight,
        statistic=statistic,
        bins=[Nx, Ny],  # [rows, cols]
        range=[[x0, x1], [y0, y1]],  # [[x_min, x_max], [y_min, y_max]]
    )

    return hist  # shape = (Nx, Ny), i.e., array[x, y]
