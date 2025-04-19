from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from quantem.core.visualization.custom_normalizations import CustomNormalization


def array_to_rgba(
    scaled_amplitude: np.ndarray,
    scaled_angle: np.ndarray | None = None,
    *,
    cmap: str | mpl.colors.Colormap = "gray",
    chroma_boost: float = 1,
):
    """ """
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


@dataclass
class ScalebarConfig:
    sampling: float = 1.0
    units: str = "pixels"
    length: float | None = None
    width_px: float = 1
    pad_px: float = 0.5
    color: str = "white"
    loc: str | int = "lower right"


def _resolve_scalebar(cfg) -> ScalebarConfig:
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
    """ """
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
    """ """
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
    """ """
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
    """ """

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
    """ """
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
