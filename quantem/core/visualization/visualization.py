import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from quantem.core.visualization.custom_normalizations import CustomNormalization


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


def _show_2D(
    array,
    *,
    interval_type: str = "quantile",
    stretch_type: str = "linear",
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float = 0.0,
    half_range: float | None = None,
    power: float = 1.0,
    logarithmic_index: float = 1000.0,
    asinh_linear_range: float = 0.1,
    cmap: str | mpl.colors.Colormap = "gray",
    chroma_boost: float = 1.0,
    cbar: bool = False,
    scalebar: bool = False,
    scalebar_sampling: float = 1.0,
    scalebar_units: str = "pixels",
    scalebar_length: float = None,
    scalebar_width_px: float = 1,
    scalebar_pad_px: float = 0.5,
    scalebar_color: str = "white",
    scalebar_loc: str | int = "lower right",
    figax: tuple = None,
    figsize: tuple = (8, 8),
):
    """ """
    is_complex = np.iscomplexobj(array)
    if is_complex:
        amplitude = np.abs(array)
        angle = np.angle(array)
    else:
        amplitude = array
        angle = None

    norm = CustomNormalization(
        interval_type=interval_type,
        stretch_type=stretch_type,
        data=amplitude,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        vmin=vmin,
        vmax=vmin,
        vcenter=vcenter,
        half_range=half_range,
        power=power,
        logarithmic_index=logarithmic_index,
        asinh_linear_range=asinh_linear_range,
    )

    scaled_amplitude = norm(amplitude)
    rgba = array_to_rgba(scaled_amplitude, angle, cmap=cmap, chroma_boost=chroma_boost)

    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    ax.imshow(rgba)
    ax.set(xticks=[], yticks=[])

    if cbar:
        divider = make_axes_locatable(ax)
        ax_cb_abs = divider.append_axes("right", size="5%", pad="2.5%")
        cb_abs = add_cbar_to_ax(fig, ax_cb_abs, norm, cmap)

        if is_complex:
            ax_cb_angle = divider.append_axes("right", size="5%", pad="10%")
            cb_angle = add_arg_cbar_to_ax(fig, ax_cb_angle, chroma_boost=chroma_boost)
            cb_abs.set_label("abs", rotation=0, ha="center", va="bottom")
            cb_abs.ax.yaxis.set_label_coords(0.5, -0.05)

    if scalebar:
        add_scalebar_to_ax(
            ax,
            array.shape[1],
            scalebar_sampling,
            scalebar_length,
            scalebar_units,
            scalebar_width_px,
            scalebar_pad_px,
            scalebar_color,
            scalebar_loc,
        )

    return None
