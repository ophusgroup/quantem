from collections.abc import Sequence
from functools import update_wrapper

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from quantem.core.visualization.custom_normalizations import (
    CustomNormalization,
    NormalizationConfig,
    _resolve_normalization,
)
from quantem.core.visualization.visualization_utils import (
    ScalebarConfig,
    _resolve_scalebar,
    add_arg_cbar_to_ax,
    add_cbar_to_ax,
    add_scalebar_to_ax,
    array_to_rgba,
    list_of_arrays_to_rgba,
)


def _show_2d(
    array,
    *,
    norm: NormalizationConfig | dict | str = None,
    scalebar: ScalebarConfig | dict | bool = None,
    cmap: str | mpl.colors.Colormap = "gray",
    chroma_boost: float = 1.0,
    cbar: bool = False,
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

    norm_config = _resolve_normalization(norm)
    scalebar_config = _resolve_scalebar(scalebar)

    norm = CustomNormalization(
        interval_type=norm_config.interval_type,
        stretch_type=norm_config.stretch_type,
        lower_quantile=norm_config.lower_quantile,
        upper_quantile=norm_config.upper_quantile,
        vmin=norm_config.vmin,
        vmax=norm_config.vmin,
        vcenter=norm_config.vcenter,
        half_range=norm_config.half_range,
        power=norm_config.power,
        logarithmic_index=norm_config.logarithmic_index,
        asinh_linear_range=norm_config.asinh_linear_range,
        data=amplitude,
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

    if scalebar_config is not None:
        add_scalebar_to_ax(
            ax,
            array.shape[1],
            scalebar_config.sampling,
            scalebar_config.length,
            scalebar_config.units,
            scalebar_config.width_px,
            scalebar_config.pad_px,
            scalebar_config.color,
            scalebar_config.loc,
        )

    return fig, ax


def _show_2d_combined(
    list_of_arrays,
    *,
    norm: NormalizationConfig | dict | str = None,
    scalebar: ScalebarConfig | dict | bool = None,
    cmap: str | mpl.colors.Colormap = "gray",
    chroma_boost: float = 1.0,
    cbar: bool = False,
    figax: tuple = None,
    figsize: tuple = (8, 8),
):
    """ """

    norm_config = _resolve_normalization(norm)
    scalebar_config = _resolve_scalebar(scalebar)

    norm = CustomNormalization(
        interval_type=norm_config.interval_type,
        stretch_type=norm_config.stretch_type,
        lower_quantile=norm_config.lower_quantile,
        upper_quantile=norm_config.upper_quantile,
        vmin=norm_config.vmin,
        vmax=norm_config.vmin,
        vcenter=norm_config.vcenter,
        half_range=norm_config.half_range,
        power=norm_config.power,
        logarithmic_index=norm_config.logarithmic_index,
        asinh_linear_range=norm_config.asinh_linear_range,
    )

    rgba = list_of_arrays_to_rgba(
        list_of_arrays,
        norm=norm,
        chroma_boost=chroma_boost,
    )

    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    ax.imshow(rgba)
    ax.set(xticks=[], yticks=[])

    if cbar:
        raise NotImplementedError()

    if scalebar_config is not None:
        add_scalebar_to_ax(
            ax,
            array.shape[1],
            scalebar_config.sampling,
            scalebar_config.length,
            scalebar_config.units,
            scalebar_config.width_px,
            scalebar_config.pad_px,
            scalebar_config.color,
            scalebar_config.loc,
        )

    return fig, ax


def _normalize_show_input_to_grid(arrays):
    """ """
    if isinstance(arrays, np.ndarray):
        return [[arrays]]
    if isinstance(arrays, Sequence) and not isinstance(arrays[0], Sequence):
        return [arrays]
    return arrays


def show_2d(
    arrays,
    *,
    figax=None,
    axsize=(6, 6),
    tight_layout=True,
    combine_images=False,
    **kwargs,
):
    """ """
    grid = _normalize_show_input_to_grid(arrays)
    nrows = len(grid)
    ncols = max(len(row) for row in grid)

    if combine_images:
        if nrows > 1:
            raise ValueError()

        return _show_2d_combined(grid[0], figax=figax, **kwargs)

    if figax is not None:
        fig, axs = figax
        if not isinstance(axs, np.ndarray):
            axs = np.array([[axs]])
        elif axs.ndim == 1:
            axs = axs.reshape(1, -1)
        if axs.shape != (nrows, ncols):
            raise ValueError()
    else:
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(axsize[0] * ncols, axsize[1] * nrows), squeeze=False
        )

    for i, row in enumerate(grid):
        for j, array in enumerate(row):
            figax = (fig, axs[i][j])
            _show_2d(
                array,
                figax=figax,
                **kwargs,
            )

    # Hide unused axes in incomplete rows
    for j in range(len(row), ncols):
        axs[i][j].axis("off")

    if tight_layout:
        fig.tight_layout()

    return fig, axs


update_wrapper(show_2d, _show_2d)
