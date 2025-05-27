from collections.abc import Sequence
from functools import update_wrapper
from typing import Any, List, Optional, Tuple, Union, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

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
    array: NDArray,
    *,
    norm: Optional[Union[NormalizationConfig, dict, str]] = None,
    scalebar: Optional[Union[ScalebarConfig, dict, bool]] = None,
    cmap: Union[str, colors.Colormap] = "gray",
    chroma_boost: float = 1.0,
    cbar: bool = False,
    figax: Optional[Tuple[Any, Any]] = None,
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Display a 2D array as an image with optional colorbar and scalebar.

    This function visualizes a 2D array, handling both real and complex data.
    For complex data, it displays amplitude and phase information using a
    perceptually-uniform color representation.

    Parameters
    ----------
    array : ndarray
        The 2D array to visualize. Can be real or complex.
    norm : NormalizationConfig or dict or str, optional
        Configuration for normalizing the data before visualization.
    scalebar : ScalebarConfig or dict or bool, optional
        Configuration for adding a scale bar to the plot.
    cmap : str or Colormap, default="gray"
        Colormap to use for real data or amplitude of complex data.
    chroma_boost : float, default=1.0
        Factor to boost color saturation when displaying complex data.
    cbar : bool, default=False
        Whether to add a colorbar to the plot.
    figax : tuple, optional
        (fig, ax) tuple to use for plotting. If None, a new figure and axes are created.
    figsize : tuple, default=(8, 8)
        Figure size in inches, used only if figax is None.
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    """
    is_complex = np.iscomplexobj(array)
    if is_complex:
        amplitude = np.abs(array)
        angle = np.angle(array)
    else:
        amplitude = array
        angle = None

    norm_config = _resolve_normalization(norm)
    scalebar_config = _resolve_scalebar(scalebar)
    
    norm_obj = CustomNormalization(
        interval_type=norm_config.interval_type,
        stretch_type=norm_config.stretch_type,
        lower_quantile=norm_config.lower_quantile,
        upper_quantile=norm_config.upper_quantile,
        vmin=norm_config.vmin,
        vmax=norm_config.vmax,
        vcenter=norm_config.vcenter,
        half_range=norm_config.half_range,
        power=norm_config.power,
        logarithmic_index=norm_config.logarithmic_index,
        asinh_linear_range=norm_config.asinh_linear_range,
        data=amplitude,
    )

    scaled_amplitude = norm_obj(amplitude)
    rgba = array_to_rgba(scaled_amplitude, angle, cmap=cmap, chroma_boost=chroma_boost)

    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    ax.imshow(rgba)
    ax.set(xticks=[], yticks=[], title=title)

    if cbar:
        divider = make_axes_locatable(ax)
        ax_cb_abs = divider.append_axes("right", size="5%", pad="2.5%")
        # Convert cmap to Colormap if it's a string
        cmap_obj = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
        cb_abs = add_cbar_to_ax(fig, ax_cb_abs, norm_obj, cmap_obj)

        if is_complex:
            ax_cb_angle = divider.append_axes("right", size="5%", pad="10%")
            add_arg_cbar_to_ax(fig, ax_cb_angle, chroma_boost=chroma_boost)
            cb_abs.set_label("abs", rotation=0, ha="center", va="bottom")
            cb_abs.ax.yaxis.set_label_coords(0.5, -0.05)

    if scalebar_config is not None:
        add_scalebar_to_ax(
            ax,
            rgba.shape[1],
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
    list_of_arrays: Sequence[NDArray],
    *,
    norm: Optional[Union[NormalizationConfig, dict, str]] = None,
    scalebar: Optional[Union[ScalebarConfig, dict, bool]] = None,
    cmap: Union[str, colors.Colormap] = "gray",
    chroma_boost: float = 1.0,
    cbar: bool = False,
    figax: Optional[Tuple[Any, Any]] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Any, Any]:
    """Display multiple 2D arrays as a single combined image.

    This function takes a list of 2D arrays and creates a single visualization
    where each array is assigned a unique color, and their amplitudes determine
    the contribution to the final color. This is useful for comparing multiple
    related datasets.

    Parameters
    ----------
    list_of_arrays : sequence of ndarray
        Sequence of 2D arrays to combine into a single visualization.
    norm : NormalizationConfig or dict or str, optional
        Configuration for normalizing the data before visualization.
    scalebar : ScalebarConfig or dict or bool, optional
        Configuration for adding a scale bar to the plot.
    cmap : str or Colormap, default="gray"
        Base colormap to use (though each array will get a unique color).
    chroma_boost : float, default=1.0
        Factor to boost color saturation.
    cbar : bool, default=False
        Whether to add a colorbar to the plot (not yet implemented).
    figax : tuple, optional
        (fig, ax) tuple to use for plotting. If None, a new figure and axes are created.
    figsize : tuple, default=(8, 8)
        Figure size in inches, used only if figax is None.

    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.

    Raises
    ------
    NotImplementedError
        If cbar is True (colorbar for combined visualization not yet implemented).
    """
    norm_config = _resolve_normalization(norm)
    scalebar_config = _resolve_scalebar(scalebar)

    norm_obj = CustomNormalization(
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

    # Convert Sequence to List for list_of_arrays_to_rgba
    list_of_arrays_list = list(list_of_arrays)
    rgba = list_of_arrays_to_rgba(
        list_of_arrays_list,
        norm=norm_obj,
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
            rgba.shape[1],
            scalebar_config.sampling,
            scalebar_config.length,
            scalebar_config.units,
            scalebar_config.width_px,
            scalebar_config.pad_px,
            scalebar_config.color,
            scalebar_config.loc,
        )

    return fig, ax


def _normalize_show_input_to_grid(
    arrays: Union[NDArray, Sequence[NDArray], Sequence[Sequence[NDArray]]],
) -> List[List[NDArray]]:
    """Convert various input formats to a consistent grid format for visualization.

    This helper function normalizes different input formats to a consistent
    grid format that can be used by the visualization functions.

    Parameters
    ----------
    arrays : ndarray or sequence of ndarray or sequence of sequences of ndarray
        Input arrays in various formats.

    Returns
    -------
    list of lists of ndarray
        Normalized grid format where each inner list represents a row of arrays.
    """
    if isinstance(arrays, np.ndarray):
        return [[arrays]]
    if isinstance(arrays, Sequence) and not isinstance(arrays[0], Sequence):
        # Convert sequence to list and ensure each element is an NDArray
        return [[cast(NDArray, arr) for arr in arrays]]
    # Convert outer and inner sequences to lists, ensuring proper types
    return [[cast(NDArray, arr) for arr in row] for row in arrays]


def show_2d(
    arrays: Union[NDArray, Sequence[NDArray], Sequence[Sequence[NDArray]]],
    *,
    figax: Optional[Tuple[Any, Any]] = None,
    axsize: Tuple[int, int] = (4, 4),
    tight_layout: bool = True,
    combine_images: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Display one or more 2D arrays in a grid layout.

    This is the main visualization function that can display a single array,
    a list of arrays, or a grid of arrays. It supports both individual and
    combined visualization modes.

    Parameters
    ----------
    arrays : ndarray or sequence of ndarray or sequence of sequences of ndarray
        The arrays to visualize. Can be a single array, a sequence of arrays,
        or a nested sequence representing a grid of arrays.
    figax : tuple, optional
        (fig, axs) tuple to use for plotting. If None, a new figure and axes are created.
    axsize : tuple, default=(4, 4)
        Size of each subplot in inches.
    tight_layout : bool, default=True
        Whether to apply tight_layout to the figure.
    combine_images : bool, default=False
        If True and arrays is a sequence, combine all arrays into a single visualization
        using color encoding. Only works for a single row of arrays.
    **kwargs : dict
        Additional keyword arguments passed to _show_2d or _show_2d_combined.

    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    axs : ndarray of Axes
        The matplotlib axes objects. If multiple arrays are displayed, this is a 2D array.

    Raises
    ------
    ValueError
        If combine_images is True but arrays contains multiple rows, or if
        figax is provided but the axes shape doesn't match the grid shape.
    """
    grid = _normalize_show_input_to_grid(arrays)
    nrows = len(grid)
    ncols = max(len(row) for row in grid)

    title = kwargs.pop("title", None)

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
            if title is None:
                t = None
            elif isinstance(title, str):
                t = title
            elif isinstance(title[0], str):
                # Flat list of titles
                t = title[i * ncols + j] if i * ncols + j < len(title) else None
            else:
                # Grid of titles
                t = title[i][j] if i < len(title) and j < len(title[i]) else None

            _show_2d(
                array,
                figax=figax,
                title=t,
                **kwargs,
            )

            # figax = (fig, axs[i][j])
            # _show_2d(
            #     array,
            #     figax=figax,
            #     **kwargs,
            # )

    # Hide unused axes in incomplete rows
    for i, row in enumerate(grid):
        for j in range(len(row), ncols):
            axs[i][j].axis("off")  # type: ignore

    if tight_layout:
        fig.tight_layout()

    # Squeeze the axes to the expected shape
    if axs.shape == (1, 1):
        axs = axs[0, 0]
    elif axs.shape[0] == 1:
        axs = axs[0]
    elif axs.shape[1] == 1:
        axs = axs[:, 0]

    return fig, axs


update_wrapper(show_2d, _show_2d)
