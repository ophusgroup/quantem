from typing import Any

import numpy as np
from attrs import Attribute, Converter, Factory, define, field

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization.visualization import show_2d
from quantem.core.visualization.visualization_utils import ScalebarConfig

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


# because converters run before validators, some validation has to be done here too
# not sure how we want to handle dtypes, currently enforcing the initially set dtype, easily changed
def _convert_array_dtype(
    value: np.ndarray | cp.ndarray, self_: Any, field: Attribute
) -> np.ndarray | cp.ndarray:
    if not isinstance(value, (np.ndarray, cp.ndarray)):
        raise ValueError(f"{field.name} must be a np.ndarray or cp.ndarray")
    if hasattr(self_, "array"):
        return value.astype(self_.dtype)
    else:
        return value


def _validate_array(
    instance: Any, attribute: Attribute, value: np.ndarray | cp.ndarray
) -> None:
    if not hasattr(instance, "array"):
        return
    current_ndim = getattr(instance, "ndim")
    if value.ndim != current_ndim:
        raise ValueError(
            f"ndim of array, {value.ndim}, must equal current: {current_ndim}"
        )


def _convert_ndinfo(
    value: np.ndarray | tuple | list, self_: Any, field: Attribute
) -> np.ndarray:
    if not isinstance(value, (np.ndarray, tuple, list)):
        raise TypeError(
            f"{field.name} should a ndarray/list/tuple. Got type {type(value)}"
        )
    return np.array(value)


def _validate_ndinfo(instance: Any, attribute: Attribute, value: np.ndarray) -> None:
    """Confirm dimension of origin, sampling, units"""
    # could do if len(value) != instance.ndim here cuz array has been set, but fails with autoserialize
    if not hasattr(instance, "array"):
        return
    current_ndim = getattr(instance, "ndim")
    if len(value) != current_ndim:
        raise ValueError(
            f"Setting {attribute.name} of length {len(value)} which does not match array dimension {current_ndim}"
        )


@define
class Dataset(AutoSerialize):
    """
    A class representing a multi-dimensional dataset with metadata.

    The Dataset class wraps an n-dimensional array (either numpy or cupy) and provides
    metadata about the array's dimensions, units, and sampling. It supports automatic
    serialization through inheritance from AutoSerialize.

    Attributes:
        array (np.ndarray | cp.ndarray): The underlying n-dimensional array data
        name (str): A descriptive name for the dataset
        origin (np.ndarray): The origin coordinates for each dimension
        sampling (np.ndarray): The sampling rate/spacing for each dimension
        units (list[str]): Units for each dimension (e.g. "nm", "eV", etc.)
        signal_units (str): Units for the array values
    """

    array: np.ndarray | cp.ndarray = field(
        converter=Converter(_convert_array_dtype, takes_self=True, takes_field=True),
        validator=_validate_array,
    )
    name: str = field(
        default=Factory(lambda self: f"{self.array.ndim}d dataset", takes_self=True)
    )
    origin: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.array.ndim), takes_self=True),
        converter=Converter(_convert_ndinfo, takes_self=True, takes_field=True),
        validator=_validate_ndinfo,
    )
    sampling: np.ndarray = field(
        default=Factory(lambda self: np.zeros(self.array.ndim), takes_self=True),
        converter=np.ndarray,
        validator=_validate_ndinfo,
    )
    units: list[str] = field(
        default=Factory(lambda self: ["pixels"] * self.array.ndim, takes_self=True),
        converter=list[str],
        validator=_validate_ndinfo,
    )
    signal_units: str = field(default="arb. units", converter=str)

    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def device(self) -> str:
        """
        Outputting a string is likely temporary -- once we have our use cases we can
        figure out a more permanent device solution that enables easier translation between
        numpy <-> cupy <-> torch <-> numpy
        """
        return str(self.array.device)

    # Summaries
    def __repr__(self):
        description = [
            f"Dataset(shape={self.shape}, dtype={self.dtype}, name={self.name},)",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: {self.signal_units}",
        ]
        return "\n".join(description)

    def __str__(self):
        description = [
            f"quantem Dataset named {self.name}",
            f"  shape: {self.shape}",
            f"  dtype: {self.dtype}",
            f"  origin: {self.origin}",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: {self.signal_units}",
        ]
        return "\n".join(description)

    def copy(self, copy_attributes=False):  # TODO
        """
        Copies Dataset

        Parameters
        ----------
        copy_attributes: bool
            If True, copies attributes
        """
        dataset = Dataset(
            array=self.array.copy(),
            name=self.name,
            origin=self.origin.copy(),
            sampling=self.sampling.copy(),
            units=self.units,
            signal_units=self.signal_units,
        )

        if copy_attributes:
            for attr in vars(self):
                if not hasattr(dataset, attr):
                    try:
                        setattr(dataset, attr, getattr(self, attr).copy())
                    except AttributeError:
                        pass
        return dataset

    def mean(self, axes: tuple | None = None):
        """
        Computes and returns mean of Dataset

        Parameters
        ----------
        axes: tuple
            Axes over which to compute mean. If None specified, all axes are used.

        Returns
        --------
        mean: Dataset
            Mean of Dataset
        """
        mean = self.array.mean(axis=axes)
        return mean

    def max(self, axes: tuple | None = None):
        """
        Computes and returns max of Dataset

        Parameters
        ----------
        axes: tuple
            Axes over which to compute max. If None specified, all axes are used.

        Returns
        --------
        maximum: Dataset
            Maximum of Dataset
        """
        maximum = self.array.max(axis=axes)
        return maximum

    def min(self, axes: tuple | None = None):
        """
        Computes and returns min of Dataset

        Parameters
        ----------
        axes: tuple
            Axes over which to compute min. If None specified, all axes are used.

        Returns
        --------
        minimum: Dataset
            Minimum of Dataset
        """
        minimum = self.array.min(axis=axes)
        return minimum

    def pad(self, pad_width: tuple, modify_in_place: bool = False, **kwargs):
        """
        Pads Dataset

        Parameters
        ----------
        pad_width: tuple
            Number of values padded to the edges of each axis. `((before_1, after_1), ... (before_N, after_N))`
            unique pad widths for each axis. `(before, after)` or `((before, after),)` yields same before and
            after pad for each axis. `(pad,)` or `int` is a shortcut for before = after = pad width for all axes.
        modify_in_place: bool
            If True, modifies dataset

        Returns
        --------
        Dataset (padded) only if modify_in_place is False

        """
        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = np.pad(dataset.array, pad_width=pad_width, **kwargs)
            return dataset
        else:
            self.array = np.pad(self.array, pad_width=pad_width, **kwargs)

    def crop(self, crop_widths, axes=None, modify_in_place=False):
        """
        Crops Dataset

        Parameters
        ----------
        crop_widths:tuple
            Min and max for cropping each axis specified as a tuple
        axes:
            Axes over which to crop. If None specified, all are cropped.
        modify_in_place: bool
            If True, modifies dataset

        Returns
        --------
        Dataset (cropped) only if modify_in_place is False
        """
        if axes is None:
            if len(crop_widths) != self.ndim:
                raise ValueError(
                    "crop_widths must match number of dimensions when axes is None."
                )
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (axes,)
            crop_widths = (crop_widths,)
        else:
            axes = tuple(axes)

        if len(crop_widths) != len(axes):
            raise ValueError("Length of crop_widths must match length of axes.")

        full_slices = []
        crop_dict = dict(zip(axes, crop_widths))
        for axis, dim in enumerate(self.shape):
            if axis in crop_dict:
                before, after = crop_dict[axis]
                start = before
                stop = dim - after if after != 0 else None
                full_slices.append(slice(start, stop))
            else:
                full_slices.append(slice(None))
        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = dataset.array[tuple(full_slices)]
            return dataset
        else:
            self.array = self.array[tuple(full_slices)]

    def bin(
        self,
        bin_factors,
        axes=None,
        modify_in_place=False,
    ):
        """
        Bins Dataset

        Parameters
        ----------
        bin_factors:tuple or int
            bin factors for each axis
        axes:
            Axis over which to bin. If None is specified, all axes are binned.
        modify_in_place: bool
            If True, modifies dataset

        Returns
        --------
        Dataset (binned) only if modify_in_place is False
        """
        if axes is None:
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (axes,)

        if isinstance(bin_factors, int):
            bin_factors = tuple([bin_factors] * len(axes))
        elif isinstance(bin_factors, (list, tuple)):
            if len(bin_factors) != len(axes):
                raise ValueError("bin_factors and axes must have the same length.")
            bin_factors = tuple(bin_factors)
        else:
            raise TypeError("bin_factors must be an int or tuple of ints.")

        axis_to_factor = dict(zip(axes, bin_factors))

        slices = []
        new_shape = []
        for axis in range(self.ndim):
            if axis in axis_to_factor:
                factor = axis_to_factor[axis]
                length = self.shape[axis] - (self.shape[axis] % factor)
                slices.append(slice(0, length))
                new_shape.extend([length // factor, factor])
            else:
                slices.append(slice(None))
                new_shape.append(self.shape[axis])

        reshape_dims = []
        reduce_axes = []
        current_axis = 0

        for axis in range(self.ndim):
            if axis in axis_to_factor:
                reshape_dims.extend([new_shape[current_axis], axis_to_factor[axis]])
                reduce_axes.append(len(reshape_dims) - 1)
                current_axis += 2
            else:
                reshape_dims.append(new_shape[current_axis])
                current_axis += 1

        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = np.sum(
                dataset.array[tuple(slices)].reshape(reshape_dims),
                axis=tuple(reduce_axes),
            )
            return dataset
        else:
            self.array = np.sum(
                self.array[tuple(slices)].reshape(reshape_dims), axis=tuple(reduce_axes)
            )

    def show(
        self,
        scalebar: ScalebarConfig | bool = True,
        title: str | None = None,
        **kwargs,
    ):
        """
        Displays Dataset as a 2D image

        Parameters
        ----------
        scalebar: ScalebarConfig or bool
            If True, displays scalebar
        title: str
            Title of Dataset
        kwargs: dict
            Keyword arguments for show_2d
        """
        if self.ndim != 2:
            raise NotImplementedError()  # base class only provides 2D. subclasses can override.

        if scalebar is True:
            scalebar = ScalebarConfig(
                sampling=self.sampling[-1],
                units=self.units[-1],
            )

        if title is None:
            title = self.name

        return show_2d(self.array, scalebar=scalebar, title=title, **kwargs)
