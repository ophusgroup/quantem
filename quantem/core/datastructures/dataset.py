from functools import wraps
from typing import Any, Callable, TypeVar, cast

import numpy as np

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization.visualization import show_2d
from quantem.core.visualization.visualization_utils import ScalebarConfig

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp

T = TypeVar("T")


# --- Validation Decorator ---
def validated(validator_factory: Callable) -> Callable:
    """
    Decorator for property-style validated fields.

    Example:
        @validated(lambda self: pipe(ensure_int, is_positive))
        def my_attr(self): ...
    """

    def decorator(func):
        attr_name = func.__name__
        private_name = f"_{attr_name}"

        @property
        def prop(self):
            return getattr(self, private_name)

        @prop.setter
        def prop(self, value):
            validator = validator_factory(self)
            validated_value = validator(value)
            setattr(self, private_name, validated_value)

        return prop

    return decorator


# --- Validator composition ---
def pipe(*funcs: Callable) -> Callable:
    """Compose multiple validator/converter functions into one."""

    def composed(val):
        for f in funcs:
            val = f(val)
        return val

    return composed


# --- Dataset-specific validators and converters ---
def ensure_array(value: Any) -> np.ndarray | cp.ndarray:
    """Convert value to numpy or cupy array."""
    if not isinstance(value, (np.ndarray, cp.ndarray)):
        raise ValueError("Value must be a numpy or cupy array")
    return value


def ensure_array_dtype(
    value: np.ndarray | cp.ndarray, dtype: np.dtype | None = None
) -> np.ndarray | cp.ndarray:
    """Ensure array has the correct dtype."""
    if dtype is not None:
        return value.astype(dtype)
    return value


def ensure_ndinfo(value: Any) -> np.ndarray:
    """Convert value to numpy array for ndinfo fields (origin, sampling)."""
    if not isinstance(value, (np.ndarray, tuple, list)):
        raise TypeError(f"Value should be a ndarray/list/tuple. Got type {type(value)}")
    return np.array(value)


def ensure_units(value: Any) -> list[str]:
    """Convert value to list of strings for units."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Units must be a list or tuple. Got type {type(value)}")
    return [str(unit) for unit in value]


def ensure_str(value: Any) -> str:
    """Convert value to string."""
    return str(value)


def validate_array_dimensions(
    value: np.ndarray | cp.ndarray, ndim: int | None = None
) -> np.ndarray | cp.ndarray:
    """Validate array dimensions match expected ndim."""
    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"Array dimension {value.ndim} must equal expected dimension {ndim}"
        )
    return value


def validate_ndinfo_length(value: np.ndarray, ndim: int | None = None) -> np.ndarray:
    """Validate ndinfo array length matches expected ndim."""
    if ndim is not None and len(value) != ndim:
        raise ValueError(f"Length {len(value)} must match dimension {ndim}")
    return value


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

    _token = object()

    def __init__(self, _token=None):
        if _token is not self._token:
            raise RuntimeError("Use Dataset.from_array() to instantiate this class.")
        self._array = None
        self._name = None
        self._origin = None
        self._sampling = None
        self._units = None
        self._signal_units = "arb. units"

    @classmethod
    def from_array(
        cls,
        array: np.ndarray | cp.ndarray,
        name: str | None = None,
        origin: np.ndarray | tuple | list | None = None,
        sampling: np.ndarray | tuple | list | None = None,
        units: list[str] | None = None,
        signal_units: str = "arb. units",
    ) -> "Dataset":
        """
        Create a new Dataset from an array.

        Parameters
        ----------
        array : np.ndarray | cp.ndarray
            The underlying n-dimensional array data
        name : str | None, optional
            A descriptive name for the dataset. If None, defaults to "{array.ndim}d dataset"
        origin : np.ndarray | tuple | list | None, optional
            The origin coordinates for each dimension. If None, defaults to zeros
        sampling : np.ndarray | tuple | list | None, optional
            The sampling rate/spacing for each dimension. If None, defaults to zeros
        units : list[str] | None, optional
            Units for each dimension. If None, defaults to ["pixels"] * array.ndim
        signal_units : str, optional
            Units for the array values, by default "arb. units"

        Returns
        -------
        Dataset
            A new Dataset instance
        """
        dataset = cls(_token=cls._token)
        dataset.array = array
        dataset.name = name if name is not None else f"{array.ndim}d dataset"
        dataset.origin = origin if origin is not None else np.zeros(array.ndim)
        dataset.sampling = sampling if sampling is not None else np.zeros(array.ndim)
        dataset.units = units if units is not None else ["pixels"] * array.ndim
        dataset.signal_units = signal_units
        return dataset

    @validated(
        lambda self: pipe(
            ensure_array,
            lambda x: ensure_array_dtype(x, getattr(self, "_dtype", None)),
            lambda x: validate_array_dimensions(x, getattr(self, "_ndim", None)),
        )
    )
    def array(self) -> np.ndarray | cp.ndarray:
        """The underlying n-dimensional array data."""
        pass

    @validated(lambda self: ensure_str)
    def name(self) -> str:
        """A descriptive name for the dataset."""
        pass

    @validated(
        lambda self: pipe(
            ensure_ndinfo,
            lambda x: validate_ndinfo_length(x, getattr(self, "_ndim", None)),
        )
    )
    def origin(self) -> np.ndarray:
        """The origin coordinates for each dimension."""
        pass

    @validated(
        lambda self: pipe(
            ensure_ndinfo,
            lambda x: validate_ndinfo_length(x, getattr(self, "_ndim", None)),
        )
    )
    def sampling(self) -> np.ndarray:
        """The sampling rate/spacing for each dimension."""
        pass

    @validated(
        lambda self: pipe(
            ensure_units,
            lambda x: validate_ndinfo_length(x, getattr(self, "_ndim", None)),
        )
    )
    def units(self) -> list[str]:
        """Units for each dimension (e.g. "nm", "eV", etc.)."""
        pass

    @validated(lambda self: ensure_str)
    def signal_units(self) -> str:
        """Units for the array values."""
        pass

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
        dataset = Dataset.from_array(self.array.copy())
        dataset.name = self.name
        dataset.origin = self.origin.copy()
        dataset.sampling = self.sampling.copy()
        dataset.units = self.units
        dataset.signal_units = self.signal_units

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
