from collections.abc import Iterable

import numpy as np

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


# base class for quantem datasets
class Dataset(AutoSerialize):
    def __init__(
        self,
        array: np.ndarray | cp.ndarray,
        name: str | None = None,
        origin: np.ndarray | list | None = None,
        sampling: np.ndarray | list | None = None,
        units: list[str] | None = None,
        signal_units: str | None = None,
    ):
        self.array = array
        self.name = f"{array.ndim}d arrayset" if name is None else name
        self.origin = np.zeros(array.ndim) if origin is None else origin
        self.sampling = np.zeros(array.ndim) if sampling is None else sampling
        self.units = ["pixels"] * array.ndim if units is None else units
        self.signal_units = "arb. units" if signal_units is None else signal_units

    # Properties
    @property
    def array(self) -> np.ndarray | cp.ndarray:
        return self._array

    @array.setter
    def array(self, arr):
        if isinstance(arr, np.ndarray):
            pass
        elif config.get("has_cupy"):
            if isinstance(arr, cp.ndarray):
                pass
        elif isinstance(arr, (list, tuple)):
            arr = np.array(arr)
        else:
            raise TypeError(f"bad type{type(arr)}")

        if hasattr(self, "_array"):
            if arr.ndim != self.ndim:
                raise ValueError(
                    f"Dimension of new array, {arr.ndim}, must equal current ndim: {self.ndim}"
                )
            self._array = arr.astype(self.dtype)
        else:
            self._array = arr

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str):
        self._name = str(val)

    @property
    def origin(self) -> np.ndarray:
        return self._origin

    @origin.setter
    def origin(self, val: np.ndarray | list | tuple):
        if not isinstance(val, Iterable):
            raise TypeError(
                f"origin should be set with a ndarray/list/tuple. Got type {type(val)}"
            )
        origin = np.array(val)
        if len(origin) != self.ndim:
            raise ValueError(
                f"Got origin length {len(origin)} which does not match array dimension {self.ndim}"
            )
        self._origin = origin

    @property
    def sampling(self) -> np.ndarray:
        return self._sampling

    @sampling.setter
    def sampling(self, val: np.ndarray | list | tuple):
        if not isinstance(val, Iterable):
            raise TypeError(
                f"sampling should be set with a ndarray/list/tuple. Got type {type(val)}"
            )
        sampling = np.array(val)
        if len(sampling) != self.ndim:
            raise ValueError(
                f"Got sampling length {len(sampling)} which does not match array dimension {self.ndim}"
            )
        self._sampling = sampling

    @property
    def units(self) -> list[str]:
        return self._units

    @units.setter
    def units(self, val: list | tuple):
        if not isinstance(val, Iterable):
            raise TypeError(
                f"units should be set with a ndarray/list/tuple. Got type {type(val)}"
            )
        units = [str(v) for v in val]
        if len(units) != self.ndim:
            raise ValueError(
                f"Got units length {len(units)} which does not match array dimension {self.ndim}"
            )
        self._units = units

    @property
    def signal_units(self) -> str:
        return self._signal_units

    @signal_units.setter
    def signal_units(self, val: str):
        self._signal_units = str(val)

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

    def copy(self):
        dataset = Dataset(
            array=self.array.copy(),
            name=self.name,
            origin=self.origin.copy(),
            sampling=self.sampling.copy(),
            units=self.units,
            signal_units=self.units,
        )
        return dataset

    def mean(self, axes=None):
        if axes is None:
            axes = tuple(np.arange(self.ndim))
        mean = self.array.mean(axis=axes)
        return mean

    def max(self, axes=None):
        if axes is None:
            axes = tuple(np.arange(self.ndim))
        maximum = self.array.max(axis=axes)
        return maximum

    def min(self, axes=None):
        if axes is None:
            axes = tuple(np.arange(self.ndim))
        minimum = self.array.max(axis=axes)
        return minimum

    def pad(self, pad_width, modify_in_place=False, **kwargs):
        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = np.pad(dataset.array, pad_width=pad_width, **kwargs)
            return dataset
        else:
            self.array = np.pad(self.array, pad_width=pad_width, **kwargs)

    def crop(self, crop_widths, modify_in_place=False):
        if len(crop_widths) != self.ndim:
            raise ValueError(
                "Length of crop_widths must match number of array dimensions."
            )
        slices = tuple(
            slice(before, dim - after if after != 0 else None)
            for (before, after), dim in zip(crop_widths, self.shape)
        )

        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = dataset.array[slices]
            return dataset
        else:
            self.array = self.array[slices]
