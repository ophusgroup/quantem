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
        data: np.ndarray | cp.ndarray,
        name: str | None = None,
        origin: list | None = None,
        sampling: list | None = None,
        units: list[str] | None = None,
        signal_units: str | None = None,
    ):
        self.array = data
        self.name = f"{data.ndim}d dataset" if name is None else name
        self.origin = np.zeros(data.ndim) if origin is None else origin
        self.sampling = np.zeros(data.ndim) if sampling is None else sampling
        self.units = ["pixels"] * data.ndim if units is None else units
        self.signal_units = "arb. units" if signal_units is None else signal_units

    # Properties
    @property
    def array(self) -> np.ndarray | cp.ndarray:
        return self._array

    @array.setter
    def array(self, arr):
        if isinstance(arr, np.ndarray):
            self._array = arr
        elif config.get("has_cupy"):
            if isinstance(arr, cp.ndarray):
                self._array = arr
        elif isinstance(arr, (list, tuple)):
            self._array = np.array(arr)
        else:
            raise TypeError(f"bad type{type(arr)}")

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
    def origin(self, val: np.ndarray | list):
        origin = np.array(val)
        if origin.ndim != self.ndim:
            raise ValueError(
                f"Got origin dimension {origin.ndim} which does not match data dimension {self.ndim}"
            )
        self._origin = origin

    @property
    def sampling(self) -> np.ndarray:
        return self._sampling

    @sampling.setter
    def sampling(self, val: np.ndarray | list):
        sampling = np.array(val)
        if sampling.ndim != self.ndim:
            raise ValueError(
                f"Got sampling dimension {sampling.ndim} which does not match data dimension {self.ndim}"
            )
        self._sampling = sampling

    @property
    def units(self) -> list[str]:
        return self._units

    @units.setter
    def units(self, val: np.ndarray | list):
        units = [str(v) for v in val]
        if len(units) != self.ndim:
            raise ValueError(
                f"Got units dimension {len(units)} which does not match data dimension {self.ndim}"
            )
        self._units = units

    @property
    def signal_units(self) -> str:
        return self._signal_units

    @signal_units.setter
    def signal_units(self, val: str):
        self._signal_units = str(val)

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def device(self):
        return self.array.device

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

    def mean(self, axes=None):
        if axes is None:
            axes = tuple(np.arange(self.ndim))
        mean = self.array.mean((axes))
        return mean

    def max(self, axes=None):
        if axes is None:
            axes = tuple(np.arange(self.ndim))
        maximum = self.array.max((axes))
        return maximum

    def min(self, axes=None):
        if axes is None:
            axes = tuple(np.arange(self.ndim))
        minimum = self.array.max((axes))
        return minimum
