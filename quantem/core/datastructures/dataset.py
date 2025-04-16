from typing import TYPE_CHECKING

import numpy as np

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize

if TYPE_CHECKING:
    import cupy as cp
else:
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
        if name is None:
            self.name = f"{data.ndim}d dataset"
        else:
            self.name = name
        if origin is None:
            self.origin = np.zeros(data.ndim)
        else:
            self.origin = origin
        if sampling is None:
            self.sampling = np.ones(data.ndim)
        else:
            self.sampling = sampling
        if units is None:
            self.units = ["pixels"] * data.ndim
        else:
            self.units = units
        if signal_units is None:
            self.signal_units = "arb. units"
        else:
            self.signal_units = signal_units

    # Properties
    @property
    def array(self) -> np.ndarray | cp.ndarray:
        return self._array

    @array.setter
    def array(self, arr):
        if isinstance(arr, (np.ndarray, cp.ndarray)):
            self._array = arr
        elif isinstance(arr, (list, tuple)):
            self._array = np.array(arr)
        else:
            raise TypeError

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
