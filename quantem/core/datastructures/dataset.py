from quantem.core.io.serialize import AutoSerialize
import numpy as np


# base class for quantem datasets
class Dataset(AutoSerialize):
    def __init__(
        self,
        data,
        name=None,
        origin=None,
        sampling=None,
        units=None,
        signal_units=None,
    ):
        self.array = data
        if name is None:
            self.name = f"{data.ndim}d dataset"
        if origin is None:
            self.origin = np.zeros(data.ndim)
        if sampling is None:
            self.sampling = np.ones(data.ndim)
        if units is None:
            self.units = ["pixels"] * data.ndim
        if signal_units is None:
            self.signal_units = "arb. units"

    # Properties
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
