from quantem.core.io.serialize import AutoSerialize
import numpy as np


# base class for quantem datasets
class Dataset(AutoSerialize):
    def __init__(
        self,
        data,
        origin = None,
        sampling = None,
        units = None,
    ):
        self.array = data

        if origin is None:
            self.origin = np.zeros(data.ndim)

        if sampling is None:
            self.sampling = np.ones(data.ndim)

        if units is None:
            self.units = ['pixels']*data.ndim

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype



