from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from quantem.diffractive_imaging.ptychography_base import PtychographyBase

if TYPE_CHECKING:
    import cupy as cp
    import torch

ArrayLike = Union[np.ndarray, "cp.ndarray", "torch.Tensor"]


# TODO make dataclass
class PtychographyVisualizations(PtychographyBase):
    def show_object(self, obj: ArrayLike = None):
        if obj is None:
            obj = self.obj
        else:
            obj = self._to_numpy(obj)

        pass

    def show_probe(self):
        pass

    def show_fourier_probe(self):
        pass

    def show_object_and_probe(self):
        pass

    def visualize(self):
        # losses, object, probe
        pass

    def plot_losses(self):
        pass

    def show_object_epochs(self):
        # image grid
        pass

    def show_object_slices(self):
        # image grid
        pass
