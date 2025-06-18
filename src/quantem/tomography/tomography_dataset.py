import numpy as np
from torch._tensor import Tensor

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize


class TomographyDataset(AutoSerialize):
    _token = object()

    """
    A tomography dataset which contains the tilt series, and also instatiates the 
    z1, z3, and shifts of the tilt series.
    
    Idea for this dataset is so that we can avoid moving things around as a torch tensor,
    since the SIRT reconstruction algorthim, and AD reconstruction we have are all torch based.
    """

    def __init__(
        self,
        tilt_series: Tensor,
        z1_angles: Tensor,
        z3_angles: Tensor,
        shifts: Tensor,
    ):
        self.tilt_series = tilt_series
        self.z1_angles = z1_angles
        self.z3_angles = z3_angles
        self.shifts = shifts

    def from_data(
        self,
        tilt_series: Dataset3d | np.ndarray | Tensor,
        z1_angles: np.ndarray | Tensor | None = None,
        z3_angles: np.ndarray | Tensor | None = None,
        shifts: np.ndarray | Tensor | None = None,
        name: str | None = None,
        origin: np.ndarray | tuple | list | float | int | None = None,
        sampling: np.ndarray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        pass
