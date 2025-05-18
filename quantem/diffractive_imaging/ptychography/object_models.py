from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from quantem.core.datastructures import Dataset2d


class ObjectModelBase(ABC):
    """
    Base class for all ObjectModels to inherit from.
    """

    @abstractmethod
    def initialize_object(self, *args):
        pass

    @abstractmethod
    def forward_object(self, *args):
        pass

    @abstractmethod
    def backward_object(self, *args):
        pass

    def return_patch_indices(
        self,
        positions_px: torch.Tensor,
        roi_shape: Tuple[int, int],
        obj_shape: Tuple[int, int],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute wrapped patch indices into the object array for each probe position.
        Note this assumes corner-centered probes.

        Parameters
        ----------
        positions_px : torch.Tensor
            Tensor of shape (N, 2), float32. Probe positions in pixels.
        roi_shape : tuple of int
            (Sx, Sy), the shape of the probe (patch).
        obj_shape : tuple of int
            (Hx, Hy), the shape of the object array.
        device : torch.device, optional
            Where to place tensors. If None, inferred from positions_px.

        Returns
        -------
        row : torch.Tensor
            (N, Sx, Sy), int64 tensor of row indices.
        col : torch.Tensor
            (N, Sx, Sy), int64 tensor of col indices.
        """
        if device is None:
            device = positions_px.device

        # Round and convert to int
        x0: torch.Tensor = torch.round(positions_px[:, 0]).to(torch.int64)
        y0: torch.Tensor = torch.round(positions_px[:, 1]).to(torch.int64)

        # Frequency-based index grid
        x_ind: torch.Tensor = torch.fft.fftfreq(
            roi_shape[0], d=1.0 / roi_shape[0], device=device
        ).to(torch.int64)
        y_ind: torch.Tensor = torch.fft.fftfreq(
            roi_shape[1], d=1.0 / roi_shape[1], device=device
        ).to(torch.int64)

        # Broadcast and wrap
        row: torch.Tensor = (x0[:, None, None] + x_ind[None, :, None]) % obj_shape[0]
        col: torch.Tensor = (y0[:, None, None] + y_ind[None, None, :]) % obj_shape[1]

        return row, col


class ComplexSingleSliceObjectModel(ObjectModelBase):
    """ """

    def __init__(
        self,
        positions_px: torch.Tensor,
        padding_px: Tuple[int, int, int, int],
        sampling: Tuple[int, int],
    ):
        """ """

        self.dataset = Dataset2d.from_array(
            self.initialize_object(positions_px, padding_px),
            name="ptychographic object",
            sampling=sampling,
            units=("A", "A"),
        )

    def initialize_object(
        self, positions_px: torch.Tensor, padding_px: Tuple[int, int, int, int]
    ):
        """ """

        bbox = positions_px.max(dim=0).values - positions_px.min(dim=0).values
        bbox = torch.round(bbox).to(torch.int)
        obj = torch.ones(*bbox, dtype=torch.complex64)
        obj = F.pad(obj, padding_px, value=1.0)
        obj.requires_grad = True

        return obj

    def forward_object(
        self,
        probe_array: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
    ):
        obj_patches = self.dataset.array[row, col]
        exit_waves = obj_patches * probe_array
        return obj_patches, exit_waves

    def backward_object(self, gradient_array, positions_px, *args):
        pass
