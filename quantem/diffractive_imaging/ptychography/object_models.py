from typing import Optional, Self, Tuple

import torch
import torch.nn.functional as F

from quantem.core.datastructures import Dataset2d
from quantem.core.io.serialize import AutoSerialize


def sum_overlapping_patches(
    patches: torch.Tensor,
    positions_px: torch.Tensor,
    obj_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Sums overlapping patches into a global array using scatter_add, supports complex inputs.

    Parameters
    ----------
    patches : (N, sx, sy) torch.Tensor (real or complex)
        Array of N patches to be summed.
    positions_px : (N, 2) torch.Tensor
        Integer (x, y) positions for each patch.
    object_shape : (Hx, Hy)
        Shape of the full object array.

    Returns
    -------
    summed : (Hx, Hy) torch.Tensor
        Accumulated object array.
    """

    device = patches.device
    dtype = patches.dtype

    N, sx, sy = patches.shape
    Hx, Hy = obj_shape

    x0 = positions_px[:, 0].round().to(torch.long)
    y0 = positions_px[:, 1].round().to(torch.long)

    dx = torch.fft.fftfreq(sx, d=1 / sx, device=device).to(torch.long)
    dy = torch.fft.fftfreq(sy, d=1 / sy, device=device).to(torch.long)
    dx_grid, dy_grid = torch.meshgrid(dx, dy, indexing="ij")

    x_idx = (x0[:, None, None] + dx_grid[None, :, :]) % Hx
    y_idx = (y0[:, None, None] + dy_grid[None, :, :]) % Hy

    flat_indices = x_idx * Hy + y_idx
    flat_indices = flat_indices.reshape(-1)
    flat_weights = patches.reshape(-1)

    summed = torch.zeros(Hx * Hy, dtype=dtype, device=device)
    summed = summed.scatter_add(0, flat_indices, flat_weights)

    return summed.reshape(Hx, Hy)


class ObjectModelBase(AutoSerialize):
    """
    Base class for all ObjectModels to inherit from.
    """

    def initialize(self, *args):
        raise NotImplementedError()

    def forward(self, *args):
        raise NotImplementedError()

    def backward(self, *args):
        raise NotImplementedError()

    @property
    def tensor(self) -> torch.Tensor:
        return self.dataset.array

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

    _token = object()

    def __init__(
        self,
        obj_dataset: Dataset2d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use ComplexSingleSliceObjectModel.from_array() or ComplexSingleSliceObjectModel.from_positions() to instantiate this class."
            )

        self.dataset = obj_dataset
        self.obj_shape = self.dataset.shape

    @classmethod
    def from_array(
        cls,
        array: torch.Tensor,
        sampling: Tuple[int, int],
    ) -> Self:
        obj_dataset = Dataset2d.from_array(
            torch.as_tensor(array).to(torch.cfloat),
            name="ptychographic object",
            sampling=sampling,
            units=("A", "A"),
        )

        return cls(
            obj_dataset,
            cls._token,
        )

    @classmethod
    def from_positions(
        cls,
        positions_px: torch.Tensor,
        padding_px: Tuple[int, int, int, int],
        sampling: Tuple[int, int],
    ) -> Self:
        """ """

        bbox = positions_px.max(dim=0).values - positions_px.min(dim=0).values
        bbox = torch.round(bbox).to(torch.int)
        obj = torch.ones(*bbox, dtype=torch.cfloat)
        obj = F.pad(obj, padding_px, value=1.0)

        return cls.from_array(
            obj,
            sampling,
        )

    def forward(
        self,
        probe_array: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
    ):
        obj_patches = self.tensor[row, col]
        exit_waves = obj_patches * probe_array
        return obj_patches, exit_waves

    def backward(self, gradient_array, probe_array, positions_px):
        if self.tensor.requires_grad:
            obj_gradient = sum_overlapping_patches(
                gradient_array * torch.conj(probe_array), positions_px, self.obj_shape
            )

            self.tensor.grad = obj_gradient.clone().detach()

            return obj_gradient

    @property
    def tensor(self) -> torch.Tensor:
        return self.dataset.array
