from typing import Self, Tuple

import torch
import torch.nn.functional as F

from quantem.core.datastructures import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.diffractive_imaging.ptychography.ptychography_utils import (
    compute_propagator_array,
    fourier_convolve_array,
    sum_overlapping_patches,
)


class ObjectModelBase(AutoSerialize):
    """
    Base class for all ObjectModels to inherit from.
    """

    def forward(self, *args):
        raise NotImplementedError()

    def backward(self, *args):
        raise NotImplementedError()

    @property
    def tensor(self) -> torch.Tensor:
        return self.dataset.array


class PixelatedObjectModel(ObjectModelBase):
    """ """

    _token = object()

    def __init__(
        self,
        obj_dataset: Dataset3d,
        energy,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use PixelatedObjectModel.from_array() or PixelatedObjectModel.from_positions() to instantiate this class."
            )

        self.dataset = obj_dataset
        self.energy = energy
        self.obj_shape = self.dataset.shape[-2:]
        self.sampling = self.dataset.sampling[-2:]
        self.num_slices = self.dataset.shape[0]
        self.slice_thickness = self.dataset.sampling[0]
        self.propagator_array = compute_propagator_array(
            energy, self.obj_shape, self.sampling, self.slice_thickness
        )

    @classmethod
    def from_array(
        cls,
        array: torch.Tensor,
        energy: float,
        sampling: Tuple[int, int],
        slice_thickness: float,  # TODO: support varying slice-thicknesses
    ) -> Self:
        obj_dataset = Dataset3d.from_array(
            torch.as_tensor(array).to(torch.cfloat),
            name="ptychographic object",
            sampling=(slice_thickness,) + tuple(sampling),
            units=("A", "A", "A"),
        )

        return cls(
            obj_dataset,
            energy,
            cls._token,
        )

    @classmethod
    def from_positions(
        cls,
        energy: float,
        positions_px: torch.Tensor,
        padding_px: Tuple[int, int, int, int],
        sampling: Tuple[int, int],
        slice_thickness: float,
        num_slices: int,
    ) -> Self:
        """ """

        bbox = positions_px.max(dim=0).values - positions_px.min(dim=0).values
        bbox = torch.round(bbox).to(torch.int)
        obj = torch.ones(*bbox, dtype=torch.cfloat)
        obj = torch.tile(F.pad(obj, padding_px, value=1.0), (num_slices, 1, 1))

        return cls.from_array(
            obj,
            energy,
            sampling,
            slice_thickness,
        )

    def forward(
        self,
        probe_array: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
    ):
        """ """
        num_slices = self.num_slices
        propagator = self.propagator_array
        obj_patches = self.tensor[..., row, col]

        propagated_probes = torch.empty_like(obj_patches)
        propagated_probes[0] = probe_array

        for s in range(num_slices):
            exit_waves = obj_patches[s] * propagated_probes[s]
            if s + 1 < num_slices:
                propagated_probes[s + 1] = fourier_convolve_array(
                    exit_waves, propagator
                )

        return propagated_probes, obj_patches, exit_waves

    def backward(self, gradient_array, probe_array, obj_patches, positions_px):
        """ """
        if self.tensor.requires_grad:
            num_slices = self.num_slices
            propagator = self.propagator_array.conj()
            obj_gradient = torch.empty_like(self.tensor)

            for s in reversed(range(num_slices)):
                probe = probe_array[s]
                obj = obj_patches[s]

                probe_normalization = (
                    sum_overlapping_patches(
                        torch.square(torch.abs(probe)), positions_px, self.obj_shape
                    )
                    + 1e-10
                )

                obj_gradient[s] = (
                    sum_overlapping_patches(
                        gradient_array * torch.conj(probe),
                        positions_px,
                        self.obj_shape,
                    )
                    / probe_normalization
                )

                if s > 0:
                    gradient_array *= torch.conj(obj)  # back-transmit
                    gradient_array = fourier_convolve_array(
                        gradient_array, propagator
                    )  # back-propagate

            self.tensor.grad = obj_gradient.clone().detach()

            return gradient_array

    @property
    def tensor(self) -> torch.Tensor:
        return self.dataset.array
