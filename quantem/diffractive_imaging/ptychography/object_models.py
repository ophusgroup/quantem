from typing import Self, Tuple

import torch
import torch.nn.functional as F

from quantem.core.datastructures import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.diffractive_imaging.ptychography.ptychography_utils import (
    sum_overlapping_patches,
)


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
            probe_normalization = (
                sum_overlapping_patches(
                    torch.square(torch.abs(probe_array)), positions_px, self.obj_shape
                )
                + 1e-10
            )

            obj_gradient = (
                sum_overlapping_patches(
                    gradient_array * torch.conj(probe_array),
                    positions_px,
                    self.obj_shape,
                )
                / probe_normalization
            )

            self.tensor.grad = obj_gradient.clone().detach()

            return obj_gradient

    @property
    def tensor(self) -> torch.Tensor:
        return self.dataset.array
