from abc import abstractmethod
from typing import Any

import torch

from quantem.core.io.serialize import AutoSerialize


class ObjectBase(AutoSerialize):
    """
    Base class for all ObjectModels to inherit from.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        device: str,
        offset_obj: float = 1e-5,
    ):
        self._shape = volume_shape

        self._obj = torch.zeros(self._shape, device=device, dtype=torch.float32) + offset_obj

        self._device = device
        self._constraints = {}

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @shape.setter
    def shape(self, shape: tuple[int, int, int]):
        self._shape = shape

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str):
        self._device = device

    @abstractmethod
    def forward(
        self, z1: torch.Tensor, z3: torch.Tensor, shift_x: torch.Tensor, shift_y: torch.Tensor
    ):
        pass

    @abstractmethod
    def obj(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def to(self, device: str):
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class ObjectConstraints(ObjectBase):
    DEFAULT_CONSTRAINTS = {
        "fourier_filter": False,
        "positivity": False,
        "shrinkage": False,
        "circular_mask": False,
    }

    @property
    def constraints(self) -> dict[str, Any]:
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: dict[str, Any]):
        gkeys = self.DEFAULT_CONSTRAINTS.keys()
        for key, value in constraints.items():
            if key not in gkeys:  # This might be redundant since add_constraint is checking.
                raise KeyError(f"Invalid object constraint key '{key}', allowed keys are {gkeys}")
            self._constraints[key] = value

    def add_constraint(self, constraint: str, value: Any):
        """Add constraints to the object model."""
        gkeys = self.DEFAULT_CONSTRAINTS.keys()
        if constraint not in gkeys:
            raise KeyError(
                f"Invalid object constraint key '{constraint}', allowed keys are {gkeys}"
            )
        self._constraints[constraint] = value

    def apply_constraints(
        self,
        obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply constraints to the object model.
        """
        if self.constraints["positivity"]:
            obj2 = torch.clamp(obj, min=0.0, max=None)

        return obj2


class ObjectVoxelwise(ObjectBase, ObjectConstraints):
    """
    Object model for voxelwise objects.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        device: str,
    ):
        super().__init__(
            volume_shape=volume_shape,
            device=device,
        )
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()

    @property
    def obj(self):
        return self.apply_constraints(self._obj)

    def forward(self):
        return self.obj

    def reset(self):
        self._obj = (
            torch.zeros(self._shape, device=self._device, dtype=torch.float32) + self._offset_obj
        )

    def to(self, device: str):
        self._device = device
        self._obj = self._obj.to(self._device)

    @property
    def name(self) -> str:
        return "ObjectVoxelwise"


ObjectModelType = ObjectVoxelwise  # | ObjectDIP | ObjectImplicit (ObjectFFN?)
