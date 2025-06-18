from abc import abstractmethod
from typing import Any

import torch

from quantem.core.io.serialize import AutoSerialize
from quantem.tomography.utils import get_TV_loss


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
        self._offset_obj = offset_obj
        self._device = device
        self._hard_constraints = {}
        self._soft_constraints = {}

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @shape.setter
    def shape(self, shape: tuple[int, int, int]):
        self._shape = shape

    @property
    def offset_obj(self) -> float:
        return self._offset_obj

    @offset_obj.setter
    def offset_obj(self, offset_obj: float):
        self._offset_obj = offset_obj

    @property
    def obj(self) -> torch.Tensor:
        pass

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

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

    @abstractmethod
    def params(self) -> torch.Tensor:
        pass


class ObjectConstraints(ObjectBase):
    DEFAULT_HARD_CONSTRAINTS = {
        "fourier_filter": False,
        "positivity": False,
        "shrinkage": False,
        "circular_mask": False,
    }

    DEFAULT_SOFT_CONSTRAINTS = {
        "tv_vol": 0,
    }

    @property
    def hard_constraints(self) -> dict[str, Any]:
        return self._hard_constraints

    @hard_constraints.setter
    def hard_constraints(self, hard_constraints: dict[str, Any]):
        gkeys = self.DEFAULT_HARD_CONSTRAINTS.keys()
        for key, value in hard_constraints.items():
            if key not in gkeys:  # This might be redundant since add_constraint is checking.
                raise KeyError(f"Invalid object constraint key '{key}', allowed keys are {gkeys}")
            self._hard_constraints[key] = value

    @property
    def soft_constraints(self) -> dict[str, Any]:
        return self._soft_constraints

    @soft_constraints.setter
    def soft_constraints(self, soft_constraints: dict[str, Any]):
        gkeys = self.DEFAULT_SOFT_CONSTRAINTS.keys()
        for key, value in soft_constraints.items():
            if key not in gkeys:
                raise KeyError(f"Invalid object constraint key '{key}', allowed keys are {gkeys}")
            self._soft_constraints[key] = value

    def add_hard_constraint(self, constraint: str, value: Any):
        """Add constraints to the object model."""
        gkeys = self.DEFAULT_HARD_CONSTRAINTS.keys()
        if constraint not in gkeys:
            raise KeyError(
                f"Invalid object constraint key '{constraint}', allowed keys are {gkeys}"
            )
        self._hard_constraints[constraint] = value

    def add_soft_constraint(self, constraint: str, value: Any):
        """Add constraints to the object model."""
        gkeys = self.DEFAULT_SOFT_CONSTRAINTS.keys()
        if constraint not in gkeys:
            raise KeyError(
                f"Invalid object constraint key '{constraint}', allowed keys are {gkeys}"
            )
        self._soft_constraints[constraint] = value

    def apply_hard_constraints(
        self,
        obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply constraints to the object model.
        """
        obj2 = obj.clone()
        if self.hard_constraints["positivity"]:
            obj2 = torch.clamp(obj, min=0.0, max=None)
        if self.hard_constraints["shrinkage"]:
            obj2 = torch.max(obj2 - self.hard_constraints["shrinkage"], torch.zeros_like(obj2))

        return obj2

    def apply_soft_constraints(
        self,
        obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        'Applies' soft constraints to the object model. This will return additional loss terms.
        """
        soft_loss = 0.0
        if self.soft_constraints["tv_vol"] > 0:
            tv_loss = get_TV_loss(
                obj.unsqueeze(0).unsqueeze(0), factor=self.soft_constraints["tv_vol"]
            )

            soft_loss += tv_loss

        return soft_loss


class ObjectVoxelwise(ObjectConstraints):
    """
    Object model for voxelwise objects.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        device: str,
        initial_volume: torch.Tensor | None = None,
    ):
        super().__init__(
            volume_shape=volume_shape,
            device=device,
        )
        self.hard_constraints = self.DEFAULT_HARD_CONSTRAINTS.copy()
        self.soft_constraints = self.DEFAULT_SOFT_CONSTRAINTS.copy()

        if initial_volume is not None:
            self._initial_obj = initial_volume
        else:
            self.initial_obj = (
                torch.zeros(self._shape, device=self._device, dtype=torch.float32)
                + self.offset_obj
            )

    @property
    def obj(self):
        return self.apply_hard_constraints(self._obj)

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

    @property
    def initial_obj(self):
        return self._initial_obj

    @initial_obj.setter
    def initial_obj(self, initial_obj: torch.Tensor):
        if not isinstance(initial_obj, torch.Tensor):
            raise ValueError("initial_obj must be a torch.Tensor")

        self._initial_obj = initial_obj

    def forward(self):
        return self.obj

    def reset(self):
        self._obj = (
            torch.zeros(self._shape, device=self._device, dtype=torch.float32) + self.offset_obj
        )

    def to(self, device: str):
        self._device = device
        self._obj = self._obj.to(self._device)

    @property
    def name(self) -> str:
        return "ObjectVoxelwise"

    @property
    def params(self) -> torch.Tensor:
        return self._obj

    @property
    def soft_loss(self) -> torch.Tensor:
        return self.apply_soft_constraints(self._obj)


ObjectModelType = ObjectVoxelwise  # | ObjectDIP | ObjectImplicit (ObjectFFN?)
