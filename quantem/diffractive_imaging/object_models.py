from abc import abstractmethod
from typing import Any, Literal

import torch

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize


class ObjectBase(AutoSerialize):
    """
    Base class for all ObjectModels to inherit from.
    """

    def __init__(
        self,
        shape: tuple,
        device: str,
        obj_type: Literal["complex", "pure_phase", "potential"],
        *args,
    ):
        self._shape = shape
        self._device = device
        self._obj_type = obj_type
        self._constraints = {}

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> "torch.dtype":
        if self.obj_type == "potential":
            return getattr(torch, config.get("dtype_real"))
        else:
            return getattr(torch, config.get("dtype_complex"))

    @property
    def device(self) -> str:
        return self._device

    @property
    def obj_type(self) -> str:
        return self._obj_type

    @obj_type.setter
    def obj_type(self, t: str | None) -> None:
        self._obj_type = self._process_obj_type(t)

    def _process_obj_type(self, obj_type: str | None) -> str:
        if obj_type is None:
            return self._obj_type
        t_str = str(obj_type).lower()
        if t_str in ["potential", "pot", "potentials"]:
            return "potential"
        elif t_str in ["pure_phase", "purephase", "pure phase", "pure"]:
            return "pure_phase"
        elif t_str in ["complex", "c"]:
            return "complex"
        else:
            raise ValueError(
                f"Object type should be 'potential', 'complex', or 'pure_phase', got {obj_type}"
            )

    @property
    def num_slices(self) -> int:
        return self.shape[0]

    @property
    def obj(self):
        """get the full object"""
        raise NotImplementedError()

    @property
    def params(self):
        """optimization parameters"""
        raise NotImplementedError()

    @property
    def model_input(self):
        """get the model input"""
        raise NotImplementedError()

    @abstractmethod
    def forward(self, patch_indices: torch.Tensor):
        """Get patch indices of the object"""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """
        Reset the object model to its initial or pre-trained state
        """
        raise NotImplementedError()

    def reinitialize(self, shape: tuple, *args):
        # TBD -- allow preprocess to be run again to change num_slices or similar
        raise NotImplementedError()

    @abstractmethod
    def to_device(self, device: str | torch.device):
        """Move all relevant tensors to a different device."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the object model."""
        raise NotImplementedError()


class ObjectConstraints(ObjectBase):
    DEFAULT_CONSTRAINTS = {
        "fix_potential_baseline": False,
        "identical_slices": False,
        "apply_fov_mask": False,
    }
    # Defaults are kept in PtychographyConstraints for now, not sure where they'll end up
    # eventually, as it contains hard + soft constraints, but if adding new constraints here
    # also need to be added along with default values there

    @property
    def constraints(self) -> dict[str, Any]:
        return self._constraints

    @constraints.setter
    def constraints(self, c: dict[str, Any]):
        gkeys = self.DEFAULT_CONSTRAINTS.keys()
        for key, value in c.items():
            if key not in gkeys:
                raise KeyError(f"Invalid constraint key '{key}', allowed keys are {gkeys}")
            self._constraints[key] = value

    def add_constraint(self, key: str, value: Any):
        """Add a constraint to the object model."""
        gkeys = self.DEFAULT_CONSTRAINTS.keys()
        if key not in gkeys:
            raise KeyError(f"Invalid constraint key '{key}', allowed keys are {gkeys}")
        self._constraints[key] = value

    def apply_constraints(
        self, obj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply constraints to the object model.
        """
        if self.obj_type in ["complex", "pure_phase"]:
            if self.obj_type == "complex":
                amp = torch.clamp(torch.abs(obj), 0.0, 1.0)
            else:
                amp = 1.0
            phase = obj.angle()
            if mask is not None and self.constraints["apply_fov_mask"]:
                obj2 = amp * mask * torch.exp(1.0j * phase * mask)  # .type(obj_dtype_torch)
            else:
                obj2 = amp * torch.exp(1.0j * phase)  # .type(obj_dtype_torch)
        else:  # is potential, apply positivity
            obj2 = torch.clamp(obj, min=0.0)
            if self.constraints["fix_potential_baseline"]:
                ### pushing towards 0 can make reconstruction worse?
                if mask is not None:
                    offset = obj2[mask < 0.5 * mask.max()].mean()
                    obj2 -= offset.item()

        if self.num_slices > 1:
            if self.constraints["identical_slices"]:
                obj_mean = torch.mean(obj2, dim=0, keepdim=True)
                obj2[:] = obj_mean  # type:ignore # TODO fix this, see if breaks graph to just do in place

        return obj2


class ObjectPixelized(ObjectConstraints, ObjectBase):
    """
    Object model for pixelized objects.
    """

    def __init__(
        self,
        shape: tuple,
        device: str,
        obj_type: Literal["complex", "pure_phase", "potential"],
    ):
        super().__init__(shape, device, obj_type)
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()
        self._obj = torch.ones(shape, dtype=self.dtype, device=device)

    @property
    def obj(self):
        return self.apply_constraints(self._obj)
        # return self._obj

    @property
    def params(self):
        """optimization parameters"""
        return self._obj

    @property
    def model_input(self):
        return None

    def forward(self, patch_indices: torch.Tensor):
        """Get patch indices of the object"""
        return self._get_obj_patches(self.obj, patch_indices)

    def reset(self):
        """
        Reset the object model to its initial or pre-trained state
        """
        self._obj = torch.ones(self._obj.shape, dtype=self.dtype, device=self.device)

    def _get_obj_patches(self, obj_array, patch_indices):
        """Extracts complex-valued roi-shaped patches from `obj_array` using patch_indices."""
        if self.obj_type == "potential":
            obj_array = torch.exp(1.0j * obj_array)
        obj_flat = obj_array.reshape(obj_array.shape[0], -1)
        # patch_indices shape: (batch_size, roi_shape[0], roi_shape[1])
        # Output shape: (num_slicpaes, batch_size, roi_shape[0], roi_shape[1])
        patches = obj_flat[:, patch_indices]
        return patches

    def to_device(self, device: str | torch.device):
        self._obj = self._obj.to(device)

    @property
    def name(self) -> str:
        return "ObjPixelized"


# class ObjectDIP(ObjectBase):
#     """
#     Object for
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._obj = None
#         self._obj_shape = None
#         self._num_slices = None


#     @property
#     def model_input(self):
#         """get the model input"""
#         return self._obj

#     def pretrain

# class ObjectImplicit(ObjectBase):
#     """
#     Object model for implicit objects.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._obj = None
#         self._obj_shape = None
#         self._num_slices = None

#     def pretrain(self, *args, **kwargs):


#     ### here the forward call will take the batch indices and create the appropriate
#     ### input (which maybe is just the raw patch indices? tbd) for the implicit input
#     ### so it will be parallelized inference across the batches rather than inference once
#     ### and then patching that, like it will be for DIP
