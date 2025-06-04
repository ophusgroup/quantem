from abc import abstractmethod
from typing import Any, Literal, Self, Sequence
from warnings import warn

import torch

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import (
    validate_arr_gt,
    validate_array,
    validate_gt,
    validate_np_len,
)
from quantem.diffractive_imaging.ptycho_utils import sum_patches


class ObjectBase(AutoSerialize):
    """
    Base class for all ObjectModels to inherit from.
    """

    def __init__(
        self,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | None = None,
        device: str = "cpu",
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        *args,
    ):
        self._shape = (-1, -1, -1)
        self.num_slices = num_slices
        self.slice_thicknesses = slice_thicknesses
        self._device = device
        self._obj_type = obj_type
        self._constraints = {}

    @property
    def shape(self) -> tuple:
        return self._shape

    @shape.setter
    def shape(self, s: tuple) -> None:
        """set in Ptychography as the shape is determined by com_rotation, sampling, etc."""
        s = tuple(s)
        if len(s) != 3:
            raise ValueError(
                f"Shape must be a tuple of length 3 (depth, row, col), got {len(s)}: {s}"
            )
        self._shape = s

    @property
    def dtype(self) -> "torch.dtype":
        if self.obj_type == "potential":
            return getattr(torch, config.get("dtype_real"))
        else:
            return getattr(torch, config.get("dtype_complex"))

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str | torch.device):
        dev, _id = config.validate_device(device)
        self._device = dev

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

    @num_slices.setter
    def num_slices(self, n: int) -> None:
        validate_gt(n, 0, "num_slices")
        self._shape = (n, *self._shape[1:])

    @property
    def slice_thicknesses(self) -> torch.Tensor:
        return self._slice_thicknesses

    @slice_thicknesses.setter
    def slice_thicknesses(self, val: float | Sequence | None) -> None:
        if val is None:
            if self.num_slices > 1:
                raise ValueError(
                    f"num slices = {self.num_slices}, so slice_thicknesses cannot be None"
                )
            else:
                self._slice_thicknesses = torch.tensor([-1])
        elif isinstance(val, (float, int)):
            val = validate_gt(float(val), 0, "slice_thicknesses")
            self._slice_thicknesses = val * torch.ones(self.num_slices - 1)
        else:
            if self.num_slices == 1:
                warn("Single slice reconstruction so not setting slice_thicknesses")
            arr = validate_array(
                val,
                name="slice_thicknesses",
                dtype=config.get("dtype_real"),
                ndim=1,
                shape=(self.num_slices - 1,),
            )
            arr = validate_arr_gt(arr, 0, "slice_thicknesses")
            arr = validate_np_len(arr, self.num_slices - 1, name="slice_thicknesses")
            self._slice_thicknesses = torch.tensor(
                arr, dtype=config.get("dtype_real"), device=self.device
            )

    @property
    def obj(self):
        """get the full object"""
        raise NotImplementedError()

    @property
    def params(self):
        """optimization parameters"""
        raise NotImplementedError()

    @abstractmethod
    def forward(self, patch_indices: torch.Tensor):
        """Get patch indices of the object"""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset the object model to its initial or pre-trained state"""
        raise NotImplementedError()

    @abstractmethod
    def to(self, device: str | torch.device):
        """Move all relevant tensors to a different device."""
        self.device = device
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the object model."""
        raise NotImplementedError()

    def _propagate_array(
        self, array: "torch.Tensor", propagator_array: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Propagates array by Fourier convolving array with propagator_array.

        Parameters
        ----------
        array: np.ndarray
            Wavefunction array to be convolved
        propagator_array: np.ndarray
            Propagator array to convolve array with

        Returns
        -------
        propagated_array: np.ndarray
            Fourier-convolved array
        """
        propagated = torch.fft.ifft2(torch.fft.fft2(array) * propagator_array)
        return propagated

    def backward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {Self}, use autograd=True"
        )


class ObjectConstraints(ObjectBase):
    DEFAULT_CONSTRAINTS = {
        "fix_potential_baseline": False,
        "identical_slices": False,
        "apply_fov_mask": False,
    }

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
            phase = obj.angle() - obj.angle().mean()
            if mask is not None and self.constraints["apply_fov_mask"]:
                obj2 = amp * mask * torch.exp(1.0j * phase * mask)
            else:
                obj2 = amp * torch.exp(1.0j * phase)
        else:  # is potential, apply positivity
            obj2 = torch.clamp(obj, min=0.0)
            if self.constraints["fix_potential_baseline"]:
                ### pushing towards 0 can make reconstruction worse?
                if mask is not None:
                    offset = obj2[mask < 0.5 * mask.max()].mean()
                    obj2 -= offset.item()

        if self.num_slices > 1:
            if self.constraints["identical_slices"]:
                obj2[:] = torch.mean(obj2, dim=0, keepdim=True)

        return obj2


class ObjectPixelized(ObjectConstraints, ObjectBase):
    """
    Object model for pixelized objects.
    """

    def __init__(
        self,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | None = None,
        device: str = "cpu",
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
    ):
        super().__init__(
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            device=device,
            obj_type=obj_type,
        )
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()

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
        self._obj = torch.ones(self.shape, dtype=self.dtype, device=self.device)

    def _get_obj_patches(self, obj_array, patch_indices):
        """Extracts complex-valued roi-shaped patches from `obj_array` using patch_indices."""
        if self.obj_type == "potential":
            obj_array = torch.exp(1.0j * obj_array)
        obj_flat = obj_array.reshape(obj_array.shape[0], -1)
        # patch_indices shape: (batch_size, roi_shape[0], roi_shape[1])
        # Output shape: (num_slicpaes, batch_size, roi_shape[0], roi_shape[1])
        patches = obj_flat[:, patch_indices]
        return patches

    def to(self, device: str | torch.device):
        self.device = device
        self._obj = self._obj.to(self.device)

    @property
    def name(self) -> str:
        return "ObjPixelized"

    def backward(
        self,
        gradient: torch.Tensor,
        obj_patches: torch.Tensor,
        shifted_probes: torch.Tensor,
        propagators: torch.Tensor,
        patch_indices: torch.Tensor,
    ):
        obj_shape = self._obj.shape[-2:]
        obj_gradient = torch.zeros_like(self._obj)
        for s in reversed(range(self.num_slices)):
            probe_slice = shifted_probes[s]
            obj_slice = obj_patches[s]
            probe_normalization = torch.zeros_like(self._obj[s])
            obj_update = torch.zeros_like(self._obj[s])
            for a0 in range(shifted_probes.shape[1]):
                probe = probe_slice[a0]
                grad = gradient[a0]
                probe_normalization += sum_patches(
                    torch.abs(probe) ** 2, patch_indices, obj_shape
                ).max()

                if self.obj_type == "potential":
                    obj_update += sum_patches(
                        torch.real(-1j * torch.conj(obj_slice) * torch.conj(probe) * grad),
                        patch_indices,
                        obj_shape,
                    )
                else:
                    obj_update += sum_patches(torch.conj(probe) * grad, patch_indices, obj_shape)

            obj_gradient[s] = obj_update / probe_normalization

            # back-transmit and back-propagate
            gradient *= torch.conj(obj_slice)
            if s > 0:
                gradient = self._propagate_array(gradient, torch.conj(propagators[s - 1]))

        self._obj.grad = -1 * obj_gradient.clone().detach()
        return gradient


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
