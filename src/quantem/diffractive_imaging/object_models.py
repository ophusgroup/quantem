from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Literal, Self, Sequence
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.blocks import reset_weights
from quantem.core.ml.loss_functions import get_loss_function
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.core.utils.validators import (
    validate_arr_gt,
    validate_array,
    validate_gt,
    validate_np_len,
    validate_tensor,
)
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.constraints import BaseConstraints
from quantem.diffractive_imaging.ptycho_utils import sum_patches
from quantem.diffractive_imaging.rng_mixin import RNGMixin

"""
Currently all object models.obj are complex valued for "complex" or "pure_phase" object types,
and real valued for "potential" object types. This could be changed to be always complex valued, 
(after applying constraints) as currently the real-valued potential is made complex in get_obj_patches, 
which will not be used for implicit NNs, which leads to an inconsistency. Leaving for now as I'm not 
sure if this would lead to other issues, so a bit of testing will be needed.
"""

# TODO -- class method and protect object inits
# - DIP from_model,
# - pixelated doesn't make much sense, from_array works, but doesn't make sense
# as the default would be random/uniform init, so not necessary to actually pass an array

# TODO - clean up how shape and num_slices are handled, redundant and allows for errors


class ObjectBase(RNGMixin, OptimizerMixin, AutoSerialize):
    """
    Base class for all ObjectModels to inherit from.
    """

    DEFAULT_LRS = {
        "object": 5e-3,
        "tv_weight_z": 0,
        "tv_weight_yx": 0,
    }

    def __init__(
        self,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | torch.Tensor | None = None,
        device: str = "cpu",
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        rng: np.random.Generator | int | None = None,
        shape: tuple[int, int, int] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._slice_thicknesses = None
        self._rng = None
        self._shape = shape if shape is not None else (1, 1, 1)
        self._mask = None
        self.device = device
        self.obj_type = obj_type
        self.num_slices = num_slices
        self.slice_thicknesses = slice_thicknesses
        self.rng = rng

    @property
    def shape(self) -> tuple:
        return self._shape

    @shape.setter
    def shape(self, s: tuple) -> None:
        s = tuple(int(x) for x in s)
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
    def slice_thicknesses(self) -> torch.Tensor | None:
        return self._slice_thicknesses

    @slice_thicknesses.setter
    def slice_thicknesses(self, val: float | Sequence | torch.Tensor | None) -> None:
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
            dt = getattr(torch, config.get("dtype_real"))
            self._slice_thicknesses = torch.tensor(arr, dtype=dt, device=self.device)

    @property
    def mask(self) -> torch.Tensor | None:
        return self._mask

    @mask.setter
    def mask(self, mask: torch.Tensor | np.ndarray | None):
        if mask is not None:
            mask = validate_tensor(
                mask,
                name="mask",
                dtype=self.dtype,
                ndim=3,
                expand_dims=True,
            )
            self._mask = mask.to(self.device).expand(self.num_slices, -1, -1)
        else:
            self._mask = None

    @property
    def obj(self):
        raise NotImplementedError()

    @property
    def params(self):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, patch_indices: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    def to(self, device: str | torch.device):
        self.device = device
        self._rng_to_device(device)

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    def get_optimization_parameters(self):
        """Get the parameters that should be optimized for this model."""
        try:
            params = self.params
            if params is None:
                return []
            return params
        except NotImplementedError:
            # This happens when params is not implemented yet in abstract base
            return []

    def _propagate_array(
        self, array: "torch.Tensor", propagator_array: "torch.Tensor"
    ) -> "torch.Tensor":
        propagated = torch.fft.ifft2(
            torch.fft.fft2(array, norm="ortho") * propagator_array, norm="ortho"
        )
        return propagated

    def _get_obj_patches(self, obj_array, patch_indices):
        if not obj_array.is_complex():
            obj_array = torch.exp(1.0j * obj_array)
        obj_flat = obj_array.reshape(obj_array.shape[0], -1)
        patches = obj_flat[:, patch_indices]
        return patches

    def backward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {Self}, use autograd=True"
        )


class ObjectConstraints(BaseConstraints, ObjectBase):
    DEFAULT_CONSTRAINTS = {
        "fix_potential_baseline": False,
        "fix_potential_baseline_factor": 1.0,
        "identical_slices": False,
        "apply_fov_mask": False,
        "tv_weight_z": 0,
        "tv_weight_yx": 0,
        "surface_zero_weight": 0,
    }

    def apply_hard_constraints(
        self, obj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        else:
            if self.constraints["fix_potential_baseline"]:
                print("fixing baseline")
                if mask is not None:
                    offset = obj[mask < 0.5 * mask.max()].mean()
                else:
                    offset = torch.min(obj)
                offset *= self.constraints["fix_potential_baseline_factor"]
            else:
                offset = 0

            obj2 = torch.clamp(obj - offset, min=0.0)

        if self.constraints["apply_fov_mask"] and mask is not None:
            obj2 *= mask

        if self.num_slices > 1:
            if self.constraints["identical_slices"]:
                obj2[:] = torch.mean(obj2, dim=0, keepdim=True)

        return obj2

    def apply_soft_constraints(
        self, obj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._soft_constraint_loss = {}

        tv_loss = self.get_tv_loss(
            obj,
            weights=(
                self.constraints["tv_weight_z"],
                self.constraints["tv_weight_yx"],
            ),
        )
        self._soft_constraint_loss["tv_loss"] = tv_loss

        surface_zero_loss = self.get_surface_zero_loss(
            obj,
            weight=self.constraints["surface_zero_weight"],
        )
        self._soft_constraint_loss["surface_zero_loss"] = surface_zero_loss

        return tv_loss + surface_zero_loss

    def get_tv_loss(
        self, array: torch.Tensor, weights: None | tuple[float, float] = None
    ) -> torch.Tensor:
        loss = self._get_zero_loss_tensor()
        if weights is None:
            w = (
                self.constraints["tv_weight_z"],
                self.constraints["tv_weight_yx"],
            )
        elif isinstance(weights, (float, int)):
            if weights == 0:
                return loss
            w = (weights, weights)
        else:
            if not any(weights):
                return loss
            if len(weights) != 2:
                raise ValueError(f"weights must be a tuple of length 2, got {weights}")
            w = weights

        if array.is_complex():
            ph = array.angle()
            loss += self._calc_tv_loss(ph, w)
            amp = array.abs()
            if self.obj_type == "complex":
                loss += self._calc_tv_loss(amp, w)
        else:
            loss += self._calc_tv_loss(array, w)

        return loss

    def _calc_tv_loss(self, array: torch.Tensor, weight: tuple[float, float]) -> torch.Tensor:
        loss = self._get_zero_loss_tensor()
        calc_dim = 0
        for dim in range(array.ndim):
            if dim == 0 and array.ndim == 3:
                w = weight[0]
            else:
                w = weight[1]
            if w > 0:
                calc_dim += 1
                loss += w * torch.mean(torch.abs(array.diff(dim=dim)))
        loss /= calc_dim
        return loss

    def get_surface_zero_loss(
        self, array: torch.Tensor, weight: float | int = 0.0
    ) -> torch.Tensor:
        loss = self._get_zero_loss_tensor()
        if weight == 0:
            return loss
        if array.shape[0] < 3:
            return loss
        if array.is_complex():
            amp = array.abs()
            ph = array.angle()
            loss += weight * (torch.mean(amp[0]) + torch.mean(amp[-1]))
            loss += weight * (torch.mean(torch.diff(ph[0])) + torch.mean(torch.diff(ph[-1])))
        else:
            loss += weight * (torch.mean(array[0]) + torch.mean(array[-1]))
        return loss


class ObjectPixelated(ObjectConstraints):
    """
    Object model for pixelized objects.
    """

    def __init__(
        self,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | None = None,
        device: str = "cpu",
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        rng: np.random.Generator | int | None = None,
        shape: tuple[int, int, int] | None = None,
    ):
        super().__init__(
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            device=device,
            obj_type=obj_type,
            rng=rng,
            shape=shape,
        )

    @property
    def obj(self):
        return self.apply_hard_constraints(self._obj, mask=self.mask)
        # return self._obj

    @property
    def params(self):
        """optimization parameters"""
        return self._obj

    def forward(self, patch_indices: torch.Tensor):
        """Get patch indices of the object"""
        return self._get_obj_patches(self.obj, patch_indices)

    def reset(self):
        """
        Reset the object model to its initial or pre-trained state
        """
        self._obj = torch.ones(self.shape, dtype=self.dtype, device=self.device)

    def to(self, device: str | torch.device):
        super().to(device)
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


class ObjectDIP(ObjectConstraints):
    """
    DIP/model based object model.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        model_input: torch.Tensor | None = None,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | torch.Tensor | None = None,
        input_noise_std: float = 0.025,
        device: str = "cpu",
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        rng: np.random.Generator | int | None = None,
        shape: tuple[int, int, int] | None = None,
    ):
        super().__init__(
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            device=device,
            obj_type=obj_type,
            rng=rng,
            shape=shape,
        )
        self.model = model

        if model_input is None:
            # Create default model input - will be set properly in set_initial_obj
            if shape is not None:
                input_shape = (1, *shape)
            else:
                input_shape = (1, num_slices, 1, 1)
            self.model_input = torch.randn(
                input_shape, dtype=self.dtype, device=self.device, generator=self._rng_torch
            )
        else:
            self.model_input = model_input.clone().detach()

        self.pretrain_target = self.model_input.clone().detach()
        self._pretrain_losses = []
        self._pretrain_lrs = []
        self._model_input_noise_std = input_noise_std

    @property
    def name(self) -> str:
        return "ObjectDIP"

    @property
    def dtype(self) -> "torch.dtype":
        if hasattr(self.model, "dtype"):
            return getattr(self.model, "dtype")
        else:
            if self.obj_type in ["complex"]:
                return config.get("dtype_complex")
            else:
                return config.get("dtype_real")

    @property
    def model(self) -> "torch.nn.Module":
        """get the DIP model"""
        return self._model

    @model.setter
    def model(self, dip: "torch.nn.Module"):
        """set the DIP model"""
        if not isinstance(dip, torch.nn.Module):
            raise TypeError(f"DIP must be a torch.nn.Module, got {type(dip)}")
        if hasattr(dip, "dtype"):
            dt = getattr(dip, "dtype")
            if self.obj_type in ["complex"] and not dt.is_complex:
                raise ValueError("DIP model must be a complex-valued model for complex objects")
        self._model = dip.to(self.device)
        self.set_pretrained_weights(self._model)

    @property
    def pretrained_weights(self) -> dict[str, torch.Tensor]:
        """get the pretrained weights of the DIP model"""
        return self._pretrained_weights

    def set_pretrained_weights(self, model: torch.nn.Module):
        """set the pretrained weights of the DIP model"""
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Pretrained model must be a torch.nn.Module, got {type(model)}")
        self._pretrained_weights = deepcopy(model.state_dict())

    @property
    def model_input(self) -> torch.Tensor:
        """get the model input"""
        return self._model_input

    @model_input.setter
    def model_input(self, input_tensor: torch.Tensor):
        """set the model input"""
        inp = validate_tensor(
            input_tensor,
            name="model_input",
            dtype=self.dtype,
            ndim=4,
            expand_dims=True,
        )
        self._model_input = inp.to(self.device)

    @property
    def pretrain_target(self) -> torch.Tensor:
        """get the pretrain target"""
        return self._pretrain_target

    @pretrain_target.setter
    def pretrain_target(self, target: torch.Tensor):
        """set the pretrain target"""
        if target.ndim == 4:
            target = target.squeeze(0)
        target = validate_tensor(
            target,
            name="pretrain_target",
            ndim=3,
            dtype=self.dtype,
            expand_dims=True,
        )
        if target.shape[-3:] != self.model_input.shape[-3:]:
            raise ValueError(
                f"Pretrain target shape {target.shape} does not match model input shape {self.model_input.shape}"
            )
        self._pretrain_target = target.to(self.device)

    @property
    def _model_input_noise_std(self) -> float:
        """standard deviation of the gaussian noise added to the model input each forward call"""
        return self._input_noise_std

    @_model_input_noise_std.setter
    def _model_input_noise_std(self, std: float):
        validate_gt(std, 0.0, "input_noise_std", geq=True)
        self._input_noise_std = std

    @property
    def pretrain_losses(self) -> np.ndarray:
        return np.array(self._pretrain_losses)

    @property
    def pretrain_lrs(self) -> np.ndarray:
        return np.array(self._pretrain_lrs)

    @property
    def obj(self):
        """get the full object"""
        obj = self.model(self._model_input)[0]
        if self.obj_type == "pure_phase" and "complex" not in str(self.dtype):
            # using a real-valued model for a pure-phase (complex) object
            obj = torch.ones_like(obj) * torch.exp(1j * obj)
        return self.apply_hard_constraints(obj, mask=self.mask)

    @property
    def _obj(self):
        return self.model(self._model_input)[0]

    def forward(self, patch_indices: torch.Tensor):
        """Get object patches at given indices"""
        if self._input_noise_std > 0.0:
            noise = (
                torch.randn(
                    self.model_input.shape,
                    dtype=self.dtype,
                    device=self.device,
                    generator=self._rng_torch,
                )
                * self._input_noise_std
            )
            model_input = self.model_input + noise
        else:
            model_input = self.model_input

        obj_array = self.model(model_input)[0]
        if self.mask is not None:
            obj_array = obj_array * self.mask
        return self._get_obj_patches(obj_array, patch_indices)

    def to(self, device: str | torch.device):
        """Move all relevant tensors to a different device."""
        super().to(device)
        self._model = self._model.to(self.device)
        self._model_input = self._model_input.to(self.device)
        if hasattr(self, "_pretrain_target"):
            self._pretrain_target = self._pretrain_target.to(self.device)

    @property
    def params(self):
        """optimization parameters"""
        return self.model.parameters()

    def get_optimization_parameters(self):
        """Get the parameters that should be optimized for this model."""
        # Return a fresh list of parameters each time to avoid generator exhaustion
        return list(self._model.parameters())

    def reset(self):
        """Reset the object model to its initial or pre-trained state"""
        self.model.load_state_dict(self.pretrained_weights.copy())

    def pretrain(
        self,
        model_input: torch.Tensor | None = None,
        pretrain_target: torch.Tensor | None = None,
        reset: bool = False,
        num_epochs: int = 100,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        loss_fn: Callable | str = "l2",
        apply_constraints: bool = False,
        show: bool = True,
    ):
        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)

        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_epochs)

        if reset:
            self._model.apply(reset_weights)
            self._pretrain_losses = []
            self._pretrain_lrs = []

        if model_input is not None:
            self.model_input = model_input
        if pretrain_target is not None:
            if pretrain_target.shape[-3:] != self.model_input.shape[-3:]:
                raise ValueError(
                    f"Model target shape {pretrain_target.shape} does not match model input shape {self.model_input.shape}"
                )
            self.pretrain_target = pretrain_target.clone().detach().to(self.device)
        elif self.pretrain_target is None:
            raise ValueError(
                "No pretrain target set. Provide pretrain_target or set it beforehand."
            )

        loss_fn = get_loss_function(loss_fn, self.dtype)
        self._pretrain(
            num_epochs=num_epochs,
            loss_fn=loss_fn,
            apply_constraints=apply_constraints,
            show=show,
        )
        self.set_pretrained_weights(self.model)

    def _pretrain(
        self,
        num_epochs: int,
        loss_fn: Callable,
        apply_constraints: bool = False,
        show: bool = False,
    ):
        """Pretrain the DIP model."""
        if not hasattr(self, "pretrain_target"):
            raise ValueError("Pretrain target is not set. Use pretrain_target to set it.")

        self._model.train()
        optimizer = self.optimizer
        if optimizer is None:
            raise ValueError("Optimizer not set. Call set_optimizer() first.")

        scheduler = self.scheduler
        pbar = tqdm(range(num_epochs))
        output = self.obj

        for a0 in pbar:
            if self._input_noise_std > 0.0:
                noise = (
                    torch.randn(
                        self.model_input.shape,
                        dtype=self.dtype,
                        device=self.device,
                        generator=self._rng_torch,
                    )
                    * self._input_noise_std
                )
                model_input = self.model_input + noise
            else:
                model_input = self.model_input

            if apply_constraints:
                output = self.apply_hard_constraints(self.model(model_input)[0])
            else:
                output = self.model(model_input)[0]
            loss: torch.Tensor = loss_fn(output, self.pretrain_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss.item())
                else:
                    scheduler.step()

            self._pretrain_losses.append(loss.item())
            self._pretrain_lrs.append(optimizer.param_groups[0]["lr"])
            pbar.set_description(f"Epoch {a0 + 1}/{num_epochs}, Loss: {loss.item():.3e}, ")

        if show:
            self.visualize_pretrain(output)

    def visualize_pretrain(self, pred_obj: torch.Tensor):
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax = fig.add_subplot(gs[0])
        lines = []
        lines.extend(
            ax.semilogy(
                np.arange(len(self._pretrain_losses)), self._pretrain_losses, c="k", label="loss"
            )
        )
        ax.set_ylabel("Loss", color="k")
        ax.tick_params(axis="y", which="both", colors="k")
        ax.spines["left"].set_color("k")
        ax.set_xlabel("Epochs")
        nx = ax.twinx()
        nx.spines["left"].set_visible(False)
        lines.extend(
            nx.semilogy(
                np.arange(len(self._pretrain_lrs)),
                self._pretrain_lrs,
                c="tab:orange",
                label="LR",
            )
        )
        labs = [lin.get_label() for lin in lines]
        nx.legend(lines, labs, loc="upper center")
        nx.set_ylabel("LRs")

        n_bot = 4 if self.obj_type == "complex" else 2
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, n_bot, subplot_spec=gs[1])
        axs_bot = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(n_bot)])
        target = self.pretrain_target
        if n_bot == 4:
            show_2d(
                [
                    pred_obj.mean(0).angle().cpu().detach().numpy(),
                    target.mean(0).angle().cpu().detach().numpy(),
                    pred_obj.mean(0).abs().cpu().detach().numpy(),
                    target.mean(0).abs().cpu().detach().numpy(),
                ],
                figax=(fig, axs_bot),
                title=[
                    "Predicted Phase",
                    "Target Phase",
                    "Predicted Amplitude",
                    "Target Amplitude",
                ],
                cmap="magma",
                cbar=True,
            )
        else:
            show_2d(
                [
                    pred_obj.mean(0).cpu().detach().numpy(),
                    target.mean(0).cpu().detach().numpy(),
                ],
                figax=(fig, axs_bot),
                title=[f"Pred obj ({self.obj_type})", f"Target obj ({self.obj_type})"],
                cmap="magma",
                cbar=True,
            )
        plt.suptitle(
            f"Final loss: {self._pretrain_losses[-1]:.3e} | Epochs: {len(self._pretrain_losses)}",
            fontsize=14,
            y=0.94,
        )
        plt.show()


# class ObjectImplicit(ObjectBase):
#     """
#     Object model for implicit objects. Importantly, the forward call from scan positions
#     for this model will not require subpixel shifting of the object probe, as subpixel shifting
#     will be done in the object model itself, so it is properly aligned around the probe positions
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

ObjectModelType = ObjectPixelated | ObjectDIP  # | ObjectImplicit
