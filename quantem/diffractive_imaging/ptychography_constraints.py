from typing import TYPE_CHECKING, Any, Union, overload

import numpy as np
import torch

from quantem.core.utils import array_funcs as arr
from quantem.diffractive_imaging.ptycho_utils import fourier_shift, get_com_2d
from quantem.diffractive_imaging.ptychography_base import PtychographyBase

if TYPE_CHECKING:
    import cupy as cp
    import torch

ArrayLike = Union[np.ndarray, "cp.ndarray", "torch.Tensor"]


# TODO make dataclass
class PtychographyConstraints(PtychographyBase):
    DEFAULT_CONSTRAINTS = {
        "object": {
            "fix_potential_baseline": False,
            "identical_slices": True,
            "tv_weight_yx": 0.0,
            "tv_weight_z": 0.0,
            "apply_fov_mask": False,
        },
        "probe": {
            "fix_probe": False,
            "fix_probe_com": True,
        },
    }

    _constraints = DEFAULT_CONSTRAINTS.copy()

    @property
    def constraints(self) -> dict[str, Any]:
        return self._constraints

    @constraints.setter
    def constraints(self, c: dict[str, Any]):
        for key, value in c.items():
            if key not in self.DEFAULT_CONSTRAINTS:
                raise KeyError(
                    f"Invalid constraint key '{key}', allowed keys are {list(self.DEFAULT_CONSTRAINTS.keys())}"
                )

            if not isinstance(value, dict):
                raise ValueError(f"Constraint '{key}' must be a dictionary.")

            allowed_subkeys = self.DEFAULT_CONSTRAINTS[key].keys()
            for subkey, subvalue in value.items():
                if subkey not in allowed_subkeys:
                    raise KeyError(
                        f"Invalid subkey '{subkey}' for constraint '{key}', allowed subkeys are {list(allowed_subkeys)}"
                    )

                self._constraints[key][subkey] = subvalue

    @overload
    def apply_constraints(
        self,
        object: "torch.Tensor",
        probe: "torch.Tensor",
        object_fov_mask: ArrayLike | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def apply_constraints(
        self, object: "torch.Tensor", probe: None, object_fov_mask: ArrayLike | None
    ) -> tuple[torch.Tensor, None]: ...
    @overload
    def apply_constraints(
        self, object: np.ndarray, probe: np.ndarray, object_fov_mask: ArrayLike | None
    ) -> tuple[np.ndarray, np.ndarray]: ...
    @overload
    def apply_constraints(
        self, object: np.ndarray, probe: None, object_fov_mask: ArrayLike | None
    ) -> tuple[np.ndarray, None]: ...
    def apply_constraints(
        self,
        object: ArrayLike,
        probe: ArrayLike | None,
        object_fov_mask: ArrayLike | None = None,
    ) -> tuple[ArrayLike, ArrayLike | None]:
        object = self._apply_object_constraints(
            object,
            mask=object_fov_mask,
        )
        if probe is not None and not self.constraints["probe"]["fix_probe"]:
            probe = self._apply_probe_constraints(probe)
        return object, probe

    # overloading could be applied to these as well, not sure if necessary
    def _apply_object_constraints(
        self, object, mask: ArrayLike | None = None
    ) -> ArrayLike:
        if self.object_type in ["complex", "pure_phase"]:
            if self.object_type == "complex":
                amp = arr.clip(arr.abs(object), 0.0, 1.0)
            else:
                amp = 1.0
            phase = arr.angle(object)
            if mask is not None and self.constraints["object"]["apply_fov_mask"]:
                obj2 = (
                    amp * mask * arr.exp(1.0j * phase * mask)
                )  # .type(obj_dtype_torch)
            else:
                obj2 = amp * arr.exp(1.0j * phase)  # .type(obj_dtype_torch)
        else:  # is potential, apply positivity
            obj2 = arr.clip(object, a_min=0.0)
            if self.constraints["object"]["fix_potential_baseline"]:
                ### pushing towards 0 can make reconstruction worse?
                if mask is not None:
                    offset = obj2[mask < 0.5].mean()
                    obj2 -= offset.item()

        if self.num_slices > 1:
            if self.constraints["object"]["identical_slices"]:
                object_mean = arr.mean(obj2, axis=0, keepdim=True)
                obj2[:] = object_mean  # type:ignore # TODO fix this, see if breaks graph to just do in place

        return obj2

    def _apply_probe_constraints(self, probe: ArrayLike) -> ArrayLike:
        if self.num_probes > 1:
            probe = self._probe_orthogonalization_constraint(probe)

        if self.constraints["probe"]["fix_probe_com"]:
            probe = self._probe_center_of_mass_constraint(probe)

        return probe

    def _probe_center_of_mass_constraint(self, start_probe: ArrayLike) -> ArrayLike:
        """
        Ptychographic center of mass constraint.
        Used for centering corner-centered probe intensity.
        """
        probe_intensity = arr.abs(start_probe) ** 2
        com = get_com_2d(probe_intensity, corner_centered=True)
        shifted_probe = fourier_shift(start_probe, -1 * com, match_dim=True)  # type:ignore # not sure why this breaks
        return shifted_probe

    def _probe_orthogonalization_constraint(self, start_probe: ArrayLike) -> ArrayLike:
        """
        Ptychographic probe-orthogonalization constraint.
        Used to ensure mixed states are orthogonal to each other.
        Adapted from https://github.com/AdvancedPhotonSource/tike/blob/main/src/tike/ptycho/probe.py#L690
        """

        n_probes = start_probe.shape[0]
        orthogonal_probes = []

        original_norms = arr.norm(
            arr.reshape(start_probe, (n_probes, -1)), axis=1, keepdim=True
        )

        # Gram-Schmidt orthogonalization
        for i in range(n_probes):
            probe_i = start_probe[i]
            for j in range(len(orthogonal_probes)):
                projection = (
                    arr.sum(orthogonal_probes[j].conj() * probe_i)
                    * orthogonal_probes[j]
                )
                probe_i = probe_i - projection
            orthogonal_probes.append(probe_i / arr.norm(probe_i))

        orthogonal_probes = arr.stack(orthogonal_probes)
        orthogonal_probes = orthogonal_probes * arr.reshape(original_norms, (-1, 1, 1))

        # Sort probes by real-space intensity
        intensities = arr.sum(arr.abs(orthogonal_probes) ** 2, axis=(-2, -1))
        intensities_order = arr.flip(arr.argsort(intensities), axis=0)

        return orthogonal_probes[intensities_order]
