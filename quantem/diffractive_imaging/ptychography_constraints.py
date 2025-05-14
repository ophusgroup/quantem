from typing import TYPE_CHECKING, Any, Union, overload

import numpy as np
import torch

from quantem.core.utils import array_funcs as arr
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
        "probe": {},
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
        if probe is not None:
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

        return probe

    def _probe_orthogonalization_constraint(self, start_probe: ArrayLike) -> ArrayLike:
        """
        Ptychographic probe-orthogonalization constraint.
        Used to ensure mixed states are orthogonal to each other.
        Adapted from https://github.com/AdvancedPhotonSource/tike/blob/main/src/tike/ptycho/probe.py#L690

        Parameters
        --------
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        constrained_probe: np.ndarray
            Orthogonalized probe estimate
        """
        # n_probes = self.num_probes

        # # compute upper half of P* @ P
        # pairwise_dot_product = torch.empty((n_probes, n_probes), dtype=start_probe.dtype, device=start_probe.device)

        # for i in range(n_probes):
        #     for j in range(i, n_probes):
        #         pairwise_dot_product[i, j] = torch.sum(
        #             start_probe[i].conj() * start_probe[j]
        #         )

        # # compute eigenvectors (effectively cheaper way of computing V* from SVD)
        # _, evecs = torch.linalg.eigh(pairwise_dot_product, UPLO="U")
        # start_probe = torch.tensordot(evecs.T, start_probe, dims=1)

        # # sort by real-space intensity
        # intensities = torch.sum(torch.abs(start_probe) ** 2, dim=(-2, -1))
        # intensities_order = torch.flip(torch.argsort(intensities), dims=(0,))
        # return start_probe[intensities_order]

        #### old ptycho way
        n_probes = start_probe.shape[0]
        orthogonal_probes = []

        # Compute the original probe magnitudes
        original_norms = arr.norm(
            arr.reshape(start_probe, (n_probes, -1)), axis=1, keepdim=True
        )

        # Apply Gram-Schmidt process
        for i in range(n_probes):
            probe_i = start_probe[i]

            # Subtract projections onto previously computed orthogonal probes
            for j in range(len(orthogonal_probes)):
                projection = (
                    arr.sum(orthogonal_probes[j].conj() * probe_i)
                    * orthogonal_probes[j]
                )
                probe_i = probe_i - projection

            # Normalize and store the probe
            orthogonal_probes.append(probe_i / arr.norm(probe_i))

        orthogonal_probes = arr.stack(orthogonal_probes)

        # Restore original intensities by scaling with original norms
        orthogonal_probes = orthogonal_probes * arr.reshape(original_norms, (-1, 1, 1))

        # Sort probes by real-space intensity
        intensities = arr.sum(arr.abs(orthogonal_probes) ** 2, axis=(-2, -1))
        intensities_order = arr.flip(arr.argsort(intensities), axis=0)

        return orthogonal_probes[intensities_order]

        ### py4dstem raw
        # print('start probe shape: ', start_probe.shape)

        # xp = get_array_module(start_probe)
        # n_probes = self._num_probes

        # # compute upper half of P* @ P
        # pairwise_dot_product = xp.empty((n_probes, n_probes), dtype=start_probe.dtype)

        # for i in range(n_probes):
        #     for j in range(i, n_probes):
        #         pairwise_dot_product[i, j] = xp.sum(
        #             start_probe[i].conj() * start_probe[j]
        #         )

        # # compute eigenvectors (effectively cheaper way of computing V* from SVD)
        # _, evecs = xp.linalg.eigh(pairwise_dot_product, UPLO="U")
        # start_probe = xp.tensordot(evecs.T, start_probe, axes=1)

        # # sort by real-space intensity
        # intensities = xp.sum(xp.abs(start_probe) ** 2, axis=(-2, -1))
        # intensities_order = xp.argsort(intensities, axis=None)[::-1]
        # return start_probe[intensities_order]
