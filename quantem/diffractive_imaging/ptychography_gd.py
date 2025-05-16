from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm import tqdm

from quantem.core import config
from quantem.core.utils import array_funcs as arr
from quantem.diffractive_imaging.ptycho_utils import (
    generate_batches,
    sum_patches,
)
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_constraints import PtychographyConstraints

if TYPE_CHECKING:
    import cupy as cp
else:
    if config.get("has_cupy"):
        import cupy as cp


class PtychographyGD(PtychographyConstraints, PtychographyBase):
    def reconstruct(
        self,
        num_iter: int = 0,
        reset: bool = True,
        batch_size: int | None = None,
        step_size: float = 0.5,
        constraints: dict = {},
        device: Literal["gpu", "cpu"] | None = None,
    ) -> None:
        # self.device = device
        self._check_preprocessed()
        if device is not None:
            self.device = device
        if batch_size is None:
            batch_size = self.gpts[0] * self.gpts[1]
        if reset:
            self.reset_recon()
            self.reset_constraints()
            obj = np.ones_like(self.obj)
            probe = self.initial_probe.copy()
        else:
            obj = self.obj
            probe = self.probe
        self.constraints = constraints  # doesn't overwrite unspecified constraints

        obj = self._to_xp(obj)
        probe = self._to_xp(probe)
        pos_frac = self._to_xp(self.positions_px_fractional)
        patch_indices = self._to_xp(self.patch_indices)
        amplitudes = self._to_xp(self.shifted_amplitudes)
        fov_mask = self._to_xp(self.object_fov_mask).astype(self._object_dtype)
        self._propagators = self._to_xp(self._propagators)

        shuffled_indices = np.arange(self.gpts[0] * self.gpts[1])
        for a0 in tqdm(range(num_iter)):
            error = np.float32(0)
            np.random.shuffle(shuffled_indices)
            for start, end in generate_batches(
                num_items=self.gpts[0] * self.gpts[1], max_batch=batch_size
            ):
                batch_indices = shuffled_indices[start:end]
                obj_patches, propagated_probes, overlap = self.forward_operator(
                    obj,
                    probe,
                    patch_indices[batch_indices],
                    pos_frac[batch_indices],
                )

                obj, probe = self.adjoint_operator(
                    obj,
                    probe,
                    obj_patches,
                    patch_indices[batch_indices],
                    propagated_probes,
                    amplitudes[batch_indices],
                    overlap,
                    step_size,
                    fix_probe=self.constraints["probe"]["fix_probe"],
                )

                error += self.error_estimate(
                    obj,
                    probe,
                    patch_indices[batch_indices],
                    pos_frac[batch_indices],
                    amplitudes[batch_indices],
                )

            obj, probe = self.apply_constraints(obj, probe, object_fov_mask=fov_mask)
            error /= self._mean_diffraction_intensity * np.prod(self.gpts)
            self._epoch_losses.append(error.item())
            self._record_lrs(step_size)
            self._epoch_recon_types.append("GD")
            if self.store_iterations and (
                (a0 + 1) % self.store_iterations_every == 0 or a0 == 0
            ):
                self.append_recon_iteration(obj, probe)

        if self.device == "gpu":
            if isinstance(obj, cp.ndarray):
                obj = self._as_numpy(obj)
            if isinstance(probe, cp.ndarray):
                probe = self._as_numpy(probe)

        self.obj = obj
        self.probe = probe
        return

    def _record_lrs(self, step_size) -> None:
        if "GD" in self._epoch_lrs.keys():
            self._epoch_lrs["GD"].append(step_size)
        else:
            prev_lrs = [0.0] * self.num_epochs
            prev_lrs.append(step_size)
            self._epoch_lrs["GD"] = prev_lrs
        for key in self._epoch_lrs.keys():  # update rest of lrs
            if key == "GD":
                continue
            self._epoch_lrs[key].append(0.0)

    # def _move_recon_arrays_to_device(self):
    #     self.obj = self._to_xp(obj)
    #     self.probe = self._to_xp(probe)
    #     self.pos_frac = self._to_xp(self.positions_px_fractional)
    #     self.patch_row = self._to_xp(self.patch_row)
    #     self.patch_col = self._to_xp(self.patch_col)
    #     self.amplitudes = self._to_xp(self.shifted_amplitudes)
    #     self.fov_mask = self._to_xp(self.object_fov_mask).astype(self._object_dtype)
    #     self._propagators = self._to_xp(self._propagators)

    # endregion

    # region --- adjoint ---

    def adjoint_operator(
        self,
        obj_array,
        probe_array,
        obj_patches,
        patch_indices,
        shifted_probes,
        amplitudes,
        overlap,
        step_size,
        fix_probe: bool = False,
    ):
        """Single-pass adjoint operator."""
        modified_overlap = self.fourier_projection(amplitudes, overlap)
        ## mod_overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        gradient = self.gradient_step(overlap, modified_overlap)
        ## grad shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        obj_array, probe_array = self.update_object_and_probe(
            obj_array,
            probe_array,
            obj_patches,
            patch_indices,
            shifted_probes,
            gradient,
            step_size,
            fix_probe=fix_probe,
        )
        return obj_array, probe_array

    def fourier_projection(self, measured_amplitudes, overlap_array):
        """Replaces the Fourier amplitude of overlap with the measured data."""
        fourier_overlap = arr.fft2(overlap_array)  # TODO test single case this
        if self.num_probes == 1:  # faster
            fourier_modified_overlap = measured_amplitudes * arr.exp(
                1.0j * arr.angle(fourier_overlap)
            )
        else:  # necessary for mixed state
            farfield_amplitudes = self.estimate_amplitudes(overlap_array)
            farfield_amplitudes[farfield_amplitudes == 0] = np.inf
            amplitude_modification = measured_amplitudes / farfield_amplitudes
            fourier_modified_overlap = amplitude_modification * fourier_overlap

        return arr.ifft2(fourier_modified_overlap)

    def gradient_step(self, overlap_array, modified_overlap_array):
        """Computes analytical gradient."""
        return modified_overlap_array - overlap_array

    def update_object_and_probe(
        self,
        obj_array,
        probe_array,
        obj_patches,
        patch_indices,
        shifted_probes,
        gradient,
        step_size,
        fix_probe: bool = False,
    ):
        """
        Updates object and probe arrays.
        """
        obj_shape = obj_array.shape[-2:]

        for s in reversed(range(self.num_slices)):
            probe_slice = shifted_probes[s]
            obj_slice = obj_patches[s]
            probe_normalization = arr.match_device(
                np.zeros_like(obj_array[s]), probe_slice
            )
            object_update = arr.match_device(np.zeros_like(obj_array[s]), obj_slice)

            for a0 in range(self.num_probes):
                probe = probe_slice[a0]
                obj = obj_slice
                grad = gradient[a0]
                probe_normalization += sum_patches(
                    np.abs(probe) ** 2,
                    patch_indices,
                    obj_shape,
                ).max()

                if self.object_type == "potential":
                    object_update += step_size * sum_patches(
                        np.real(-1j * np.conj(obj) * np.conj(probe) * grad),
                        patch_indices,
                        obj_shape,
                    )
                else:
                    object_update += step_size * sum_patches(
                        np.conj(probe) * grad,
                        patch_indices,
                        obj_shape,
                    )

            obj_array[s] += object_update / probe_normalization

            # back-transmit
            gradient *= np.conj(obj_slice)

            if s > 0:
                # back-propagate
                gradient = self._propagate_array(
                    gradient, np.conj(self._propagators[s - 1])
                )
            elif not fix_probe:
                obj_normalization = np.sum(np.abs(obj_slice) ** 2, axis=(0)).max()
                probe_array = probe_array + (
                    step_size * np.sum(gradient, axis=1) / obj_normalization
                )
        return obj_array, probe_array

    # endregion
