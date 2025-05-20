from typing import Literal, Self

import numpy as np
from tqdm import tqdm

from quantem.core.utils import array_funcs as arr
from quantem.core.utils.utils import generate_batches
from quantem.diffractive_imaging.ptycho_utils import sum_patches
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_constraints import PtychographyConstraints
from quantem.diffractive_imaging.ptychography_visualizations import PtychographyVisualizations


class PtychographyGD(PtychographyConstraints, PtychographyVisualizations, PtychographyBase):
    def reconstruct(
        self,
        num_iter: int = 0,
        reset: bool = False,
        constraints: dict = {},
        step_size: float = 0.5,
        obj_type: Literal["complex", "pure_phase", "potential"] | None = None,
        batch_size: int | None = None,
        store_iterations: bool | None = None,
        store_iterations_every: int | None = None,
        device: Literal["gpu", "cpu"] | None = None,
    ) -> Self:
        self._check_preprocessed()
        self.device = device
        batch_size = self.gpts[0] * self.gpts[1] if batch_size is None else batch_size
        self.store_iterations = store_iterations
        self.store_iterations_every = store_iterations_every
        self.set_obj_type(obj_type, force=reset)
        if reset:
            self.reset_recon()
            self.reset_constraints()

        self.constraints = constraints  # doesn't overwrite unspecified constraints
        self._move_recon_arrays_to_device()

        shuffled_indices = np.arange(self.gpts[0] * self.gpts[1])
        for a0 in tqdm(range(num_iter), disable=not self.verbose):
            error = np.float32(0)
            np.random.shuffle(shuffled_indices)
            # TODO add get patches here if updating probe positions
            for start, end in generate_batches(
                num_items=self.gpts[0] * self.gpts[1], max_batch=batch_size
            ):
                batch_indices = shuffled_indices[start:end]
                obj_patches, propagated_probes, overlap = self.forward_operator(
                    self._obj,
                    self._probe,
                    self._patch_indices[batch_indices],
                    self._positions_px_fractional[batch_indices],
                )

                error += self.error_estimate(
                    overlap,
                    self._shifted_amplitudes[batch_indices],
                )

                self.adjoint_operator(
                    obj_patches,
                    propagated_probes,
                    overlap,
                    self._patch_indices[batch_indices],
                    self._shifted_amplitudes[batch_indices],
                    step_size,
                    fix_probe=self.constraints["probe"]["fix_probe"],
                )

            self._obj, self._probe = self.apply_constraints(
                self._obj, self._probe, obj_fov_mask=self._obj_fov_mask
            )
            error /= self._mean_diffraction_intensity * np.prod(self.gpts)
            self._epoch_losses.append(error.item())
            self._record_lrs(step_size)
            self._epoch_recon_types.append("GD")
            if self.store_iterations and ((a0 + 1) % self.store_iterations_every == 0 or a0 == 0):
                self.append_recon_iteration(self._obj, self._probe)

        return self

    def _record_lrs(self, step_size) -> None:
        if "GD" in self._epoch_lrs.keys():
            self._epoch_lrs["GD"].append(step_size)
        else:
            prev_lrs = [0.0] * (self.num_epochs - 1)
            prev_lrs.append(step_size)
            self._epoch_lrs["GD"] = prev_lrs
        for key in self._epoch_lrs.keys():  # update rest of lrs
            if key == "GD":
                continue
            self._epoch_lrs[key].append(0.0)

    def _move_recon_arrays_to_device(self):
        self._obj = self._to_xp(self._obj)
        self._probe = self._to_xp(self._probe)
        self._patch_indices = self._to_xp(self._patch_indices)
        self._shifted_amplitudes = self._to_xp(self._shifted_amplitudes)
        self.positions_px = self._to_xp(self._positions_px)
        self._fov_mask = self._to_xp(self._obj_fov_mask)
        self._propagators = self._to_xp(self._propagators)

    # endregion

    # region --- adjoint ---

    def adjoint_operator(
        self,
        obj_patches,
        propagated_probes,
        overlap,
        patch_indices,
        amplitudes,
        step_size,
        fix_probe: bool = False,
    ):
        """Single-pass adjoint operator."""
        modified_overlap = self.fourier_projection(amplitudes, overlap)
        ## mod_overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        gradient = self.gradient_step(overlap, modified_overlap)
        ## grad shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        self._obj, self._probe = self.update_obj_and_probe(
            self._obj,
            self._probe,
            obj_patches,
            patch_indices,
            propagated_probes,
            gradient,
            step_size,
            fix_probe=fix_probe,
        )
        return

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

    def update_obj_and_probe(
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
            probe_normalization = arr.match_device(np.zeros_like(obj_array[s]), probe_slice)
            obj_update = arr.match_device(np.zeros_like(obj_array[s]), obj_slice)

            for a0 in range(self.num_probes):
                probe = probe_slice[a0]
                obj = obj_slice
                grad = gradient[a0]
                probe_normalization += sum_patches(
                    np.abs(probe) ** 2,
                    patch_indices,
                    obj_shape,
                ).max()

                if self.obj_type == "potential":
                    obj_update += step_size * sum_patches(
                        np.real(-1j * np.conj(obj) * np.conj(probe) * grad),
                        patch_indices,
                        obj_shape,
                    )
                else:
                    obj_update += step_size * sum_patches(
                        np.conj(probe) * grad,
                        patch_indices,
                        obj_shape,
                    )

            obj_array[s] += obj_update / probe_normalization

            # back-transmit
            gradient *= np.conj(obj_slice)

            if s > 0:
                # back-propagate
                gradient = self._propagate_array(gradient, np.conj(self._propagators[s - 1]))
            elif not fix_probe:
                obj_normalization = np.sum(np.abs(obj_slice) ** 2, axis=(0)).max()
                probe_array = probe_array + (
                    step_size * np.sum(gradient, axis=1) / obj_normalization
                )
        return obj_array, probe_array

    # endregion
