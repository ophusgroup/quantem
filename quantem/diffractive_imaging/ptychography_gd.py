from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm import tqdm

from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging.ptycho_utils import (
    generate_batches,
    sum_patches,
)
from quantem.diffractive_imaging.ptychography_base import PtychographyBase

if TYPE_CHECKING:
    import cupy as cp
else:
    if config.get("has_cupy"):
        import cupy as cp


class PtychographyGD(PtychographyBase):
    def __init__(
        self,
        dset: Dataset4dstem,
        object_type: Literal["complex", "pure_phase", "potential"] = "complex",
        num_probes: int = 1,
        num_slices: int = 1,
        slice_thicknesses: float | None = None,
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            dset=dset,
            object_type=object_type,
            num_probes=num_probes,
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            verbose=verbose,
            rng=rng,
        )

    # region --- reconstruction ---

    def reconstruct(
        self,
        num_iter: int = 0,
        reset: bool = True,
        fix_probe: bool = False,
        batch_size: int | None = None,
        step_size: float = 0.5,
        constraints: dict | None = None,
        device: str = "cpu",
    ):
        if device == "gpu" and config.get("has_cupy"):
            xp = cp
        else:
            xp = np
        self.xp = xp

        if batch_size is None:
            batch_size = self.gpts[0] * self.gpts[1]
            num_batches = 1
        else:
            num_batches = 1 + ((self.gpts[0] * self.gpts[1]) // batch_size)

        if reset:
            obj = xp.ones_like(self.object)
            probe = self.initial_probe.copy()
            self._losses = []
        else:
            obj = self.object
            probe = self.probe

        obj = xp.asarray(obj)
        probe = xp.asarray(probe)
        pos_frac = xp.asarray(self.positions_px_fractional)
        patch_row = xp.asarray(self.patch_row)
        patch_col = xp.asarray(self.patch_col)
        amplitudes = xp.asarray(self.shifted_amplitudes)

        shuffled_indices = np.arange(self.gpts[0] * self.gpts[1])
        for a0 in tqdm(range(num_iter)):
            error = np.float32(0)
            for start, end in generate_batches(
                num_items=self.gpts[0] * self.gpts[1], max_batch=batch_size
            ):
                batch_indices = shuffled_indices[start:end]
                obj_patches, propagated_probes, overlap = self.forward_operator(
                    obj,
                    probe,
                    patch_row[batch_indices],
                    patch_col[batch_indices],
                    pos_frac[batch_indices],
                )

                print(
                    "overlap max min std: ", overlap.max(), overlap.min(), overlap.std()
                )

                obj, probe = self.adjoint_operator(
                    obj,
                    probe,
                    obj_patches,
                    patch_row[batch_indices],
                    patch_col[batch_indices],
                    propagated_probes,
                    amplitudes[batch_indices],
                    overlap,
                    step_size,
                    fix_probe=fix_probe,
                )
                print("obj2 max min std: ", obj.max(), obj.min(), obj.std())

                error += self.error_estimate(
                    obj,
                    probe,
                    patch_row[batch_indices],
                    patch_col[batch_indices],
                    pos_frac[batch_indices],
                    amplitudes[batch_indices],
                )

            self._losses.append(error.item() / num_batches)

        if self.device == "gpu":
            if isinstance(obj, cp.ndarray):
                obj = self._to_numpy(obj)
            if isinstance(probe, cp.ndarray):
                probe = self._to_numpy(probe)

        self.object = obj
        self.probe = probe
        return obj, probe

    # endregion

    # region --- adjoint ---

    def adjoint_operator(
        self,
        obj_array,
        probe_array,
        obj_patches,
        patch_row,
        patch_col,
        shifted_probes,
        amplitudes,
        overlap,
        step_size,
        fix_probe: bool = False,
    ):
        """Single-pass adjoint operator."""
        # print("amplitudes max min std: ", amplitudes.max(), amplitudes.min(), amplitudes.std())
        modified_overlap = self.fourier_projection(amplitudes, overlap)
        # print("modified overlap max min std: ", modified_overlap.max(), modified_overlap.min(), modified_overlap.std())
        gradient = self.gradient_step(overlap, modified_overlap)
        print("gradient max min std: ", gradient.max(), gradient.min(), gradient.std())
        obj_array, probe_array = self.update_object_and_probe(
            obj_array,
            probe_array,
            obj_patches,
            patch_row,
            patch_col,
            shifted_probes,
            gradient,
            step_size,
            fix_probe=fix_probe,
        )
        return obj_array, probe_array

    def fourier_projection(self, measured_amplitudes, overlap_array):
        """Replaces the Fourier amplitude of overlap with the measured data."""
        xp = self.xp
        fourier_overlap = xp.fft.fft2(overlap_array)
        # print("f_overlap max/min/mean: ", fourier_overlap.max(), fourier_overlap.min(), fourier_overlap.mean())
        if self.num_probes == 1:
            fourier_modified_overlap = measured_amplitudes * xp.exp(
                1.0j * xp.angle(fourier_overlap)
            )
        else:
            farfield_amplitudes = self.estimate_amplitudes(overlap_array)
            farfield_amplitudes[farfield_amplitudes == 0] = xp.inf
            amplitude_modification = measured_amplitudes / farfield_amplitudes
            fourier_modified_overlap = amplitude_modification[:, None] * fourier_overlap

        return xp.fft.ifft2(fourier_modified_overlap)

    def gradient_step(self, overlap_array, modified_overlap_array):
        """Computes analytical gradient."""
        return modified_overlap_array - overlap_array

    def update_object_and_probe(
        self,
        obj_array,
        probe_array,
        obj_patches,
        patch_row,
        patch_col,
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
            probe_normalization = self.xp.zeros_like(obj_array[s])
            object_update = self.xp.zeros_like(obj_array[s])

            for a0 in range(self.num_probes):
                probe = probe_slice[a0]
                obj = obj_slice[a0]
                grad = gradient[a0]
                # object-update
                probe_normalization += sum_patches(
                    self.xp.abs(probe) ** 2,
                    patch_row,
                    patch_col,
                    obj_shape,
                ).max()

                if self.object_type == "potential":
                    object_update += step_size * sum_patches(
                        self.xp.real(
                            -1j * self.xp.conj(obj) * self.xp.conj(probe) * grad
                        ),
                        patch_row,
                        patch_col,
                        obj_shape,
                    )
                else:
                    object_update += step_size * sum_patches(
                        self.xp.conj(probe) * grad,
                        patch_row,
                        patch_col,
                        obj_shape,
                    )

                obj_array[s] += object_update / probe_normalization

                # back-transmit
                gradient *= self.xp.conj(obj_slice)

                if s > 0:
                    raise NotImplementedError
                    # back-propagate
                    gradient = self._propagate_array(
                        gradient, self.xp.conj(self._propagators[s - 1])
                    )
                elif not fix_probe:
                    obj_normalization = self.xp.sum(
                        self.xp.abs(obj_slice) ** 2, axis=(0, 1)
                    ).max()
                    probe_array = probe_array + (
                        step_size
                        * self.xp.sum(gradient, axis=(0, 1))
                        / obj_normalization
                    )
        return obj_array, probe_array

    # endregion
