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
        print(f"amp {amplitudes.shape} overlap: {overlap.shape}")
        modified_overlap = self.fourier_projection(amplitudes, overlap)
        # mod_overlap shape same as overlap: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        print("modified overlap shape: ", modified_overlap.shape)
        gradient = self.gradient_step(overlap, modified_overlap)
        # grad shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        print("gradient shape: ", gradient.shape)
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
        if self.num_probes == 1:
            fourier_modified_overlap = measured_amplitudes * xp.exp(
                1.0j * xp.angle(fourier_overlap)
            )
        else:
            farfield_amplitudes = self.estimate_amplitudes(overlap_array)
            farfield_amplitudes[farfield_amplitudes == 0] = xp.inf
            amplitude_modification = measured_amplitudes / farfield_amplitudes
            fourier_modified_overlap = amplitude_modification * fourier_overlap

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

        print("\tupd obj_patches.shape: ", obj_patches.shape)

        for s in reversed(range(self.num_slices)):
            probe_slice = shifted_probes[s]
            obj_slice = obj_patches[s]
            probe_normalization = self._to_xp(np.zeros_like(obj_array[s]))
            object_update = self._to_xp(np.zeros_like(obj_array[s]))
            print("\tobj slice: ", obj_slice.shape)

            for a0 in range(self.num_probes):
                probe = probe_slice[a0]
                obj = obj_slice
                grad = gradient[a0]
                # object-update
                probe_normalization += sum_patches(
                    np.abs(probe) ** 2,
                    patch_row,
                    patch_col,
                    obj_shape,
                ).max()

                if self.object_type == "potential":
                    object_update += step_size * sum_patches(
                        np.real(-1j * np.conj(obj) * np.conj(probe) * grad),
                        patch_row,
                        patch_col,
                        obj_shape,
                    )
                else:
                    object_update += step_size * sum_patches(
                        np.conj(probe) * grad,
                        patch_row,
                        patch_col,
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
                        step_size * np.sum(gradient, axis=(0, 1)) / obj_normalization
                    )
        return obj_array, probe_array

    # endregion
