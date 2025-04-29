from typing import Literal

import numpy as np
from tqdm import trange

from quantem.core import config
from quantem.core.datastructures import Dataset, Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.diffractive_imaging.ptycho_utils import (
    fourier_shift,
    fourier_translation_operator,
    generate_batches,
    sum_patches,
)

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


### For now doing a barebones implementation of ptychography gradient descent
### This will be replaced with a more sophisticated implementation in the future

## struggling to have xp be cp, np _or_ torch, because torch requires Tensor inputs


class PtychographyGD(AutoSerialize):
    """
    A class for performing phase retrieval using the Ptychography Gradient Descent algorithm.

    This class implements the Ptychography Gradient Descent algorithm, which is a popular method for
    phase retrieval from a series of diffraction patterns.

    Attributes:
        probe (Dataset): The probe function to be used in the algorithm
    """

    def __init__(
        self,
        probe: Dataset,
        data: Dataset4dstem,
        device: Literal["cpu", "gpu"] = "cpu",
        verbose: int | bool = 1,
        object_padding_px: tuple[int, int] = (0, 0),
    ):
        self.xp = cp if device == "gpu" else np
        self.probe = self.xp.asarray(probe.array)
        self.initial_probe = self.xp.asarray(probe.array.copy())
        self.data = self.xp.asarray(data.array)
        self.device = device

        self.num_probes = 1
        self.num_slices = 1
        self.object_type = "complex"
        self._verbose = verbose

        ### skipping all the preprocessing for now
        self.com_rotation_rad = 0
        self.com_transpose = False
        self.sampling = probe.sampling
        self.scan_sampling = data.sampling[:2]
        self.shifted_amplitudes = self.xp.asarray(
            np.sqrt(
                np.fft.ifftshift(self.data, axes=(-1, -2)).reshape(
                    (-1,) + tuple(self.roi_shape)
                )
            )
        )

        self.object_padding_px = np.array(object_padding_px)
        self.positions_px = self._calculate_scan_positions_in_pixels(
            object_padding_px=self.object_padding_px,
        )
        self.patch_row, self.patch_col = self.get_patch_indices()
        self.object = self.xp.ones(self.object_shape_full)
        self._losses = []

    def _calculate_scan_positions_in_pixels(
        self,
        positions: np.ndarray | None = None,
        positions_mask: np.ndarray | None = None,
        object_padding_px: np.ndarray | None = None,
        positions_offset_ang: tuple[float, float] | None = None,
    ):
        """
        Method to compute the initial guess of scan positions in pixels.

        Parameters
        ----------
        positions: (J,2) np.ndarray or None
            Input probe positions in Å.
            If None, a raster scan using experimental parameters is constructed.
        positions_mask: np.ndarray, optional
            Boolean real space mask to select positions in datacube to skip for reconstruction
        object_padding_px: Tuple[int,int], optional
            Pixel dimensions to pad object with
            If None, the padding is set to half the probe ROI dimensions
        positions_offset_ang, np.ndarray, optional
            Offset of positions in A

        Returns
        -------
        positions_in_px: (J,2) np.ndarray
            Initial guess of scan positions in pixels
        object_padding_px: Tupe[int,int]
            Updated object_padding_px
        """

        rotation_angle = self.com_rotation_rad
        transpose = self.com_transpose
        sampling = self.sampling
        if object_padding_px is None:
            object_padding_px = np.array([0, 0])

        if positions is None:
            nx, ny = self.gpts
            sx, sy = self.scan_sampling
            x = np.arange(nx) * sx
            y = np.arange(ny) * sy

            x, y = np.meshgrid(x, y, indexing="ij")
            if positions_offset_ang is not None:
                x += positions_offset_ang[0]
                y += positions_offset_ang[1]

            if positions_mask is not None:
                x = x[positions_mask]
                y = y[positions_mask]

            positions = np.stack((x.ravel(), y.ravel()), axis=-1)
        else:
            positions = np.array(positions)

        if rotation_angle != 0:
            raise NotImplementedError
            # tf = AffineTransform(angle=rotation_angle)
            # positions = tf(positions, positions.mean(0))
            # assert isinstance(positions, np.ndarray)

        if transpose:
            positions = np.flip(positions, axis=1)
            sampling = sampling[::-1]

        # ensure positive
        m: np.ndarray = np.min(positions, axis=0).clip(-np.inf, 0)  # typing being weird
        positions -= m

        # finally, switch to pixels
        positions[:, 0] /= sampling[0]
        positions[:, 1] /= sampling[1]

        # top-left padding
        positions[:, 0] += object_padding_px[0]
        positions[:, 1] += object_padding_px[1]
        return positions

    def get_patch_indices(self):
        """
        Sets the vectorized row/col indices used for the overlap projection
        Note this assumes the probe is corner-centered.

        Returns
        -------
        self._vectorized_patch_indices_row: np.ndarray
            Row indices for probe patches inside object array
        self._vectorized_patch_indices_col: np.ndarray
            Column indices for probe patches inside object array
        """
        x0 = np.round(self.positions_px[:, 0]).astype(np.int32)
        y0 = np.round(self.positions_px[:, 1]).astype(np.int32)

        x_ind = np.fft.fftfreq(self.roi_shape[0], d=1 / self.roi_shape[0]).astype(
            np.int32
        )
        y_ind = np.fft.fftfreq(self.roi_shape[1], d=1 / self.roi_shape[1]).astype(
            np.int32
        )
        row = (x0[:, None, None] + x_ind[None, :, None]) % self.object_shape_full[-2]
        col = (y0[:, None, None] + y_ind[None, None, :]) % self.object_shape_full[-1]

        return row, col

    @property
    def fov(self) -> np.ndarray:
        return self.scan_sampling * (self.gpts - 1)

    @property
    def gpts(self) -> np.ndarray:
        return np.array(self.data.shape[:2])

    @property
    def roi_shape(self) -> tuple[int, ...]:
        return self.data.shape[2:]

    @property
    def positions_px_fractional(self) -> np.ndarray:
        return self.positions_px - np.round(self.positions_px)

    def vprint(self, *args, **kwargs):
        """Print messages if verbose is enabled."""
        if self._verbose:
            print(*args, **kwargs)

    @property
    def object_shape_crop(self) -> np.ndarray:
        shp = np.floor(self.fov / self.sampling)
        shp += shp % 2

        if self.num_slices > 1:
            shp = np.concatenate([[self.num_slices], shp])

        return shp.astype("int")

    @property
    def object_shape_full(self) -> np.ndarray:
        # return self.object_shape_crop + 2 * self.object_padding_px # if com_rotation = 0
        cshape = self.object_shape_crop.copy()
        shape = np.floor(
            [
                abs(cshape[-1] * np.sin(self.com_rotation_rad))
                + abs(cshape[-2] * np.cos(self.com_rotation_rad)),
                abs(cshape[-2] * np.sin(self.com_rotation_rad))
                + abs(cshape[-1] * np.cos(self.com_rotation_rad)),
            ]
        )
        shape += shape % 2
        shape += 2 * self.object_padding_px

        if self.num_slices > 1:
            shape = np.concatenate([[self.num_slices], shape])

        return shape.astype("int")

    def reconstruct(
        self,
        num_iter: int = 0,
        reset: bool = True,
        fix_probe: bool = False,
        batch_size: int | None = None,
        step_size: float = 0.5,
    ):
        # to_xp for object, probe, data

        if batch_size is None:
            batch_size = self.gpts[0] * self.gpts[1]
            num_batches = 1
        else:
            num_batches = 1 + ((self.gpts[0] * self.gpts[1]) // batch_size)

        if reset:
            object = self.xp.ones(self.object_shape_full)
            probe = self.initial_probe.copy()
            self._losses = []
        else:
            object = self.object
            probe = self.probe

        pos_frac = self.xp.asarray(self.positions_px_fractional)
        patch_row = self.xp.asarray(self.patch_row)
        patch_col = self.xp.asarray(self.patch_col)

        shuffled_indices = np.arange(self.gpts[0] * self.gpts[1])
        for a0 in trange(num_iter):
            np.random.shuffle(shuffled_indices)
            error = np.float32(0)
            for start, end in generate_batches(
                num_items=self.gpts[0] * self.gpts[1], max_batch=batch_size
            ):
                batch_indices = shuffled_indices[start:end]
                object, probe = self._single_slice_iter(
                    object,
                    probe,
                    patch_row[batch_indices],
                    patch_col[batch_indices],
                    pos_frac[batch_indices],
                    self.shifted_amplitudes[batch_indices],
                    step_size,
                    fix_probe=fix_probe,
                )

                error += self.error_estimate(
                    object,
                    probe,
                    patch_row[batch_indices],
                    patch_col[batch_indices],
                    pos_frac[batch_indices],
                    self.shifted_amplitudes[batch_indices],
                )

            self._losses.append(error.item() / num_batches)

        if self.device == "gpu":
            if isinstance(object, cp.ndarray):
                object = self._to_numpy(object)
            if isinstance(probe, cp.ndarray):
                probe = self._to_numpy(probe)

        self.object = object
        self.probe = probe
        return object, probe

    def _to_numpy(self, array):
        if self.device == "gpu":
            if isinstance(array, cp.ndarray):
                return array.get()
        return array

    def shift_probes(self, probe_array, fract_positions):
        """Fourier-shifts probe_array to subpixel `positions`"""
        return fourier_shift(probe_array, fract_positions)

    def object_patches(self, obj_array, patch_row, patch_col):
        """Extracts roi-shaped patches from `obj_array`."""
        # maybe add this back here cuz it kinda makes sense
        if self.object_type == "potential":
            obj_array = self.xp.exp(1j * obj_array)
        return obj_array[..., patch_row, patch_col]

    def overlap_projection(self, obj_patches, shifted_probes, descan_shifts=None):
        """Multiplies `shifted_probes` with roi-shaped patches from `obj_array`."""
        if self.num_probes > 1:  # mixed state
            obj_patches = obj_patches[:, None]

        if self.num_slices == 1:
            if descan_shifts is None:
                overlap = obj_patches * shifted_probes
            else:
                shifts = fourier_translation_operator(
                    descan_shifts, self.roi_shape, device=self.device
                )
                if self.num_probes > 1:
                    shifts = shifts[:, None]
                overlap = shifts * shifted_probes * obj_patches
        else:
            # shifted_probes_slices = self.xp.ones_like(obj_patches)
            # shifted_probes_slices[0] = shifted_probes
            # overlap = None
            # if descan_shifts is not None:
            #     raise NotImplementedError("descan with multislice")
            # for s in range(self.num_slices):
            #     overlap = obj_patches[s] * shifted_probes_slices[s]
            #     if s+1 < self.num_slices:
            #         shifted_probes_slices[s+1] = self._propagate_array(overlap, self._propagators[s])
            # shifted_probes = shifted_probes_slices
            raise NotImplementedError
        return shifted_probes, obj_patches, overlap

    def estimated_amplitudes(self, overlap_array: np.ndarray | cp.ndarray):
        """Returns the estimated fourier amplitudes from real-valued `overlap_array`."""
        xp = self.xp
        overlap_fft = xp.fft.fft2(overlap_array)
        if self.num_probes == 1:
            return xp.abs(overlap_fft)
        else:
            return xp.sqrt(xp.sum(xp.abs(overlap_fft) ** 2, axis=1))

    def fourier_projection(self, measured_amplitudes, overlap_array):
        """Replaces the Fourier amplitude of overlap with the measured data."""
        xp = self.xp
        fourier_overlap = xp.fft.fft2(overlap_array)
        if self.num_probes == 1:
            fourier_modified_overlap = measured_amplitudes * xp.exp(
                1.0j * xp.angle(fourier_overlap)
            )
        else:
            farfield_amplitudes = self.estimated_amplitudes(overlap_array)
            farfield_amplitudes[farfield_amplitudes == 0] = xp.inf
            amplitude_modification = measured_amplitudes / farfield_amplitudes
            fourier_modified_overlap = amplitude_modification[:, None] * fourier_overlap

        return xp.fft.ifft2(fourier_modified_overlap)

    def forward_operator(
        self,
        obj_array,
        probe_array,
        patch_row,
        patch_col,
        fract_positions,
        amplitudes,
    ):
        """Single-pass forward operator."""
        shifted_probes = self.shift_probes(probe_array, fract_positions)
        obj_patches = self.object_patches(obj_array, patch_row, patch_col)
        shifted_probes, obj_patches, overlap = self.overlap_projection(
            obj_patches, shifted_probes
        )
        modified_overlap = self.fourier_projection(amplitudes, overlap)

        return obj_patches, shifted_probes, overlap, modified_overlap

    def error_estimate(
        self,
        obj_array,
        probe_array,
        patch_row,
        patch_col,
        fract_positions,
        amplitudes,
        descan_shifts=None,
    ):
        """Computes the error between the measured and estimated amplitudes."""
        shifted_probes = self.shift_probes(probe_array, fract_positions)
        obj_patches = self.object_patches(obj_array, patch_row, patch_col)
        _, _, overlap = self.overlap_projection(
            obj_patches, shifted_probes, descan_shifts
        )
        farfield_amplitudes = self.estimated_amplitudes(overlap)
        mse = self.xp.mean(self.xp.abs(amplitudes - farfield_amplitudes) ** 2)

        return mse

    def _predicted_amplitudes(
        self,
        obj_array,
        probe_array,
        patch_row,
        patch_col,
        fract_positions,
    ):
        """Computes the error between the measured and estimated amplitudes.
        #TODO combine with error_estimate"""
        shifted_probes = self.shift_probes(probe_array, fract_positions)
        obj_patches = self.object_patches(obj_array, patch_row, patch_col)
        _, _, overlap = self.overlap_projection(obj_patches, shifted_probes)
        farfield_amplitudes = self.estimated_amplitudes(overlap)
        return farfield_amplitudes

    def gradient_step(self, overlap_array, modified_overlap_array):
        """Computes analytical gradient."""
        return modified_overlap_array - overlap_array

    #############
    ## Adjoint ##
    #############

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

        This is a stripped down version of _gradient_descent_adjoint, if failing might have to
        add more parts
        """
        obj_shape = obj_array.shape[-2:]

        # TODO  break these apart into functions for each case, mixed, multislice
        if self.num_probes == 1:
            if self.num_slices == 1:
                probe_normalization = sum_patches(
                    self.xp.abs(shifted_probes) ** 2,
                    patch_row,
                    patch_col,
                    obj_shape,
                ).max()

                if self.object_type == "potential":
                    obj_array = obj_array + (
                        step_size
                        * sum_patches(
                            self.xp.real(
                                -1j
                                * self.xp.conj(obj_patches)
                                * self.xp.conj(shifted_probes)
                                * gradient
                            ),
                            patch_row,
                            patch_col,
                            obj_shape,
                        )
                        / probe_normalization
                    )
                else:
                    obj_array = obj_array + (
                        step_size
                        * sum_patches(
                            self.xp.conj(shifted_probes) * gradient,
                            patch_row,
                            patch_col,
                            obj_shape,
                        )
                        / probe_normalization
                    )

                if not fix_probe:
                    obj_normalization = self.xp.sum(
                        self.xp.abs(obj_patches) ** 2, axis=0
                    ).max()
                    probe_array = probe_array + (
                        step_size
                        * self.xp.sum(self.xp.conj(obj_patches) * gradient, axis=0)
                        / obj_normalization
                    )
            else:  # multislice single probe
                raise NotImplementedError
                # for s in reversed(range(self.num_slices)):
                #     probe = shifted_probes[s]
                #     obj = obj_patches[s]

                #     # object-update
                #     probe_normalization = sum_patches(
                #         self.xp.abs(probe) ** 2,
                #         patch_row,
                #         patch_col,
                #         obj_shape,
                #     ).max()

                #     if self.object_type == "potential":
                #         obj_array[s] = obj_array[s] + (
                #             step_size
                #             * sum_patches(
                #                 self.xp.real(
                #                     -1j * self.xp.conj(obj) * self.xp.conj(probe) * gradient
                #                 ),
                #                 patch_row,
                #                 patch_col,
                #                 obj_shape,
                #             )
                #             / probe_normalization
                #         )
                #     else:
                #         obj_array[s] = obj_array[s] + (
                #             step_size
                #             * sum_patches(
                #                 self.xp.conj(probe) * gradient,
                #                 patch_row,
                #                 patch_col,
                #                 obj_shape,
                #             )
                #             / probe_normalization
                #         )

                #     # back-transmit
                #     gradient *= self.xp.conj(obj)

                #     if s > 0:
                #         # back-propagate
                #         gradient = self._propagate_array(
                #             gradient, self.xp.conj(self._propagators[s - 1])
                #         )
                #     elif not fix_probe:
                #         obj_normalization = self.xp.sum(self.xp.abs(obj_patches) ** 2, dim=0).max()
                #         probe_array = probe_array + (
                #             step_size
                #             * self.xp.sum(gradient, dim=0)
                #             / obj_normalization
                #         )

        else:
            if self.num_slices > 1:
                raise NotImplementedError("mixed multislice")
            probe_normalization = self.xp.zeros(
                obj_array.shape
            )  # , device=obj_array.device)
            object_update = self.xp.zeros_like(obj_array)
            for a0 in range(self.num_probes):
                probe_normalization += sum_patches(
                    self.xp.abs(shifted_probes[:, a0]) ** 2,
                    patch_row,
                    patch_col,
                    obj_shape,
                ).max()

                if self.object_type == "potential":
                    object_update += step_size * sum_patches(
                        self.xp.real(
                            -1j
                            * self.xp.conj(obj_patches.squeeze())
                            * self.xp.conj(shifted_probes[:, a0])
                            * gradient[:, a0]
                        ),
                        patch_row,
                        patch_col,
                        obj_shape,
                    )
                else:
                    object_update += step_size * sum_patches(
                        self.xp.conj(shifted_probes[:, a0]) * gradient[:, a0],
                        patch_row,
                        patch_col,
                        obj_shape,
                    )

            obj_array += object_update / self.xp.max(probe_normalization)

            if not fix_probe:
                obj_normalization = self.xp.sum(
                    self.xp.abs(obj_patches) ** 2, axis=0
                ).max()
                probe_array = probe_array + (
                    step_size
                    * self.xp.sum(self.xp.conj(obj_patches) * gradient, axis=0)
                    / obj_normalization
                )

        return obj_array, probe_array

    def adjoint_operator(
        self,
        obj_array,
        probe_array,
        obj_patches,
        patch_row,
        patch_col,
        shifted_probes,
        overlap_array,
        modified_overlap_array,
        step_size,
        fix_probe: bool = False,
    ):
        """Single-pass adjoint operator."""
        gradient = self.gradient_step(overlap_array, modified_overlap_array)
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

    def _single_slice_iter(
        self,
        obj_array,
        probe_array,
        patch_row,
        patch_col,
        fract_positions,
        amplitudes,
        step_size,
        fix_probe: bool = False,
    ):
        """Single-pass of single-slice algorithm."""
        obj_patches, shifted_probes, overlap, modified_overlap = self.forward_operator(
            obj_array, probe_array, patch_row, patch_col, fract_positions, amplitudes
        )

        obj_array, probe_array = self.adjoint_operator(
            obj_array,
            probe_array,
            obj_patches,
            patch_row,
            patch_col,
            shifted_probes,
            overlap,
            modified_overlap,
            step_size,
            fix_probe=fix_probe,
        )

        return obj_array, probe_array
