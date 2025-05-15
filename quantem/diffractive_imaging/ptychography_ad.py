import numpy as np
import torch
from torch._tensor import Tensor
from tqdm import trange

from quantem.core import config
from quantem.core.datastructures import Dataset, Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.diffractive_imaging.ptycho_utils import (
    fourier_shift,
    fourier_translation_operator,
    generate_batches,
)

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


### For now doing a barebones implementation of ptychography gradient descent
### This will be replaced with a more sophisticated implementation in the future

## struggling to have xp be cp, np _or_ torch, because torch requires Tensor inputs


class PtychographyAD(AutoSerialize):
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
        device: str = "cuda:0",
        verbose: int | bool = 1,
        object_padding_px: tuple[int, int] = (0, 0),
    ):
        self.probe = np.asarray(probe.array)
        self.initial_probe = np.asarray(probe.array.copy())
        self.data = np.asarray(data.array)
        self.device = device

        self._optimizers = {}

        self.num_probes = 1
        self.num_slices = 1
        self.object_type = "complex"
        self._verbose = verbose

        ### skipping all the preprocessing for now
        self.com_rotation_rad = 0
        self.com_transpose = False
        self.sampling = probe.sampling
        self.scan_sampling = data.sampling[:2]
        self.shifted_amplitudes = np.asarray(
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
        self.object = np.ones(self.object_shape_full)
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

    def _to_torch(self, array: np.ndarray, dtype: torch.dtype | None = None) -> Tensor:
        return torch.tensor(array.copy(), device=self.device, dtype=dtype)

    def reconstruct(
        self,
        num_iter: int = 0,
        reset: bool = True,
        fix_probe: bool = False,
        batch_size: int | None = None,
        lr: float = 1e-3,
    ):
        if batch_size is None:
            batch_size = self.gpts[0] * self.gpts[1]
            num_batches = 1
        else:
            num_batches = 1 + ((self.gpts[0] * self.gpts[1]) // batch_size)

        if reset:
            object = np.ones(self.object_shape_full)
            probe = self.initial_probe.copy()
            self._losses = []
            object = self._to_torch(object, dtype=torch.complex128)
            probe = self._to_torch(self.initial_probe, dtype=torch.complex128)
        else:
            object = self._to_torch(self.object)
            probe = self._to_torch(self.probe)

        object.requires_grad = True
        if not fix_probe:
            probe.requires_grad = True

        self._optimizers = {"object": torch.optim.Adam(params=[object], lr=lr)}
        if not fix_probe:
            self._optimizers["probe"] = torch.optim.Adam(params=[probe], lr=lr)

        pos_frac = self._to_torch(self.positions_px_fractional)
        patch_row = self._to_torch(self.patch_row)
        patch_col = self._to_torch(self.patch_col)
        amplitudes = self._to_torch(self.shifted_amplitudes)

        shuffled_indices = np.arange(self.gpts[0] * self.gpts[1])
        for a0 in trange(num_iter):
            np.random.shuffle(shuffled_indices)
            loss = torch.tensor(0, device=self.device, dtype=torch.float64)
            for start, end in generate_batches(
                num_items=self.gpts[0] * self.gpts[1], max_batch=batch_size
            ):
                batch_indices = shuffled_indices[start:end]
                loss += (
                    self.error_estimate(
                        object,
                        probe,
                        patch_row[batch_indices],
                        patch_col[batch_indices],
                        pos_frac[batch_indices],
                        amplitudes[batch_indices],
                    )
                    / num_batches
                )
            loss.backward()
            for opt in self._optimizers.values():
                opt.step()
                opt.zero_grad()

            self._losses.append(loss.item())

        object = object.detach().cpu().numpy()
        probe = probe.detach().cpu().numpy()
        self.object = object
        self.probe = probe
        return object, probe

    def shift_probes(self, probe_array, fract_positions):
        """Fourier-shifts probe_array to subpixel `positions`"""
        return fourier_shift(probe_array, fract_positions)

    def object_patches(self, obj_array, patch_row, patch_col):
        """Extracts roi-shaped patches from `obj_array`."""
        # maybe add this back here cuz it kinda makes sense
        if self.object_type == "potential":
            obj_array = torch.exp(1j * obj_array)
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
            # shifted_probes_slices = np.ones_like(obj_patches)
            # shifted_probes_slices[0] = shifted_probes
            # overlap = None
            # if descan_shifts is not None:
            #     raise NotImplementedError("descan with multislice")
            # for s in range(self.num_slices):
            #             overlap = obj_patches[s] * shifted_probes
            #             if s+1 < self.num_slices:
            #                 shifted_probes = self._propagate_array(overlap, self._propagators[s])
            raise NotImplementedError
        return shifted_probes, obj_patches, overlap

    def estimated_amplitudes(self, overlap_array: np.ndarray | cp.ndarray):
        """Returns the estimated fourier amplitudes from real-valued `overlap_array`."""
        overlap_fft = torch.fft.fft2(overlap_array)
        if self.num_probes == 1:
            return torch.abs(overlap_fft)
        else:
            return torch.sqrt(torch.sum(torch.abs(overlap_fft) ** 2, dim=1))

    def forward_operator(
        self,
        obj_array,
        probe_array,
        patch_row,
        patch_col,
        fract_positions,
        descan_shifts=None,
    ):
        """Single-pass forward operator."""
        shifted_probes = self.shift_probes(probe_array, fract_positions)
        obj_patches = self.object_patches(obj_array, patch_row, patch_col)
        shifted_probes, obj_patches, overlap = self.overlap_projection(
            obj_patches, shifted_probes, descan_shifts
        )

        return obj_patches, shifted_probes, overlap

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
        _, _, overlap = self.forward_operator(
            obj_array,
            probe_array,
            patch_row,
            patch_col,
            fract_positions,
            descan_shifts,
        )
        farfield_amplitudes = self.estimated_amplitudes(overlap)
        mse = torch.mean(torch.abs(amplitudes - farfield_amplitudes) ** 2)

        return mse
