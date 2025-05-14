import numbers
from collections.abc import Sequence
from typing import List, Union

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils_imaging import bilinear_kde, cross_correlation_shift
from quantem.core.visualization import show_2d


class DriftCorrection(AutoSerialize):
    def __init__(self):
        raise RuntimeError(
            "Use Drift.from_data(...) or Drift.from_file(...) to create a Drift object."
        )

    @classmethod
    def from_file(
        cls,
        file_paths: Sequence[str],
        scan_direction_degrees: Union[Sequence[float], np.array],
        file_type: str | None = None,
        pad_fraction: float = 0.25,
        pad_value: Union[float, str] = "median",
        number_knots: int = 1,
    ) -> "DriftCorrection":
        image_list = [Dataset2d.from_file(fp, file_type=file_type) for fp in file_paths]
        return cls.from_data(image_list, scan_direction_degrees, pad_fraction)

    @classmethod
    def from_data(
        cls,
        images: Union[List[Dataset2d], List[np.ndarray], Dataset3d, np.ndarray],
        scan_direction_degrees: Union[List[float], np.ndarray],
        pad_fraction: float = 0.25,
        pad_value: Union[float, str] = "median",
        number_knots: int = 1,
    ) -> "DriftCorrection":
        if isinstance(images, Dataset3d):
            image_list = [
                Dataset2d.from_array(
                    images.array[i],
                    origin=images.origin[:2],
                    sampling=images.sampling[:2],
                    units=images.units[:2],
                )
                for i in range(images.array.shape[0])
            ]
        elif isinstance(images, np.ndarray) and images.ndim == 3:
            image_list = [Dataset2d.from_array(im) for im in images]
        elif isinstance(images, list):
            if all(isinstance(im, Dataset2d) for im in images):
                image_list = images
            elif all(isinstance(im, np.ndarray) and im.ndim == 2 for im in images):
                image_list = [Dataset2d.from_array(im) for im in images]
            else:
                raise TypeError(
                    "If passing a list, all elements must be either 2D numpy arrays or Dataset2d instances."
                )
        else:
            raise TypeError(
                "images must be a Dataset3d, a 3D ndarray, or a list of 2D arrays or Dataset2d instances."
            )

        # Construct Drift instance
        self = object.__new__(cls)
        self._initialize(
            image_list,
            scan_direction_degrees,
            pad_fraction,
            pad_value,
            number_knots,
        )
        return self

    def _initialize(
        self,
        images,
        scan_direction_degrees,
        pad_fraction,
        pad_value,
        number_knots=1,
    ):
        # Input data
        self.images = images
        self.scan_direction_degrees = np.array(scan_direction_degrees)
        self.pad_fraction = pad_fraction
        self.number_knots = number_knots

        # Derived data
        self.scan_direction = np.deg2rad(self.scan_direction_degrees)
        self.scan_fast = np.stack(
            [
                np.sin(-self.scan_direction),
                np.cos(-self.scan_direction),
            ],
            axis=1,
        )
        self.scan_slow = np.stack(
            [
                np.cos(-self.scan_direction),
                -np.sin(-self.scan_direction),
            ],
            axis=1,
        )
        self.shape = (
            len(self.images),
            int(np.round(self.images[0].shape[0] * (1 + self.pad_fraction) / 2) * 2),
            int(np.round(self.images[1].shape[1] * (1 + self.pad_fraction) / 2) * 2),
        )
        if isinstance(pad_value, str):
            if pad_value == "median":
                self.pad_value = [np.median(im.array) for im in self.images]
            elif pad_value == "mean":
                self.pad_value = [np.mean(im.array) for im in self.images]
            elif pad_value == "min":
                self.pad_value = [np.min(im.array) for im in self.images]
            elif pad_value == "max":
                self.pad_value = [np.max(im.array) for im in self.images]
        elif isinstance(pad_value, numbers.Number):
            if pad_value < 0.0:
                raise ValueError(f"pad_value of {pad_value} is < 0.0")
            if pad_value > 1.0:
                raise ValueError(f"pad_value of {pad_value} is > 1.0")
            self.pad_value = [np.quantile(im.array, pad_value) for im in self.images]
        else:
            raise TypeError(
                f"pad_value must be a 0.0 < float < 1.0, or one of ['median', 'mean', 'min', 'max'], got {type(pad_value)}"
            )

        # Initialize Bezier knots and scan vectors for scanlines
        self.knots = []
        for a0 in range(self.shape[0]):
            shape = self.images[a0].shape

            v_slow = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0])
            u_fast = np.linspace(
                -(shape[1] - 1) / 2, (shape[1] - 1) / 2, self.number_knots
            )

            xa = (
                (self.shape[1] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 0]
                + v_slow[:, None] * self.scan_slow[a0, 0]
            )
            ya = (
                (self.shape[2] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 1]
                + v_slow[:, None] * self.scan_slow[a0, 1]
            )

            self.knots.append(np.stack([xa, ya], axis=0))

        # Precompute the interpolator for all images
        self.interpolator = []
        for a0 in range(self.shape[0]):
            self.interpolator.append(
                DriftInterpolator(
                    input_shape=self.images[a0].shape,
                    output_shape=self.shape[1:],
                    scan_fast=self.scan_fast[a0],
                    scan_slow=self.scan_slow[a0],
                    pad_value=self.pad_value[a0],
                )
            )

        # Generate initial resampled images
        self.images_transform = Dataset3d.from_shape(self.shape)
        for a0 in range(self.shape[0]):
            self.images_transform.array[a0] = self.interpolator[a0].warp_image(
                self.images[a0].array,
                self.knots[a0],
            )

    # Translation alignment
    def align_translation(
        self,
        # window_edge_fraction=1,
        upsample_factor: int = 8,
        max_shift: int = 32,
    ):
        """
        Solve for the translation between all images in DriftCorrection.images_transform
        """

        # init
        dxy = np.zeros((self.shape[0], 2))
        # window = (
        #     tukey(self.shape[1], alpha=window_edge_fraction)[:, None]
        #     * tukey(self.shape[2], alpha=window_edge_fraction)[None, :]
        # )

        # loop over images
        F_ref = np.fft.fft2(self.images_transform.array[0])  #  * window
        for ind in range(1, self.shape[0]):
            shifts, image_shift = cross_correlation_shift(
                F_ref,
                np.fft.fft2(self.images_transform.array[ind]),  # * window
                upsample_factor=upsample_factor,
                max_shift=max_shift,
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )

            dxy[ind, :] = shifts
            F_ref = F_ref * ind / (ind + 1) + image_shift / (ind + 1)

        # Normalize dxy
        dxy -= np.mean(dxy, axis=0)

        # Apply shifts to knots
        for ind in range(self.shape[0]):
            self.knots[ind][0] += dxy[ind, 0]
            self.knots[ind][1] += dxy[ind, 1]

        # Regenerate images
        for a0 in range(self.shape[0]):
            self.images_transform.array[a0] = self.interpolator[a0].warp_image(
                self.images[a0].array,
                self.knots[a0],
            )

    # Affine alignment
    def align_affine(
        self,
        step=0.01,
        num_tests=9,  # should be odd
        refine=True,
        # window_edge_fraction = 1.0,
        upsample_factor: int = 8,
        max_shift: int = 32,
    ):
        """
        Estimate affine drift from the first 2 images.
        """

        # Window function for translation estimate
        # window = (
        #     tukey(self.shape[1], alpha=window_edge_fraction)[:, None]
        #     * tukey(self.shape[2], alpha=window_edge_fraction)[None, :]
        # )

        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        xx, yy = np.meshgrid(vec, vec, indexing="ij")
        keep = xx**2 + yy**2 <= (num_tests / 2) ** 2
        dxy = (
            np.vstack(
                (
                    xx[keep],
                    yy[keep],
                )
            ).T
            * step
        )

        # dxy = np.array((
        #     (0.0,0.1),
        #     (0.0,0.0),
        # ))

        # Measure cost function
        cost = np.zeros(dxy.shape[0])
        for a0 in tqdm(range(dxy.shape[0]), desc="Solving affine drift"):
            # updated knots
            knot_0 = self.knots[0].copy()
            u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
            knot_0[0] += dxy[a0, 0] * u[:, None]
            knot_0[1] += dxy[a0, 1] * u[:, None]

            knot_1 = self.knots[1].copy()
            u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
            knot_1[0] += dxy[a0, 0] * u[:, None]
            knot_1[1] += dxy[a0, 1] * u[:, None]

            im0 = self.interpolator[0].warp_image(
                self.images[0].array,
                knot_0,
            )
            im1 = self.interpolator[1].warp_image(
                self.images[1].array,
                knot_1,
            )
            # Cross correlation alignment
            shifts, image_shift = cross_correlation_shift(
                im0,
                im1,
                upsample_factor=upsample_factor,
                fft_input=False,
                fft_output=False,
                return_shifted_image=True,
                max_shift=max_shift,
            )
            cost[a0] = np.mean(np.abs(im0 - image_shift))

            # import matplotlib.pyplot as plt
            # fig,ax = plt.subplots(1,3,figsize=(6,3))
            # ax[0].imshow(im0)
            # ax[1].imshow(im1)
            # ax[2].imshow(im0 + image_shift)

        # update all knots
        ind = np.argmin(cost)
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += dxy[ind, 0] * u[:, None]
            self.knots[a0][1] += dxy[ind, 1] * u[:, None]

        # Affine drift refinement
        if refine:
            # Potential drift vectors
            dxy /= num_tests - 1

            # Measure cost function
            cost = np.zeros(dxy.shape[0])
            for a0 in tqdm(range(dxy.shape[0]), desc="Refining affine drift"):
                # updated knots

                knot_0 = self.knots[0].copy()
                u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
                knot_0[0] += dxy[a0, 0] * u[:, None]
                knot_0[1] += dxy[a0, 1] * u[:, None]

                knot_1 = self.knots[1].copy()
                u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
                knot_1[0] += dxy[a0, 0] * u[:, None]
                knot_1[1] += dxy[a0, 1] * u[:, None]

                im0 = self.interpolator[0].warp_image(
                    self.images[0].array,
                    knot_0,
                )
                im1 = self.interpolator[1].warp_image(
                    self.images[1].array,
                    knot_1,
                )
                # Cross correlation alignment
                shifts, image_shift = cross_correlation_shift(
                    im0,
                    im1,
                    upsample_factor=upsample_factor,
                    fft_input=False,
                    fft_output=False,
                    return_shifted_image=True,
                    max_shift=max_shift,
                )
                cost[a0] = np.mean(np.abs(im0 - image_shift))

            # update all knots
            ind = np.argmin(cost)
            for a0 in range(self.shape[0]):
                u = (
                    np.arange(self.knots[a0].shape[1])
                    - (self.knots[a0].shape[1] - 1) / 2
                )
                self.knots[a0][0] += dxy[ind, 0] * u[:, None]
                self.knots[a0][1] += dxy[ind, 1] * u[:, None]

        # Regenerate images
        for a0 in range(self.shape[0]):
            self.images_transform.array[a0] = self.interpolator[a0].warp_image(
                self.images[a0].array,
                self.knots[a0],
            )

        # Translation alignment
        self.align_translation(
            max_shift=max_shift,
        )

        # import matplotlib.pyplot as plt
        # fig,ax = plt.subplots()
        # xx[:] = 0
        # xx[keep] = cost
        # ax.imshow(
        #     xx,
        #     vmin = np.min(cost),
        #     vmax = np.max(cost),
        # )

        # k = knots_test_all[0][3]
        # print(k)
        # print(self.knots[0].shape)

        # import matplotlib.pyplot as plt
        # fig,ax = plt.subplots()
        # ax.imshow(keep)

    # non-rigid alignment

    def plot_merged_images(self, show_knots: bool = True, **kwargs):
        """
        Plot the current transformed images, with knot overlays.
        """

        fig, ax = show_2d(
            self.images_transform.array.mean(0),
            # self.images_transform.array[0],
            # self.images[0].array,
            **kwargs,
        )

        if show_knots:
            for a0 in range(self.shape[0]):
                x = self.knots[a0][0]
                y = self.knots[a0][1]
                ax.plot(
                    y,
                    x,
                    # color = 'r',
                )


class DriftInterpolator:
    def __init__(
        self,
        input_shape,
        output_shape,
        scan_fast,
        scan_slow,
        pad_value,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.scan_fast = scan_fast
        self.scan_slow = scan_slow
        self.pad_value = pad_value

        self.rows_input = np.arange(input_shape[0])
        self.cols_input = np.arange(input_shape[1])

        self.u = np.linspace(0, 1, input_shape[1])
        # self.v = np.linspace(0, 1, input_shape[0])

    def warp_image(
        self,
        image: np.ndarray,
        knots: np.ndarray,  # shape: (2, rows, num_knots)
        kde_sigma: float = 0.5,
    ) -> np.ndarray:
        num_knots = knots.shape[-1]
        basis = np.linspace(0, 1, num_knots)

        xa = np.zeros(self.input_shape)
        ya = np.zeros(self.input_shape)

        if num_knots == 1:
            xa[:] = knots[0, :] + self.u[None, :] * self.scan_fast[0] * (
                self.input_shape[0] - 1
            )
            ya[:] = knots[1, :] + self.u[None, :] * self.scan_fast[1] * (
                self.input_shape[1] - 1
            )
        elif num_knots == 2:
            for i in range(self.input_shape[0]):
                xa[i, :] = interp1d(
                    basis, knots[0, i], kind="linear", assume_sorted=True
                )(self.u)
                ya[i, :] = interp1d(
                    basis, knots[1, i], kind="linear", assume_sorted=True
                )(self.u)
        else:
            kind = "quadratic" if num_knots == 3 else "cubic"
            for i in range(self.input_shape[0]):
                xa[i, :] = interp1d(
                    basis,
                    knots[0, i],
                    kind=kind,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )(self.u)
                ya[i, :] = interp1d(
                    basis,
                    knots[1, i],
                    kind=kind,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )(self.u)

        image_interp = bilinear_kde(
            xa=xa,  # rows
            ya=ya,  # cols
            intensities=image,
            output_shape=self.output_shape,
            kde_sigma=kde_sigma,
            pad_value=self.pad_value,
        )

        return image_interp
