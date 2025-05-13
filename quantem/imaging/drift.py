import numbers
from collections.abc import Sequence
from typing import List, Union

import numpy as np

# from scipy.interpolate import CloughTocher2DInterpolator, interp1d
from scipy.interpolate import interp1d

# from scipy.interpolate import interpn
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.utils.utils_imaging import bilinear_kde


class DriftCorrection:
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
                np.sin(self.scan_direction),
                np.cos(self.scan_direction),
            ],
            axis=1,
        )
        self.scan_slow = np.stack(
            [
                -np.cos(self.scan_direction),
                np.sin(self.scan_direction),
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

            rows = (
                (self.shape[1] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 0]
                + v_slow[:, None] * self.scan_slow[a0, 0]
            )
            cols = (
                (self.shape[2] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 1]
                + v_slow[:, None] * self.scan_slow[a0, 1]
            )

            self.knots.append(np.stack([rows, cols], axis=0))

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
            im = self.images[a0]
            interpolator = self.interpolator[a0]
            self.images_transform.array[a0] = interpolator.warp_image(
                im.array,
                self.knots[a0],
            )

        # # Test plotting
        # # im = np.zeros()
        # for a0 in range(self.shape[0]):
        #     fig, ax = plt.subplots(figsize=(8, 8))
        #     ax.imshow(
        #         self.images_transform.array[a0],
        #     )
        #     x = self.knots[a0][0]
        #     y = self.knots[a0][1]
        #     # ax.scatter(
        #     #     y,
        #     #     x,
        #     #     marker=".",
        #     #     color="r",
        #     # )
        #     ax.plot(
        #         y,
        #         x,
        #         color = 'r',
        #     )
        #     ax.set_axis_off()


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

    def warp_image(
        self,
        image: np.ndarray,
        knots: np.ndarray,  # shape: (2, rows, num_knots)
        kde_sigma: float = 0.5,
    ) -> np.ndarray:
        num_knots = knots.shape[-1]
        basis = np.linspace(0, 1, num_knots)

        # Interpolate each scanline separately
        rows_interp = np.zeros((self.input_shape[0], self.input_shape[1]))
        cols_interp = np.zeros((self.input_shape[0], self.input_shape[1]))

        if num_knots == 1:
            # Simple linear mapping from single knot
            for i in range(self.input_shape[0]):
                rows_interp[i, :] = knots[0, i] + self.u * self.scan_fast[0] * (
                    self.input_shape[0] - 1
                )
                cols_interp[i, :] = knots[1, i] + self.u * self.scan_fast[1] * (
                    self.input_shape[1] - 1
                )

        elif num_knots == 2:
            # Linear interpolation between two knots
            for i in range(self.input_shape[0]):
                rows_interp[i, :] = interp1d(
                    basis, knots[0, i], kind="linear", assume_sorted=True
                )(self.u)
                cols_interp[i, :] = interp1d(
                    basis, knots[1, i], kind="linear", assume_sorted=True
                )(self.u)

        else:
            # Quadratic (3 knots) or cubic (4+ knots) interpolation
            kind = "quadratic" if num_knots == 3 else "cubic"
            for i in range(self.input_shape[0]):
                rows_interp[i, :] = interp1d(
                    basis,
                    knots[0, i],
                    kind=kind,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )(self.u)
                cols_interp[i, :] = interp1d(
                    basis,
                    knots[1, i],
                    kind=kind,
                    fill_value="extrapolate",
                    assume_sorted=True,
                )(self.u)

        image_interp = bilinear_kde(
            xa=rows_interp,
            ya=cols_interp,
            intensities=image,
            output_shape=self.output_shape,
            kde_sigma=kde_sigma,
            pad_value=self.pad_value,
        )

        return image_interp
