# from typing import Union, Sequence
import numbers
from collections.abc import Sequence
from typing import List, Union

import numpy as np

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d


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
            image_list, scan_direction_degrees, pad_fraction, pad_value, number_knots
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
        self.scan_fast = np.array(
            [
                np.array(
                    (
                        np.sin(phi),
                        np.cos(phi),
                    )
                )
                for phi in self.scan_direction
            ]
        )
        self.scan_slow = np.array(
            [
                np.array(
                    (
                        np.cos(phi),
                        -np.sin(phi),
                    )
                )
                for phi in self.scan_direction
            ]
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

        # Initialize Bezier knots for scanlines
        self.knots = []
        for a0 in range(self.shape[0]):
            shape = self.images[a0].shape

            v_slow = np.arange(shape[0])[:, None]
            v_fast = np.linspace(0, shape[1], self.number_knots)[None, :]

            rows = v_fast * self.scan_fast[a0][a0, 0] + v_slow * self.scan_slow[a0, 0]
            # cols = v_fast * self.scan_fast[a0,1] \
            #     + v_slow * self.scan_slow[a0,1]

            # print(rows.shape)

            self.knots.append(rows)

        # Generate initial resampled images

    def image_resample(
        self,
    ):
        pass
