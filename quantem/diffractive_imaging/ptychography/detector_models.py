from typing import Self, Tuple

import torch

from quantem.core.datastructures import Dataset2d, Dataset3d
from quantem.core.io.serialize import AutoSerialize

# region --- Data Loaders ---


class SimpleBatcher:
    def __init__(
        self, indices: torch.Tensor, batch_size: int | None = None, shuffle: bool = True
    ):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.indices = self.indices[torch.randperm(len(self.indices))]
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i : i + self.batch_size]


# endregion --- Data Loaders ---

# region --- Pixelated Detector ---


class PixelatedDetectorModel(AutoSerialize):
    """ """

    _token = object()

    def __init__(
        self,
        intensity_data: torch.Tensor,
        data_loader: SimpleBatcher,
        reciprocal_sampling: Tuple[int, int],
        roi_shape: Tuple[int, int],
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use PixelatedDetectorModel.from_dataset3d() to instantiate this class."
            )

        self.intensity_data = intensity_data
        self.data_loader = data_loader
        self.reciprocal_sampling = reciprocal_sampling
        self.roi_shape = roi_shape
        self.sampling = tuple(
            1 / s / n for s, n in zip(reciprocal_sampling, self.roi_shape)
        )

    @classmethod
    def from_dataset3d(
        cls,
        intensity_data: Dataset3d,
    ) -> Self:
        """ """

        reciprocal_sampling = intensity_data.sampling[-2:]
        roi_shape = intensity_data.shape[-2:]
        data_loader = SimpleBatcher(
            torch.arange(intensity_data.shape[0]).to(torch.long)
        )

        return cls(
            intensity_data.array,
            data_loader,
            reciprocal_sampling,
            roi_shape,
            cls._token,
        )

    def forward(self, diffraction_intensity: torch.Tensor):
        """ """
        return diffraction_intensity


# endregion --- Pixelated Detector ---

# region ---Segmented Detector ---


class SegmentedDetectorModel(PixelatedDetectorModel):
    """ """

    _token = object()

    def __init__(
        self,
        intensity_data: torch.Tensor,
        detector_masks: torch.Tensor,
        data_loader: SimpleBatcher,
        reciprocal_sampling: Tuple[int, int],
        roi_shape: Tuple[int, int],
        num_segments: int,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use SegmentedDetectorModel.from_dataset3d() to instantiate this class."
            )

        self.intensity_data = intensity_data
        self.detector_masks = detector_masks
        self.num_segments = num_segments
        self.data_loader = data_loader
        self.reciprocal_sampling = reciprocal_sampling
        self.roi_shape = roi_shape
        self.sampling = tuple(
            1 / s / n for s, n in zip(reciprocal_sampling, self.roi_shape)
        )

    @classmethod
    def from_dataset2d(
        cls,
        intensity_data: Dataset2d,
        detector_masks: Dataset3d,
    ) -> Self:
        """ """

        reciprocal_sampling = detector_masks.sampling[-2:]
        roi_shape = detector_masks.shape[-2:]
        data_loader = SimpleBatcher(
            torch.arange(intensity_data.shape[0]).to(torch.long)
        )

        num_segments = detector_masks.shape[0]
        if num_segments != intensity_data.shape[-1]:
            raise ValueError()

        return cls(
            intensity_data.array,
            detector_masks.array,
            data_loader,
            reciprocal_sampling,
            roi_shape,
            num_segments,
            cls._token,
        )

    def forward(self, diffraction_intensity: torch.Tensor):
        """ """
        masked_intensity = []
        for mask in self.detector_masks:
            masked_intensity.append(
                torch.sum(diffraction_intensity * mask, dim=(-2, -1))
            )

        return torch.stack(masked_intensity, dim=-1)


# endregion --- Segmented Detector ---
