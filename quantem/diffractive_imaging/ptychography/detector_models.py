from typing import Self, Tuple

import torch

from quantem.core.datastructures import Dataset3d
from quantem.core.io.serialize import AutoSerialize

# region --- data loaders ---


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


# endregion --- data loaders ---

# region --- PixelatedDetector ---


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
                "Use PixelatedDetector.from_array() to instantiate this class."
            )

        self.intensity_data = intensity_data
        self.roi_shape = self.intensity_data.shape[-2:]
        self.data_loader = data_loader
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

    def forward(self, batch_idx: torch.Tensor):
        """ """
        return self.intensity_data[batch_idx]


# endregion --- PixelatedDetector ---
