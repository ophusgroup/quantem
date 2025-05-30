import math
from typing import Self, Tuple

import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from quantem.core.datastructures import Dataset
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import SimpleTorchBatcher
from quantem.core.utils.validators import ensure_valid_tensor

# region --- CoM Origin Model ---


class CenterOfMassOriginModel(AutoSerialize):
    """ """

    _token = object()

    def __init__(self, dataset: Dataset, _token: object | None = None):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use CenterOfMassOriginModel.from_dataset() to instantiate this class."
            )

        self.dataset = dataset
        self.num_dps = math.prod(self.dataset.shape[:-2])
        self._tensor = ensure_valid_tensor(self.dataset.array)

        # defaults
        self._origin_measured = None
        self._origin_fitted = None
        self._shifted_tensor = None

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
    ) -> Self:
        """ """
        return cls(
            dataset=dataset,
            _token=cls._token,
        )

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset):
        if isinstance(value, Dataset):
            self._dataset = value
        else:
            raise TypeError("dataset must be a valid Dataset.")

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @tensor.setter
    def tensor(self, value: torch.Tensor):
        self._tensor = ensure_valid_tensor(value)
        self._dataset.array = self._tensor.detach().numpy()

    def calculate_origin(
        self,
        max_batch_size: int | None = None,
    ):
        """ """
        nqx, nqy = self.dataset.shape[-2:]
        tensor_3d = self.tensor.view((-1, nqx, nqy))

        qx = torch.arange(nqx, dtype=torch.float)
        qy = torch.arange(nqy, dtype=torch.float)
        qxa, qya = torch.meshgrid(qx, qy, indexing="ij")

        if max_batch_size is None:
            max_batch_size = self.num_dps

        batcher = SimpleTorchBatcher(
            torch.arange(self.num_dps), batch_size=max_batch_size, shuffle=False
        )

        com_measured = torch.empty((self.num_dps, 2), dtype=torch.float)

        for batch_idx in batcher:
            intensities = tensor_3d[batch_idx]
            summed_intensities = torch.sum(intensities, dim=(-2, -1))
            com_measured[batch_idx, 0] = (
                torch.sum(intensities * qxa[None, :, :], dim=(-2, -1))
                / summed_intensities
            )
            com_measured[batch_idx, 1] = (
                torch.sum(intensities * qya[None, :, :], dim=(-2, -1))
                / summed_intensities
            )

        self.origin_measured = com_measured
        return self

    @property
    def origin_measured(self) -> torch.Tensor:
        return self._origin_measured

    @origin_measured.setter
    def origin_measured(self, value: torch.Tensor):
        self._origin_measured = (
            ensure_valid_tensor(value, dtype=torch.float)
            .view((-1, 2))
            .expand((self.num_dps, 2))
        )

    def fit_origin_background(
        self,
        probe_positions: torch.Tensor | NDArray | None = None,
        fit_method: str | None = "plane",
    ):
        """ """

        if self._origin_measured is None:
            raise ValueError(
                "measured origins not detected. Use self.calculate_origin() first."
            )

        if probe_positions is None:
            if self.dataset.ndim != 4:
                raise ValueError(
                    "probe positions could not be inferred from dataset, please pass them explicitly."
                )

            nx, ny = self.dataset.shape[:2]

            x = torch.arange(nx, dtype=torch.float)
            y = torch.arange(ny, dtype=torch.float)
            xa, ya = torch.meshgrid(x, y, indexing="ij")
            probe_positions = torch.stack([xa, ya], -1).view((-1, 2))
        else:
            probe_positions = ensure_valid_tensor(probe_positions).view((-1, 2))
            if probe_positions.shape != self.origin_measured.shape:
                raise ValueError(
                    "probe positions shape must match the measured origins."
                )

        if fit_method == "plane":

            def fit_linear_plane(points: torch.Tensor):
                """ """
                # Covariance matrix
                centroid = points.mean(0)
                centered_points = points - centroid
                covariance_matrix = torch.cov(centered_points.T)
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

                # The normal vector to the plane is the eigenvector corresponding to the smallest eigenvalue
                normal_vector = eigenvectors[:, 0]
                a, b, c = normal_vector

                # Calculate d using the centroid: d = -(ax_c + by_c + cz_c)
                d = -torch.dot(normal_vector, centroid)
                return a, b, c, d

            com_x_pts = torch.concatenate(
                (probe_positions, self.origin_measured[:, 0, None]), 1
            )
            com_y_pts = torch.concatenate(
                (probe_positions, self.origin_measured[:, 1, None]), 1
            )

            ax, bx, cx, dx = fit_linear_plane(com_x_pts)
            ay, by, cy, dy = fit_linear_plane(com_y_pts)

            com_fitted_x = (probe_positions @ torch.tensor([-ax, -bx]) - dx) / cx
            com_fitted_y = (probe_positions @ torch.tensor([-ay, -by]) - dy) / cy
            com_fitted = torch.stack([com_fitted_x, com_fitted_y], -1)
        else:
            raise NotImplementedError("only fit_method='plane' is implemented for now.")

        self.origin_fitted = com_fitted
        return self

    @property
    def origin_fitted(self) -> torch.Tensor:
        return self._origin_fitted

    @origin_fitted.setter
    def origin_fitted(self, value: torch.Tensor):
        self._origin_fitted = (
            ensure_valid_tensor(value, dtype=torch.float)
            .view((-1, 2))
            .expand((self.num_dps, 2))
        )

    def shift_origin_to(
        self,
        origin_coordinate: Tuple[int | float, int | float] = (0, 0),
        max_batch_size: int | None = None,
        mode: str = "bicubic",
        padding_mode: str = "reflection",
    ):
        """ """

        if self._origin_fitted is None:
            raise ValueError(
                "fitted origins not detected. Use self.fit_origin_background() first."
            )

        origin_fitted = self.origin_fitted
        nqx, nqy = self.dataset.shape[-2:]

        tensor_3d = self.tensor.view((-1, 1, nqx, nqy))  # note channel dim
        shifted_tensor_3d = torch.empty_like(tensor_3d)
        coordinate = torch.as_tensor(origin_coordinate)

        # Create normalized meshgrid
        qxa, qya = torch.meshgrid(
            torch.linspace(-1, 1, nqx), torch.linspace(-1, 1, nqy), indexing="ij"
        )
        base_grid = torch.stack((qxa, qya), dim=-1).unsqueeze(0)  # (1, nqx, nqy, 2)

        if max_batch_size is None:
            max_batch_size = self.num_dps

        batcher = SimpleTorchBatcher(
            torch.arange(self.num_dps), batch_size=max_batch_size, shuffle=False
        )

        for batch_idx in batcher:
            intensities = tensor_3d[batch_idx]
            shifts = coordinate - origin_fitted[batch_idx]

            # Convert shifts from pixel units to normalized units
            shifts_norm = shifts.clone().view(-1, 1, 1, 2)
            shifts_norm[..., 0] *= 2 / nqx
            shifts_norm[..., 1] *= 2 / nqy

            grid = base_grid + shifts_norm

            shifted_tensor_3d[batch_idx] = F.grid_sample(
                intensities,
                grid,
                padding_mode=padding_mode,
                mode=mode,
                align_corners=True,
            )

        self._grid = grid
        self.shifted_tensor = shifted_tensor_3d.view(self.tensor.shape)

        return self

    @property
    def shifted_tensor(self) -> torch.Tensor:
        return self._shifted_tensor

    @shifted_tensor.setter
    def shifted_tensor(self, value: torch.Tensor):
        self._shifted_tensor = ensure_valid_tensor(value, dtype=torch.float)

    def forward(
        self,
        max_batch_size: int | None = None,
        fit_origin_bkg: bool = True,
        probe_positions: torch.Tensor | NDArray | None = None,
        fit_method: str | None = "plane",
        shift_to_origin: bool = True,
        origin_coordinate: Tuple[int | float, int | float] = (0, 0),
        mode: str = "bicubic",
        padding_mode: str = "reflection",
    ):
        """
        Runs the full Center-of-Mass origin alignment workflow.

        Args:
            max_batch_size: Maximum batch size to use during CoM calculation and shifting.
            fit_origin_bkg: Whether to fit a smooth background model to the measured origins.
            probe_positions: Probe scan positions (if not inferable from dataset).
            fit_method: Method to fit the origin background ("plane" supported).
            shift_to_origin: Whether to shift all patterns to a common origin.
            origin_coordinate: Target origin position in (qx, qy) coordinates.
            mode: Interpolation mode for shifting ("bicubic", "bilinear", etc.).
            padding_mode: How to handle boundaries during shifting.

        Returns:
            self
        """
        self.calculate_origin(max_batch_size)

        if fit_origin_bkg:
            self.fit_origin_background(probe_positions, fit_method)

            if shift_to_origin:
                self.shift_origin_to(
                    origin_coordinate, max_batch_size, mode, padding_mode
                )

        return self


# endregion --- CoM Origin Model ---
