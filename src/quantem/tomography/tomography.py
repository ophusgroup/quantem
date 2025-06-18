from typing import Literal

import torch

# from torch_radon.radon import ParallelBeam as Radon
from tqdm.auto import tqdm

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.tomography.object_models import ObjectVoxelwise
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_conv import TomographyConv
from quantem.tomography.tomography_ml import TomographyML
from quantem.tomography.utils import differentiable_shift_2d, gaussian_kernel_1d, rot_ZXZ


class Tomography(TomographyConv, TomographyML, TomographyBase):
    """
    Top level class for either using conventional or ML-based reconstruction methods
    for tomography.
    """

    def __init__(
        self,
        dataset,
        volume_obj,
        device,
        _token,
    ):
        super().__init__(dataset, volume_obj, device, _token)

    # --- Reconstruction Method ---

    def sirt_recon(
        self,
        num_iterations: int = 10,
        inline_alignment: bool = False,
        enforce_positivity: bool = True,
        volume_shape: tuple = None,
        reset: bool = True,
        smoothing_sigma: float = None,
        shrinkage: float = None,
        filter_name: str = "hamming",
        circle: bool = True,
    ):
        num_angles, num_rows, num_cols = self.dataset.tilt_series.shape

        if volume_shape is None:
            volume_shape = (num_rows, num_rows, num_rows)
        else:
            D, H, W = volume_shape

        if reset:
            volume = torch.zeros((D, H, W), device=self.device, dtype=torch.float32)
            self.loss = []
        else:
            volume = torch.tensor(
                self.volume_obj.array,
                device=self.device,
                dtype=torch.float32,
            )

        proj_forward = torch.zeros_like(self.dataset.tilt_series)

        pbar = tqdm(range(num_iterations), desc="SIRT Reconstruction")

        if smoothing_sigma is not None:
            gaussian_kernel = gaussian_kernel_1d(smoothing_sigma).to(self.device)
        else:
            gaussian_kernel = None

        for iter in pbar:
            if iter > 0 and inline_alignment:
                volume, proj_forward, loss = self._sirt_run_epoch(
                    volume=volume,
                    tilt_series=self.dataset.tilt_series,
                    proj_forward=proj_forward,
                    angles=self.dataset.tilt_angles,
                    inline_alignment=True,
                    enforce_positivity=enforce_positivity,
                    shrinkage=shrinkage,
                    gaussian_kernel=gaussian_kernel,
                    filter_name=filter_name,
                    circle=circle,
                )
            else:
                volume, proj_forward, loss = self._sirt_run_epoch(
                    volume=volume,
                    tilt_series=self.dataset.tilt_series,
                    proj_forward=proj_forward,
                    angles=self.dataset.tilt_angles,
                    inline_alignment=False,
                    enforce_positivity=enforce_positivity,
                    shrinkage=shrinkage,
                    gaussian_kernel=gaussian_kernel,
                    filter_name=filter_name,
                    circle=circle,
                )

            pbar.set_description(f"SIRT Reconstruction | Loss: {loss.item():.4f}")

            self.loss.append(loss.item())

        self.volume_obj = Dataset3d.from_array(
            array=volume.cpu().numpy(),
            # name=self.tilt_series.name,
            # origin=self.tilt_series.origin,
            # sampling=self.tilt_series.sampling,
            # units=self.tilt_series.units,
            # signal_units=self.tilt_series.signal_units,
        )

    def ad_recon(
        self,
        num_iter: int = 0,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        hard_constraints: dict = {},
        soft_constraints: dict = {},
        batch_size: int | None = None,
        store_iterations: bool | None = None,
        store_iterations_every: int | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        autograd: bool = True,
    ):
        # # Check if self.volume_obj is a ObjectModelType
        if not isinstance(self.volume_obj, ObjectVoxelwise):
            raise TypeError("volume_obj must be a ObjectVoxelwise")

        self.volume_obj.to(self.device)

        self.dataset.shifts = self.dataset.shifts.detach().to(self.device).requires_grad_(True)
        self.dataset.z1_angles = (
            self.dataset.z1_angles.detach().to(self.device).requires_grad_(True)
        )
        self.dataset.z3_angles = (
            self.dataset.z3_angles.detach().to(self.device).requires_grad_(True)
        )

        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            self.set_schedulers(self.scheduler_params, num_iter=num_iter)

        self.volume_obj.hard_constraints = hard_constraints
        self.volume_obj.soft_constraints = soft_constraints

        pbar = tqdm(range(num_iter), desc="AD Reconstruction")

        for a0 in pbar:
            loss = 0.0

            pred_volume = self.volume_obj.forward()

            for i in range(len(self.dataset.tilt_series)):
                forward_projection = self.projection_operator(
                    vol=pred_volume,
                    z1=self.dataset.z1_angles[i],
                    x=self.dataset.tilt_angles[i],
                    z3=self.dataset.z3_angles[i],
                    shift_x=self.dataset.shifts[i, 0],
                    shift_y=self.dataset.shifts[i, 1],
                    device=self.device,
                )

                loss += torch.nn.functional.mse_loss(
                    forward_projection, self.dataset.tilt_series[i]
                )
            loss /= len(self.dataset.tilt_series)

            loss += self.volume_obj.soft_loss
            loss.backward()

            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad()

            if self.schedulers is not None:
                for sch in self.schedulers.values():
                    if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sch.step(loss)
                    elif sch is not None:
                        sch.step()

            pbar.set_description(f"AD Reconstruction | Loss: {loss:.4f}")

        return self

    def projection_operator(
        self,
        vol,
        z1,
        x,
        z3,
        shift_x,
        shift_y,
        device,
    ):
        projection = (
            rot_ZXZ(
                mags=vol.unsqueeze(0),  # Add batch dimension
                z1=z1,
                x=x,
                z3=z3,
                device=device,
                mode="bilinear",
            )
            .squeeze()
            .sum(axis=0)
        )

        shifted_projection = differentiable_shift_2d(
            image=projection,
            shift_x=shift_x,
            shift_y=shift_y,
            sampling_rate=1.0,  # Assuming 1 pixel = 1 physical unit
        )

        return shifted_projection
