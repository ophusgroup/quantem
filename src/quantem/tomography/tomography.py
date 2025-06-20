import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from torch_radon.radon import ParallelBeam as Radon
from tqdm.auto import tqdm

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

        self._log_dir = None
        self._logger = None

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

        hard_constraints = {
            "positivity": enforce_positivity,
            "shrinkage": shrinkage,
        }
        self.volume_obj.hard_constraints = hard_constraints

        if volume_shape is None:
            volume_shape = (num_rows, num_rows, num_rows)
        else:
            D, H, W = volume_shape

        if reset:
            self.volume_obj.reset()
            self.loss = []

        proj_forward = torch.zeros_like(self.dataset.tilt_series)

        pbar = tqdm(range(num_iterations), desc="SIRT Reconstruction")

        if smoothing_sigma is not None:
            gaussian_kernel = gaussian_kernel_1d(smoothing_sigma).to(self.device)
        else:
            gaussian_kernel = None

        for iter in pbar:
            proj_forward, loss = self._sirt_run_epoch(
                tilt_series=self.dataset.tilt_series,
                proj_forward=proj_forward,
                angles=self.dataset.tilt_angles,
                inline_alignment=iter > 0 and inline_alignment,
                filter_name=filter_name,
                gaussian_kernel=gaussian_kernel,
                circle=circle,
            )

            pbar.set_description(f"SIRT Reconstruction | Loss: {loss.item():.4f}")

            self.loss.append(loss.item())

        self.sirt_recon_vol = self.volume_obj

    def ad_recon(
        self,
        optimizer_params: dict,
        num_iter: int = 0,
        reset: bool = False,
        scheduler_params: dict | None = None,
        hard_constraints: dict | None = None,
        soft_constraints: dict | None = None,
        logging: bool = False,
        log_images_every: int = 10,
        logger_cmap: str = "turbo",
        # store_iterations: bool | None = None,
        # store_iterations_every: int | None = None,
        # autograd: bool = True,
    ):
        if logging and self.logger is None:
            print("Initializing logger")

            self.init_logger()

        if reset:
            self.reset_recon()

        self.hard_constraints = hard_constraints
        self.soft_constraints = soft_constraints

        # Make sure everything is in the correct device, might be redundant/cleaner way to do this
        self.dataset.to(self.device)
        self.volume_obj.to(self.device)

        # Making optimizable parameters into leaf tensors.
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

        if hard_constraints is not None:
            self.volume_obj.hard_constraints = hard_constraints
        if soft_constraints is not None:
            self.volume_obj.soft_constraints = soft_constraints

        pbar = tqdm(range(num_iter), desc="AD Reconstruction")

        for a0 in pbar:
            total_loss = 0.0
            tilt_series_loss = 0.0

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

                tilt_series_loss += torch.nn.functional.mse_loss(
                    forward_projection, self.dataset.tilt_series[i]
                )
            tilt_series_loss /= len(self.dataset.tilt_series)

            total_loss = tilt_series_loss + self.volume_obj.soft_loss
            self.loss.append(total_loss.item())

            total_loss.backward()

            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad()

            if self.schedulers is not None:
                for sch in self.schedulers.values():
                    if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sch.step(total_loss)
                    elif sch is not None:
                        sch.step()

            pbar.set_description(f"AD Reconstruction | Loss: {total_loss:.4f}")

            if self.logger is not None:
                self.logger.add_scalar("loss/total", total_loss.item(), self.epochs)
                self.logger.add_scalar("loss/tilt_series", tilt_series_loss.item(), self.epochs)
                self.logger.add_scalar(
                    "loss/soft constraints", self.volume_obj.soft_loss.item(), self.epochs
                )

                if self.epochs % log_images_every == 0:
                    sum_0 = self.volume_obj.obj.sum(axis=0)
                    sum_1 = self.volume_obj.obj.sum(axis=1)
                    sum_2 = self.volume_obj.obj.sum(axis=2)
                    self.logger.add_image(
                        "projections/Y-X Projection",
                        self.apply_colormap(sum_0, logger_cmap),
                        self.epochs,
                    )
                    self.logger.add_image(
                        "projections/Z-X Projection",
                        self.apply_colormap(sum_1, logger_cmap),
                        self.epochs,
                    )
                    self.logger.add_image(
                        "projections/Z-Y Projection",
                        self.apply_colormap(sum_2, logger_cmap),
                        self.epochs,
                    )

                    z1_fig, x_fig, z3_fig = self.fig_tilt_angles()
                    self.logger.add_figure("tilt_angles/z1", z1_fig, self.epochs)
                    self.logger.add_figure("tilt_angles/x", x_fig, self.epochs)
                    self.logger.add_figure("tilt_angles/z3", z3_fig, self.epochs)

        self.ad_recon_vol = self.volume_obj.forward()

    def reset_recon(self) -> None:
        if isinstance(self.volume_obj, ObjectVoxelwise):
            self.volume_obj.reset()

        self.ad_recon_vol = None

    # --- Projection Operators ----
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

    # --- Tensorboard Logging ---

    @property
    def log_dir(self) -> str:
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir: str):
        if not Path(log_dir).exists():
            raise FileNotFoundError(f"log_dir {log_dir} does not exist, make the directory first")

        self._log_dir = log_dir

    def make_logdir(self, log_dir: str):
        curr_run = f"{log_dir}/tomo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not Path(log_dir).exists():
            Path(curr_run).mkdir(parents=True, exist_ok=True)

        self._log_dir = Path(curr_run)

    @property
    def logger(self) -> SummaryWriter:
        return self._logger

    def init_logger(self):
        if self.log_dir is None:
            raise ValueError("log_dir is not set")

        if self._logger is not None:
            raise RuntimeError("Logger already initialized")

        self._logger = SummaryWriter(self.log_dir)

    def close_logger(self):
        self._logger.flush()
        self._logger.close()
        self._logger = None

    def fig_tilt_angles(self):
        fig_z1, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.dataset.z1_angles.detach().cpu().numpy())
        ax.set_title("Z1 Angles")
        ax.set_xlabel("Index")
        ax.set_ylabel("Angle")

        fig_x, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.dataset.tilt_angles.detach().cpu().numpy())
        ax.set_title("Tilt/X Angles")
        ax.set_xlabel("Index")
        ax.set_ylabel("Angle")

        fig_z3, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.dataset.z3_angles.detach().cpu().numpy())
        ax.set_title("Z3 Angles")
        ax.set_xlabel("Index")
        ax.set_ylabel("Angle")

        return fig_z1, fig_x, fig_z3

    def apply_colormap(self, tensor_2d, cmap_name="turbo"):
        """
        Apply Turbo colormap to a 2D PyTorch tensor. Output: [3, H, W] NumPy float32 in [0,1].
        """
        if isinstance(tensor_2d, torch.Tensor):
            tensor_2d = tensor_2d.detach().cpu().numpy()

        tensor_2d = tensor_2d.astype(np.float32)
        tensor_2d = (tensor_2d - np.min(tensor_2d)) / (np.ptp(tensor_2d) + 1e-8)

        cmap = plt.get_cmap(cmap_name)
        colored = cmap(tensor_2d)[..., :3]  # Shape: [H, W, 3]
        colored = colored.transpose(2, 0, 1)  # → [3, H, W]

        return colored.astype(np.float32)
