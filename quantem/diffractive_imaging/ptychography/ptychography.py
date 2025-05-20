import math
from typing import Tuple

import torch

from quantem.diffractive_imaging.ptychography.data_loaders import SimpleBatcher
from quantem.diffractive_imaging.ptychography.object_models import ObjectModelBase
from quantem.diffractive_imaging.ptychography.optimizers import PolakRibiereCG
from quantem.diffractive_imaging.ptychography.probe_models import ProbeModelBase


class PtychographicReconstruction:
    """ """

    def __init__(
        self,
        object_model: ObjectModelBase,
        probe_model: ProbeModelBase,
        positions_px: torch.Tensor,
        intensity_data: torch.Tensor,
        optimize_obj: bool = True,
        optimize_probe: bool = True,
        use_autograd: bool = False,
        optimizer_type: str = "sgd",
        lr: float = 1e-1,
    ):
        self.object = object_model
        self.probe = probe_model
        self.probe.tensor.requires_grad = optimize_probe
        self.object.tensor.requires_grad = optimize_obj

        self.intensity_data = intensity_data
        self.positions_px = positions_px
        self.use_autograd = use_autograd

        self.parameters = []
        if optimize_obj:
            self.parameters.append(self.object.tensor)
        if optimize_probe:
            self.parameters.append(self.probe.tensor)

        self.set_optimizer(optimizer_type, lr)

        self.roi_shape = self.probe.dataset.shape
        self.obj_shape = self.object.dataset.shape
        self.row, self.col = self.object.return_patch_indices(
            positions_px=self.positions_px,
            roi_shape=self.roi_shape,
            obj_shape=self.obj_shape,
        )

    def set_optimizer(
        self,
        optimizer_type,
        lr,
    ):
        """ """
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(params=self.parameters, lr=lr)
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(params=self.parameters, lr=lr)
        elif optimizer_type == "cg":
            optimizer = PolakRibiereCG(
                params=self.parameters,
                lr=lr,
                line_search=True,
            )
        else:
            raise ValueError(f"optimizer_type: {optimizer_type} is not recognized")
        self.optimizer = optimizer

    def forward(
        self,
        batch_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        pos = self.positions_px[batch_idx]
        frac_pos = pos - torch.round(pos)
        shifted_probes = self.probe.forward(frac_pos)
        obj_patches, exit_waves = self.object.forward(
            shifted_probes,
            self.row[batch_idx],
            self.col[batch_idx],
        )
        return shifted_probes, obj_patches, exit_waves

    def backward(
        self,
        batch_idx: torch.Tensor,
        shifted_probes: torch.Tensor,
        obj_patches: torch.Tensor,
        exit_waves: torch.Tensor,
    ) -> torch.Tensor:
        """ """

        with torch.set_grad_enabled(self.use_autograd):
            fourier_exit_waves = torch.fft.fft2(exit_waves)
            simulated_intensities = torch.square(torch.abs(fourier_exit_waves))
            measured_intensities = self.intensity_data[batch_idx]

            numel = simulated_intensities[0].numel()
            loss = (
                0.5
                * torch.sum(torch.square(simulated_intensities - measured_intensities))
                / numel
                / 2.0
            )
            if self.use_autograd:
                loss.backward()
            else:
                gradient = torch.fft.ifft2(
                    fourier_exit_waves * (simulated_intensities - measured_intensities),
                )

                # gradient = torch.fft.ifft2(
                #     fourier_exit_waves - torch.sqrt(measured_intensities) * torch.exp(1j*torch.angle(fourier_exit_waves))
                # )

                self.object.backward(
                    gradient, shifted_probes, self.positions_px[batch_idx]
                )

                self.probe.backward(
                    gradient,
                    obj_patches,
                )

        return loss

    def reconstruct(
        self,
        n_epochs: int,
        batch_size: int,
        use_autograd: bool | None = None,
        optimizer_type: str | None = None,
        lr: float = 1e-2,
    ):
        """ """
        if use_autograd is not None:
            self.use_autograd = use_autograd
        if optimizer_type is not None:
            self.set_optimizer(optimizer_type, lr)

        indices = torch.arange(
            self.intensity_data.shape[0], device=self.intensity_data.device
        ).to(torch.long)
        batcher = SimpleBatcher(indices, batch_size, shuffle=True)

        for epoch in range(n_epochs):
            total_loss = 0.0

            for batch_idx in batcher:
                self.optimizer.zero_grad()

                # Forward
                shifted_probes, obj_patches, exit_waves = self.forward(batch_idx)

                # Backward
                loss = self.backward(batch_idx, shifted_probes, obj_patches, exit_waves)
                total_loss += loss.item()

                # Optimizer Step
                self.optimizer.step()

            print(f"[Epoch {epoch + 1:02d}] Loss: {total_loss:.4e}")

            if not math.isfinite(total_loss):
                break
