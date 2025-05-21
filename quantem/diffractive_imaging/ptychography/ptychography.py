import math
from typing import Sequence, Tuple

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
        optimizer_type: Sequence[str] = ["sgd", "sgd"],
        lr: Sequence[float] = [0.5, 0.5],
    ):
        self.object = object_model
        self.probe = probe_model
        self.intensity_data = intensity_data
        self.positions_px = positions_px
        self.use_autograd = use_autograd

        self.set_optimized_parameters(optimize_obj, optimize_probe)
        self.set_optimizers(optimizer_type, lr)

        self.roi_shape = self.probe.dataset.shape
        self.obj_shape = self.object.dataset.shape
        self.row, self.col = self.object.return_patch_indices(
            positions_px=self.positions_px,
            roi_shape=self.roi_shape,
            obj_shape=self.obj_shape,
        )

    def set_optimized_parameters(self, optimize_obj: bool, optimize_probe: bool):
        """ """
        if optimize_obj is not None:
            self.object.tensor.requires_grad = optimize_obj
        if optimize_probe is not None:
            self.probe.tensor.requires_grad = optimize_probe

        self.parameters = []
        if self.object.tensor.requires_grad:
            self.parameters.append(self.object.tensor)
        if self.probe.tensor.requires_grad:
            self.parameters.append(self.probe.tensor)

    def set_optimizers(
        self,
        optimizer_types: Sequence[str],
        lrs: Sequence[float],
    ):
        """ """

        optimizers = []
        for param, optimizer_type, lr in zip(self.parameters, optimizer_types, lrs):
            if optimizer_type == "adam":
                optimizer = torch.optim.Adam(params=[param], lr=lr)
            elif optimizer_type == "sgd":
                optimizer = torch.optim.SGD(params=[param], lr=lr)
            elif optimizer_type == "cg":
                optimizer = PolakRibiereCG(
                    params=[param],
                    lr=lr,
                    line_search=True,
                )
            else:
                raise ValueError(f"optimizer_type: {optimizer_type} is not recognized")
            optimizers.append(optimizer)
        self.optimizers = optimizers

    def forward(
        self,
        batch_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        pos = self.positions_px[batch_idx]
        frac_pos = pos - torch.round(pos)
        shifted_probes = self.probe.forward(frac_pos)
        row = self.row[batch_idx]
        col = self.col[batch_idx]
        obj_patches, exit_waves = self.object.forward(
            shifted_probes,
            row,
            col,
        )
        fourier_exit_waves = torch.fft.fft2(exit_waves, norm="ortho")
        simulated_intensities = torch.square(torch.abs(fourier_exit_waves))
        return (
            shifted_probes,
            obj_patches,
            exit_waves,
            fourier_exit_waves,
            simulated_intensities,
        )

    def backward(
        self,
        batch_idx: torch.Tensor,
        shifted_probes: torch.Tensor,
        obj_patches: torch.Tensor,
        exit_waves: torch.Tensor,
        fourier_exit_waves: torch.Tensor,
        simulated_intensities: torch.Tensor,
    ) -> torch.Tensor:
        """ """

        with torch.set_grad_enabled(self.use_autograd):
            measured_intensities = self.intensity_data[batch_idx]
            loss = (
                torch.mean(
                    torch.sum(
                        torch.square(
                            torch.sqrt(simulated_intensities + 1e-10)
                            - torch.sqrt(measured_intensities + 1e-10)
                        ),
                        dim=(-1, -2),
                    )
                )
                / 2.0
            )

            if self.use_autograd:
                loss.backward()
            else:
                gradient = torch.fft.ifft2(
                    fourier_exit_waves
                    - torch.sqrt(measured_intensities)
                    * torch.exp(1j * torch.angle(fourier_exit_waves)),
                    norm="ortho",
                )

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
        optimize_obj: bool | None = None,
        optimize_probe: bool | None = None,
        optimizer_type: Sequence[str] | None = None,
        lr: Sequence[float] = [1e-2, 1e-3],
    ):
        """ """
        if use_autograd is not None:
            self.use_autograd = use_autograd

        self.set_optimized_parameters(optimize_obj, optimize_probe)
        if optimizer_type is not None:
            self.set_optimizers(optimizer_type, lr)

        indices = torch.arange(
            self.intensity_data.shape[0], device=self.intensity_data.device
        ).to(torch.long)
        batcher = SimpleBatcher(indices, batch_size, shuffle=True)

        for epoch in range(n_epochs):
            total_loss = 0.0

            for batch_idx in batcher:
                # prevent gradient accumulation across batches
                for opt in self.optimizers:
                    opt.zero_grad()

                def closure():
                    """ """
                    with torch.set_grad_enabled(self.use_autograd):
                        # Forward
                        (
                            shifted_probes,
                            obj_patches,
                            exit_waves,
                            fourier_exit_waves,
                            simulated_intensities,
                        ) = self.forward(batch_idx)

                        # Backward
                        loss = self.backward(
                            batch_idx,
                            shifted_probes,
                            obj_patches,
                            exit_waves,
                            fourier_exit_waves,
                            simulated_intensities,
                        )
                    return loss

                # Optimizers Step
                for opt in self.optimizers:
                    loss = opt.step(closure)

                total_loss += loss.item()

                # Constraints
                with torch.no_grad():
                    obj = self.object.tensor
                    amp = torch.clamp(torch.abs(obj), max=1.0)
                    phase = torch.angle(obj)
                    self.object.tensor.data = amp * torch.exp(1j * phase)

            print(f"[Epoch {epoch + 1:02d}] Loss: {total_loss:.4e}")

            if not math.isfinite(total_loss):
                break

        return self
