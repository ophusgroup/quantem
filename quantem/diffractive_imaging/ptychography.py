from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from tqdm import trange

from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.core.utils.utils import generate_batches
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_ml import PtychographyML
from quantem.diffractive_imaging.ptychography_visualizations import PtychographyVisualizations

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class Ptychography(PtychographyML, PtychographyVisualizations, PtychographyBase):
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """

    OPTIMIZABLE_VALS = ["object", "probe", "descan", "scan_positions"]
    DEFAULT_LRS = {
        "object": 5e-3,
        "probe": 1e-3,
        "descan": 1e-3,
        "scan_positions": 1e-3,
        "tv_weight_z": 0,
        "tv_weight_yx": 0,
    }
    DEFAULT_OPTIMIZER_TYPE = "adam"
    _token = object()

    def __init__(
        self,
        dset: Dataset4dstem,
        device: Literal["cpu", "gpu"] = "cpu",
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
        _token: None | object = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Dataset.from_array() to instantiate this class.")

        super().__init__(
            dset=dset,
            device=device,
            verbose=verbose,
            rng=rng,
            _token=self._token,
        )
        self._autograd = True

    # region --- explicit properties and setters ---

    @property
    def autograd(self) -> bool:
        return self._autograd

    @autograd.setter
    def autograd(self, autograd: bool) -> None:
        self._autograd = bool(autograd)

    # endregion --- explicit properties and setters ---

    # region --- implicit properties ---

    # endregion --- implicit properties ---

    # region --- methods ---

    def get_tv_loss(
        self, array: torch.Tensor, weights: None | tuple[float, float] = None
    ) -> torch.Tensor:
        """
        weight is tuple (weight_z, weight_yx) or float -> (weight, weight)
        for 2D array, only weight_yx is used

        """
        loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)
        if weights is None:
            w = (
                self.constraints["object"]["tv_weight_z"],
                self.constraints["object"]["tv_weight_yx"],
            )
        elif isinstance(weights, (float, int)):
            if weights == 0:
                return loss
            w = (weights, weights)
        else:
            if not any(weights):
                return loss
            if len(weights) != 2:
                raise ValueError(f"weights must be a tuple of length 2, got {weights}")
            w = weights

        if array.is_complex():
            ph = array.angle()
            loss += self._calc_tv_loss(ph, w)
            amp = array.abs()
            if torch.max(amp) - torch.min(amp) > 1e-3:  # is complex and not pure_phase
                loss += self._calc_tv_loss(amp, w)
        else:
            loss += self._calc_tv_loss(array, w)

        return loss

    def _calc_tv_loss(self, array: torch.Tensor, weight: tuple[float, float]) -> torch.Tensor:
        loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)
        for dim in range(array.ndim):
            if dim == 0 and array.ndim == 3:  # there's surely a cleaner way but whatev
                w = weight[0]
            else:
                w = weight[1]
            loss += w * torch.mean(torch.abs(array.diff(dim=dim)))
        loss /= array.ndim
        return loss

    def reset_recon(self) -> None:
        super().reset_recon()
        self._optimizers = {}
        self._schedulers = {}

    def _record_lrs(self) -> None:
        optimizers = self.optimizers
        all_keys = set(self._epoch_lrs.keys()) | set(optimizers.keys())
        for key in all_keys:
            if key in self._epoch_lrs.keys():
                if key in self.optimizers.keys():
                    self._epoch_lrs[key].append(self.optimizers[key].param_groups[0]["lr"])
                else:
                    self._epoch_lrs[key].append(0.0)
            else:  # new optimizer
                prev_lrs = [0.0] * self.num_epochs
                # prev_lrs = [0.0] * (self.num_epochs - 1)
                prev_lrs.append(self.optimizers[key].param_groups[0]["lr"])
                self._epoch_lrs[key] = prev_lrs

    # endregion --- methods ---

    # region --- reconstruction ---

    def reconstruct(
        self,
        num_iter: int = 0,
        reset: bool = False,
        optimizer_params: dict | None = None,
        obj_type: Literal["complex", "pure_phase", "potential"] | None = None,
        scheduler_params: dict | None = None,
        constraints: dict = {},
        batch_size: int | None = None,
        store_iterations: bool | None = None,
        store_iterations_every: int | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        autograd: bool = True,
    ) -> Self:
        """
        reason for having a single reconstruct() is so that updating things like constraints
        or recon_types only happens in one place, reason for having separate reoconstruction_
        methods would be to simplify the flags for this and not have to include all

        """
        # TODO maybe make an "process args" method that handles things like:
        # mode, store_iterations, device,
        self._check_preprocessed()
        self.device = device
        batch_size = self.gpts[0] * self.gpts[1] if batch_size is None else batch_size
        self.store_iterations = store_iterations
        self.store_iterations_every = store_iterations_every
        self.set_obj_type(obj_type, force=reset)
        if reset:
            self.reset_recon()
        self.constraints = constraints
        self._move_recon_arrays_to_device()

        new_scheduler = reset
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()
            new_scheduler = True

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            new_scheduler = True

        if new_scheduler:
            self.set_schedulers(self.scheduler_params, num_iter=num_iter)

        if "descan" in self.optimizer_params.keys():
            target_amplitudes = self._amplitudes
        else:
            target_amplitudes = self._shifted_amplitudes

        shuffled_indices = np.arange(self.gpts[0] * self.gpts[1])

        # TODO add pbar with loss printout
        for a0 in trange(num_iter, disable=not self.verbose):
            self.rng.shuffle(shuffled_indices)
            epoch_loss = 0.0

            for start, end in generate_batches(
                num_items=self.gpts[0] * self.gpts[1], max_batch=batch_size
            ):
                loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)
                batch_indices = shuffled_indices[start:end]
                descan_shifts = (
                    self._descan_shifts[batch_indices]
                    if "descan" in self.optimizer_params.keys()
                    else None
                )
                shifted_probes = self.probe_model.forward(
                    self._positions_px_fractional[batch_indices]
                )
                obj_patches = self.obj_model.forward(self._patch_indices[batch_indices])
                propagated_probes, overlap = self.forward_operator(
                    obj_patches, shifted_probes, descan_shifts
                )

                loss += (
                    self.error_estimate(overlap, target_amplitudes[batch_indices])
                    / self._mean_diffraction_intensity
                )

                if (
                    self.constraints["object"]["tv_weight_z"] > 0
                    or self.constraints["object"]["tv_weight_yx"] > 0
                ):
                    # TODO change this to a loss += self.obj_model.soft_constraints()
                    # likewise for probe (and descan, so make a loss += self.soft_constraints())
                    # that calls each
                    loss += self.get_tv_loss(self.obj_model.obj) * (end - start)

                self.backward(
                    loss,
                    autograd,
                    obj_patches,
                    propagated_probes,
                    overlap,
                    self._patch_indices[batch_indices],  # replace with just passing indices
                    target_amplitudes[batch_indices],
                )
                for opt in self.optimizers.values():
                    opt.step()
                    opt.zero_grad()

                # additional constraints
                # data constraints
                # detector constraints
                # with torch.no_grad():
                #     self._descan_shifts = torch.ones_like(self._descan_shifts) * self._descan_shifts.mean(dim=0, keepdim=True)

                epoch_loss += loss.item()

            for sch in self.schedulers.values():
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(epoch_loss)
                elif sch is not None:
                    sch.step()

            self._record_lrs()
            self._epoch_losses.append(epoch_loss)
            self._epoch_recon_types.append(f"{self.obj_model.name}-{self.probe_model.name}")
            if self.store_iterations and ((a0 + 1) % self.store_iterations_every == 0 or a0 == 0):
                self.append_recon_iteration(self.obj, self.probe)

        return self

    def backward(
        self,
        loss: torch.Tensor,
        autograd: bool,
        obj_patches: torch.Tensor,
        propagated_probes: torch.Tensor,
        overlap: torch.Tensor,
        patch_indices: torch.Tensor,
        amplitudes: torch.Tensor,
    ):
        """ """
        if autograd:
            loss.backward()
        else:
            gradient = self.gradient_step(amplitudes, overlap)
            prop_gradient = self.obj_model.backward(
                gradient,
                obj_patches,
                propagated_probes,
                self._propagators,
                patch_indices,
            )
            self.probe_model.backward(prop_gradient, obj_patches)
            # TODO -- connect other optimizable values to this, e.g. descan

    def gradient_step(self, amplitudes, overlap):
        """Computes analytical gradient using the Fourier projection modified overlap"""
        modified_overlap = self.fourier_projection(amplitudes, overlap)
        ## mod_overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        ## grad shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        return modified_overlap - overlap

    def fourier_projection(self, measured_amplitudes, overlap_array):
        """Replaces the Fourier amplitude of overlap with the measured data."""
        # corner centering measured amplitudes
        measured_amplitudes = torch.fft.fftshift(measured_amplitudes, dim=(-2, -1))
        fourier_overlap = torch.fft.fft2(overlap_array)
        # from quantem.core.visualization.visualization import show_2d
        # show_2d([fourier_overlap[0,0], torch.abs(fourier_overlap[0,0])])
        if self.num_probes == 1:  # faster
            fourier_modified_overlap = measured_amplitudes * torch.exp(
                1.0j * torch.angle(fourier_overlap)
            )
        else:  # necessary for mixed state
            farfield_amplitudes = self.estimate_amplitudes(overlap_array, corner_centered=True)
            farfield_amplitudes[farfield_amplitudes == 0] = torch.inf
            amplitude_modification = measured_amplitudes / farfield_amplitudes
            fourier_modified_overlap = amplitude_modification * fourier_overlap

        return torch.fft.ifft2(fourier_modified_overlap)

    # endregion --- reconstruction ---
