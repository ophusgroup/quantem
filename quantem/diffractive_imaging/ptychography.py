from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from tqdm import trange

from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.core.utils.utils import generate_batches
from quantem.diffractive_imaging.ptycho_utils import fourier_translation_operator
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_gd import PtychographyGD
from quantem.diffractive_imaging.ptychography_ml import PtychographyML
from quantem.diffractive_imaging.ptychography_visualizations import PtychographyVisualizations

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class Ptychography(PtychographyML, PtychographyGD, PtychographyVisualizations, PtychographyBase):
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
        self, arrayay: torch.Tensor, weights: None | tuple[float, float] = None
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

        if arrayay.is_complex():
            ph = arrayay.angle()
            loss += self._calc_tv_loss(ph, w)
            amp = arrayay.abs()
            if torch.max(amp) - torch.min(amp) > 1e-3:  # is complex and not pure_phase
                loss += self._calc_tv_loss(amp, w)
        else:
            loss += self._calc_tv_loss(arrayay, w)

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
            target_amplitudes = self._corner_amplitudes
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
                shifted_probes = self.probe_model.forward(
                    self._positions_px_fractional[batch_indices]
                )
                obj_patches = self.obj_model.forward(self._patch_indices[batch_indices])
                _propagated_probes, overlap = self.forward_operator(
                    obj_patches,
                    shifted_probes,
                )

                if "descan" in self.optimizer_params.keys():
                    shifts = fourier_translation_operator(
                        self._descan_shifts[batch_indices], tuple(self.roi_shape), True
                    )
                    overlap *= shifts[None]

                loss += (
                    self.error_estimate(
                        overlap,
                        target_amplitudes[batch_indices],
                        # self._corner_amplitudes[batch_indices],
                        # self._shifted_amplitudes[batch_indices],
                    )
                    / self._mean_diffraction_intensity
                )

                if (
                    self.constraints["object"]["tv_weight_z"] > 0
                    or self.constraints["object"]["tv_weight_yx"] > 0
                ):
                    loss += self.get_tv_loss(self.obj_model.obj) * (end - start)

                # TODO make a backwards method, does either autograd or sets analytic gradients
                loss.backward()
                for opt in self.optimizers.values():
                    opt.step()
                    opt.zero_grad()

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

        # TODO -- method for update_shifted_amplitudes, shift them based on com_shifts
        # if learn_descan

        return self

    # endregion --- reconstruction ---


# def sudocode():

#     for epoch in epochs:
#         indices = ptycho.get_patches(positions_px)
#         for batch in batches:
#             batch_inds = indices[batch]
#             shifted_probes = probe.forward(frac_positions[batch_inds])
#             obj_patches = object.forward(indices[batch_inds])
#             exit_waves = ptycho.forward(shifted_probes, obj_patches)
#             loss = ptycho.loss(exit_waves, shifted_probes)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
