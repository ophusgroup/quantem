from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
from tqdm import trange

from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging.ptycho_utils import generate_batches
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_constraints import PtychographyConstraints

if TYPE_CHECKING:
    import cupy as cp
    import torch
else:
    if config.get("has_torch"):
        import torch
    if config.get("has_cupy"):
        import cupy as cp


class PtychographyML(PtychographyConstraints, PtychographyBase):
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """

    OPTIMIZABLE_VALS = ["object", "probe", "descan", "positions"]
    DEFAULT_LRS = {
        "object": 5e-3,
        "probe": 1e-3,
        "descan": 1e-3,
        "positions": 1e-3,  # TODO change to scan_positions
        "tv_weight_z": 0,
        "tv_weight_yx": 0,
    }
    DEFAULT_OPTIMIZER_TYPE = "adam"

    def __init__(
        self,
        dset: Dataset4dstem,
        object_type: Literal["complex", "pure_phase", "potential"] = "complex",
        num_probes: int = 1,
        num_slices: int = 1,
        slice_thicknesses: float | None = None,
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        if not config.get("has_torch"):
            raise RuntimeError("PtychographyML requires torch to be installed.")

        super().__init__(
            dset=dset,
            object_type=object_type,
            num_probes=num_probes,
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            verbose=verbose,
            rng=rng,
        )

        self._object_padding_force_power2_level = 0
        self._schedulers = {}
        self._optimizers = {}
        self._scheduler_params = {}
        self._optimizer_params = {}
        # self._base_lrs = self.DEFAULT_LRS
        self._model_object_input = None
        self._model_probe_input = None

    # region --- explicit properties and setters ---

    @property
    def mode(self) -> Literal["model", "pixelwise"]:
        # saying model for generative model of some sort, doesn't have to be a DIP
        return self._mode

    @property
    def optimizer_params(self) -> dict[str, dict]:
        """Returns the parameters used to set the optimizers."""
        return self._optimizer_params

    @optimizer_params.setter
    def optimizer_params(self, d: dict) -> None:
        """
        # Takes a dictionary {key: torch.optim.Adam(params=[blah], lr=[blah]), ...}
        Takes a dictionary:
        {
            "key1": {
                "type": "adam",
                "lr": 0.001,
                },
            "key2": {
                ...
                },
            ...
        }
        """
        # resets _optimizers as well
        self._optimizers = {}
        self._optimizer_params = {}
        if isinstance(d, (tuple, list)):
            d = {k: {} for k in d}

        for k, v in d.items():
            if k not in self.OPTIMIZABLE_VALS:
                raise ValueError(
                    f"key to be optimized, {k}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
                )
            if "type" not in v.keys():
                v["type"] = self.DEFAULT_OPTIMIZER_TYPE
            if "lr" not in v.keys():
                v["lr"] = self.DEFAULT_LRS[k]
            self._optimizer_params[k] = v

    @mode.setter
    def mode(self, mode: Literal["model", "pixelwise"]):
        if mode not in ["model", "pixelwise"]:
            raise ValueError(f"mode must be one of ['model', 'pixelwise'], got {mode}")
        self._mode: Literal["model", "pixelwise"] = mode

    @property
    def optimizers(self) -> dict[str, torch.optim.Adam | torch.optim.AdamW]:
        return self._optimizers

    def remove_optimizer(self, key: str) -> None:
        self._optimizers.pop(key, None)
        self._optimizer_params.pop(key, None)
        return

    def _add_optimizer(self, key: str, params: "torch.Tensor|Sequence[torch.Tensor]"):
        if key not in self.OPTIMIZABLE_VALS:
            raise ValueError(
                f"key to be optimized, {key}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
            )
        if isinstance(params, torch.Tensor):
            params = [params]
        [p.requires_grad_(True) for p in params]
        opt_params = self.optimizer_params[key]
        opt_type = opt_params.pop("type")
        if opt_type == "adam":
            opt = torch.optim.Adam(params, **opt_params)
        elif opt_type == "adamw":
            opt = torch.optim.AdamW(params, **opt_params)  # TODO pass all other kwargs
        else:
            raise NotImplementedError(f"Unknown optimizer type: {opt_params['type']}")
        opt_params["type"] = opt_type  # replacing opt type
        # if key in self.optimizers.keys():
        #     self.vprint(f"Key {key} is already in optimizers, overwriting.")
        self._optimizers[key] = opt

    @property
    def scheduler_params(self) -> dict[str, dict]:
        """Returns the parameters used to set the schedulers."""
        return self._scheduler_params

    @scheduler_params.setter
    def scheduler_params(self, d: dict) -> None:
        """
        Takes a dictionary:
        {
            "key1": {
                "type": "cyclic",
                "base_lr": 0.001,
                },
            "key2": {
                ...
                },
            ...
        }
        """
        for k, v in d.items():
            if k not in self.OPTIMIZABLE_VALS:
                raise ValueError(
                    f"key to be optimized, {k}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
                )
            if v["type"] not in ["cyclic", "plateau", "exp", "gamma", "none"]:
                raise ValueError(
                    f"Unknown scheduler type: {v['type']}, expected one of ['cyclic', 'plateau', 'exp', 'gamma', 'none']"
                )
        self._scheduler_params = d

    @property
    def schedulers(
        self,
    ) -> dict[
        str,
        (
            torch.optim.lr_scheduler.CyclicLR
            | torch.optim.lr_scheduler.ReduceLROnPlateau
            | torch.optim.lr_scheduler.ExponentialLR
            | None
        ),
    ]:
        return self._schedulers

    def set_schedulers(
        self,
        params: dict[str, dict],
        num_iter: int | None = None,
    ):
        """
        TODO allow for new schedulers to be passed in when adding new optimizers without
        removing the old schedulers or overwrtiting them. Not entirely sure what usecases there
        will be for this.

        Sets the schedulers for the optimizer from a dictionary. Expects a dictionary of the form:
        {
            "optimizable_key1": {
                "type": "scheduler_type",
                "scheduler_kwarg": scheduler_kwarg_value,
                ...
            },
            "optimizable_key2": {
                "type": "scheduler_type",
                "scheduler_kwarg": scheduler_kwarg_value,
                ...
            },
            ...
        }
        where the keys are the same as the keys in self.OPTIMIZABLE_VALS.

        The scheduler type can be one of the following:
        - "cyclic"
        - "plateau" or "reducelronplateau"
        - "exponential"
        - None

        The num_iter kwarg is only used for exponential schedulers and if a "factor" is given
        as a scheduler_kwarg instead of gamma. In that case, the gamma is calculated from num_iter
        and the factor.

        TODO could update this to allow passing key:optimizer directly, would likely need to
        rewrite get_schedulers to check the tpye
        """
        if not any(self.optimizers):
            raise NameError("self.optimizers have not yet been set.")
        self._schedulers = self._get_schedulers(
            params=params,
            optimizers=self.optimizers,
            num_iter=num_iter,
        )

    def _get_schedulers(
        self,
        params: dict[str, dict],
        optimizers: dict,
        num_iter: int | None = None,
    ) -> dict[
        str,
        (
            torch.optim.lr_scheduler.CyclicLR
            | torch.optim.lr_scheduler.ReduceLROnPlateau
            | torch.optim.lr_scheduler.ExponentialLR
            | None
        ),
    ]:
        """
        return schedulers for a given set of optimizers. Kept seperate from schedulers.setter so
        that it can be called for pre-training
        """
        schedulers = {}
        for opt_key, p in params.items():
            if opt_key not in self.OPTIMIZABLE_VALS:
                raise KeyError(
                    f"Scheduler got bad key {opt_key}, schedulers can only be attached to one of {self.OPTIMIZABLE_VALS}"
                )
            elif opt_key not in optimizers.keys():
                raise KeyError(f"optimizers does not have an optimizer for: {opt_key}")
            else:
                schedulers[opt_key] = self._get_scheduler(
                    optimizer=optimizers[opt_key], params=p, num_iter=num_iter
                )
        return schedulers

    def _get_scheduler(
        self,
        optimizer: torch.optim.Adam,
        params: dict[str, Any] | torch.optim.lr_scheduler._LRScheduler,
        num_iter: int | None = None,
    ) -> (
        torch.optim.lr_scheduler._LRScheduler
        | torch.optim.lr_scheduler.CyclicLR
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | torch.optim.lr_scheduler.ExponentialLR
        | None
    ):
        if isinstance(params, torch.optim.lr_scheduler._LRScheduler):
            return params

        sched_type: str = params["type"].lower()
        base_LR = optimizer.param_groups[0]["lr"]
        if sched_type == "none":
            scheduler = None
        elif sched_type == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=params.get("base_lr", base_LR / 4),
                max_lr=params.get("max_lr", base_LR * 4),
                step_size_up=params.get("step_size_up", 100),
                mode=params.get("mode", "triangular2"),
                cycle_momentum=params.get("momentum", False),
            )
        elif sched_type.startswith(("plat", "reducelronplat")):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 50),
                threshold=params.get("threshold", 1e-4),
                min_lr=params.get("min_lr", base_LR / 20),
                cooldown=params.get("cooldown", 100),
            )
        elif sched_type in ["exp", "gamma", "exponential"]:
            if "gamma" in params.keys():
                gamma = params["gamma"]
            elif num_iter is not None:
                fac = params.get("factor", 0.01)
                gamma = fac ** (1.0 / num_iter)
            else:
                gamma = 0.999
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
        return scheduler

    @property
    def model_object(self) -> "torch.nn.Module":
        if hasattr(self, "_model_object"):
            return self._model_object
        elif hasattr(self, "_pretrained_model_object"):
            return self._pretrained_model_object
        else:
            raise AttributeError(
                "Neither model_object nor pretrained_model_object have been assigned"
            )

    @model_object.setter
    def model_object(self, model: "torch.nn.Module"):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"model_object must be a torch.nn.Module, got {type(model)}"
            )
        self._model_object = model

    @property
    def model_probe(self) -> "torch.nn.Module":
        if hasattr(self, "_model_probe"):
            return self._model_probe
        elif hasattr(self, "_pretrained_model_probe"):
            return self._pretrained_model_probe
        else:
            raise AttributeError(
                "Neither model_probe nor pretrained_model_probe have been assigned"
            )

    @model_probe.setter
    def model_probe(self, model: "torch.nn.Module"):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model_probe must be a torch.nn.Module, got {type(model)}")
        self._model_probe = model

    @property
    def model_object_input(self) -> "torch.Tensor | None":
        return self._model_object_input

    @model_object_input.setter
    def model_object_input(self, t: "torch.Tensor"):
        self._model_object_input = t.clone()

    @property
    def model_probe_input(self) -> "torch.Tensor | None":
        return self._model_probe_input

    @model_probe_input.setter
    def model_probe_input(self, t: "torch.Tensor"):
        self._model_probe_input = t.clone()

    @property
    def pretrained_model_object(self) -> "torch.nn.Module":
        self._pretrained_model_object.load_state_dict(
            self._pretrained_model_object_weights
        )
        return self._pretrained_model_object

    @pretrained_model_object.setter
    def pretrained_model_object(self, model: "torch.nn.Module"):
        self._pretrained_model_object_weights = model.state_dict().copy()
        self._pretrained_model_object: "torch.nn.Module" = deepcopy(model.cpu())

    @property
    def pretrained_model_probe(self) -> "torch.nn.Module":
        self._pretrained_model_probe.load_state_dict(
            self._pretrained_model_probe_weights
        )
        return self._pretrained_model_probe

    @pretrained_model_probe.setter
    def pretrained_model_probe(self, model: "torch.nn.Module"):
        self._pretrained_model_probe_weights = model.state_dict().copy()
        self._pretrained_model_probe = deepcopy(model.cpu())

    # @property
    # def base_lrs(self) -> dict[str,float]:
    #     return self._base_lrs

    # @base_lrs.setter
    # def base_lrs(self, lrs:dict[str,float|tuple[float,float]]) -> None:
    #     for k, v in lrs.items():
    #         if k not in self.DEFAULT_LRS.keys():
    #             raise KeyError(f"Bad lr key {k} | should be one of {self.DEFAULT_LRS.keys()}")
    #         if isinstance(v, (float|int)):
    #             v = float(v)
    #             if v < 0:
    #                 raise ValueError(f"LRs must be >= 0. Got lr {k} = {v}")
    #         else:
    #             raise ValueError(f"LRs must be float or int. Got {type(v)}")

    #     self._base_lrs = self.DEFAULT_LRS | self._base_lrs | lrs
    #     return

    # endregion --- explicit properties and setters ---

    # region --- implicit properties ---

    @property
    def _dtype_real_torch(self) -> "torch.dtype":
        # necessary because torch doesn't like passing strings to convert dtypes
        return getattr(torch, config.get("dtype_real"))

    @property
    def _dtype_complex_torch(self) -> "torch.dtype":
        return getattr(torch, config.get("dtype_complex"))

    # endregion --- implicit properties ---

    # region --- methods ---

    def get_tv_loss(
        self, arr: torch.Tensor, weights: None | tuple[float, float] = None
    ) -> torch.Tensor:
        """
        weight is tuple (weight_z, weight_yx) or float -> (weight, weight)
        for 2D array, only weight_yx is used

        """
        loss = torch.tensor(0, device=self.device, dtype=self._dtype_real_torch)
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

        if arr.is_complex():
            ph = arr.angle()
            loss += self._calc_tv_loss(ph, w)
            amp = arr.abs()
            if torch.max(amp) - torch.min(amp) > 1e-3:  # is complex and not pure_phase
                loss += self._calc_tv_loss(amp, w)
        else:
            loss += self._calc_tv_loss(arr, w)

        return loss

    def _calc_tv_loss(
        self, arr: torch.Tensor, weight: tuple[float, float]
    ) -> torch.Tensor:
        loss = torch.tensor(0, device=self.device, dtype=self._dtype_real_torch)
        for dim in range(arr.ndim):
            if dim == 0 and arr.ndim == 3:  # there's surely a cleaner way but whatev
                w = weight[0]
            else:
                w = weight[1]
            loss += w * torch.mean(torch.abs(arr.diff(dim=dim)))
        loss /= arr.ndim
        return loss

    def reset_recon(self) -> None:
        self._losses = []
        self._recon_types = []
        self._recon_iterations = []
        self._lrs = []
        self._optimizers = {}
        self._schedulers = {}
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()

    def _to_torch(
        self, arr: "np.ndarray | torch.Tensor", dtype: "str | torch.dtype" = "same"
    ) -> "torch.Tensor":
        """
        dtype can be: "same": same as input array, default
                      "object": same as object type, real or complex determined by potential/complex
                      torch.dtype type
        """
        if isinstance(dtype, str):
            dtype = dtype.lower()
            if dtype == "same":
                dt = None
            elif dtype in ["object", "obj"]:
                if np.iscomplexobj(arr):
                    dt = self._dtype_complex_torch
                else:
                    dt = self._dtype_real_torch
            else:
                raise ValueError(
                    f"Unknown string passed {dtype}, dtype should be 'same', 'object' or torch.dtype"
                )
        elif isinstance(dtype, torch.dtype):
            dt = dtype
        else:
            raise TypeError(
                f"dtype should be string or torch.dtype, got {type(dtype)} {dtype}"
            )

        if isinstance(arr, np.ndarray):
            t = torch.tensor(arr.copy(), device=self.device, dtype=dt)
        elif isinstance(arr, cp.ndarray):
            t = torch.tensor(arr, device=self.device, dtype=dt)
        elif isinstance(arr, torch.Tensor):
            t = arr.to(self.device)
            if dt is not None:
                t = t.type(dt)
        elif isinstance(arr, (list, tuple)):
            t = torch.tensor(arr, device=self.device, dtype=dt)
        else:
            raise TypeError(f"arr should be ndarray or Tensor, got {type(arr)}")
        return t

    # endregion --- methods ---

    # region --- reconstruction ---

    def reconstruct(
        self,
        mode: Literal["model", "pixelwise"] | None = None,  # "model" "pixelwise"
        num_iter: int = 0,
        reset: bool = True,
        optimizer_params: dict | None = None,
        object_type: Literal["complex", "pure_phase", "potential"] | None = None,
        models: "tuple[torch.nn.Module, torch.nn.Module] | torch.nn.Module | None" = None,
        # lrs: dict = {},
        scheduler_params: dict | None = None,
        constraints: dict = {},
        batch_size: int | None = None,
        verbose: bool | int | None = None,
        pbar: bool | None = None,
        store_iterations: bool | None = None,
        store_iterations_every: int | None = None,
    ) -> None:
        """
        reason for having a single reconstruct() is so that updating things like constraints
        or recon_types only happens in one place, reason for having separate reoconstruction_
        methods would be to simplify the flags for this and not have to include all

        """
        self._check_preprocessed()
        if mode is not None:
            self.mode = mode

        if reset:
            self.reset_recon()
            obj = self._to_torch(self.initial_object)
            probe = self._to_torch(self.initial_probe)
        else:
            obj = self._to_torch(self.object)
            probe = self._to_torch(self.probe)

        new_optimizers = reset
        new_scheduler = reset
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            new_optimizers = True
            new_scheduler = True

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            new_scheduler = True

        if object_type is not None:
            if object_type != self.object_type:
                if not reset:
                    raise ValueError(
                        f"object_type {object_type} cannot be changed to {object_type} with reset=False."
                    )
                self.object_type = object_type

        self.constraints = constraints  # doesn't overwrite if not reset

        if verbose is not None:
            self.verbose = verbose
        if pbar is None:
            pbar = bool(self.verbose)

        if store_iterations is not None:
            self.store_iterations = store_iterations
        if store_iterations_every is not None:
            self.store_iterations_every = store_iterations_every

        if batch_size is None:
            batch_size = self.gpts[0] * self.gpts[1]

        if new_optimizers:
            self._add_optimizer(key="object", params=obj)
            if "probe" in self.optimizer_params.keys():
                self._add_optimizer(key="probe", params=probe)
                self.constraints["probe"]["fix_probe"] = False
            else:
                self.constraints["probe"]["fix_probe"] = True

        if "descan" in self.optimizer_params.keys():
            # TODO clean this up, learning descan should/could just be on top of the shifted
            # amplitudes, so use the same amplitudes, start with zeros
            _starting_com = np.reshape(
                self._com_fitted, (2, self.gpts[0] * self.gpts[1])
            )
            _starting_com[0] = np.minimum(
                _starting_com[0] + self.roi_shape[0] / 2,
                _starting_com[0] - self.roi_shape[0] / 2,
            )
            _starting_com[1] = np.minimum(
                _starting_com[1] + self.roi_shape[1] / 2,
                _starting_com[1] - self.roi_shape[1] / 2,
            )
            initial_coms = _starting_com.squeeze().T
            t_descan_shifts = torch.tensor(
                -1 * initial_coms,
                dtype=self._dtype_real_torch,
                device=self.device,
                requires_grad=True,
            )
            if new_optimizers:
                self._add_optimizer(key="descan", params=t_descan_shifts)

            t_amplitudes = (
                torch.fft.fftshift(
                    self._to_torch(
                        self.raw_intensities.reshape(
                            (np.prod(self.gpts), *self.roi_shape)
                        ),
                        dtype=self._dtype_real_torch,
                    ),
                    dim=(-2, -1),
                )
                ** 0.5
            )

        else:
            t_descan_shifts = None
            t_amplitudes = self._to_torch(self.shifted_amplitudes)

        if new_scheduler:
            self._schedulers = {}
            # This could be self.schedulers = self.get_schedulers(self.scheduler_params, num_iter)
            # would need to add a setter is all
            self.set_schedulers(self.scheduler_params, num_iter=num_iter)

        t_patch_row = self._to_torch(self.patch_row)
        t_patch_col = self._to_torch(self.patch_col)
        t_position_px_fractional = self._to_torch(self.positions_px_fractional)
        self.propagators = self._to_torch(self.propagators)
        t_mask = self._to_torch(self._object_fov_mask)  # for constraints

        shuffled_indices = np.arange(self.gpts[0] * self.gpts[1])
        # TODO add pbar with loss printout
        for a0 in trange(num_iter, disable=not pbar):
            np.random.shuffle(shuffled_indices)
            loss = torch.tensor(0, device=self.device, dtype=self._dtype_real_torch)

            if self.mode == "pixelwise":
                pred_obj = obj
                pred_probe = probe
            else:
                raise NotImplementedError(f"mode {self.mode} not implemented")

            ### apply constraints here after prediction
            pred_obj, pred_probe = self.apply_constraints(
                object=pred_obj, probe=pred_probe, object_fov_mask=t_mask
            )

            for start, end in generate_batches(
                num_items=self.gpts[0] * self.gpts[1], max_batch=batch_size
            ):
                batch_indices = shuffled_indices[start:end]
                if t_descan_shifts is not None:
                    batch_descan_shifts = t_descan_shifts[batch_indices]
                else:
                    batch_descan_shifts = None

                loss += self.error_estimate(
                    pred_obj,
                    pred_probe,
                    t_patch_row[batch_indices],
                    t_patch_col[batch_indices],
                    t_position_px_fractional[batch_indices],
                    t_amplitudes[batch_indices],
                    batch_descan_shifts,
                )

            loss /= self._mean_diffraction_intensity * np.prod(self.gpts)

            if (
                self.constraints["object"]["tv_weight_z"] > 0
                or self.constraints["object"]["tv_weight_yx"] > 0
            ):
                loss += self.get_tv_loss(pred_obj)

            loss.backward()
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad()

            for sch in self.schedulers.values():
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(loss.item())
                elif sch is not None:
                    sch.step()

            self._lrs.append(self.optimizers["object"].param_groups[0]["lr"])
            self._losses.append(loss.item())
            self._recon_types.append("pixelwise")
            if self.store_iterations and (
                (a0 + 1) % self.store_iterations_every == 0 or a0 == 0
            ):
                self.append_recon_iteration(pred_obj, pred_probe)

        # final constraints application
        if self.mode == "pixelwise":
            pred_obj = obj
            pred_probe = probe
        else:
            raise NotImplementedError
        obj, probe = self.apply_constraints(
            object=pred_obj.detach(), probe=pred_probe.detach(), object_fov_mask=t_mask
        )

        if "descan" in self.optimizer_params.keys():
            assert isinstance(t_descan_shifts, torch.Tensor)  # TODO improve/fix this
            self._descan_shifts = t_descan_shifts.detach()
        else:
            self._descan_shifts = None

        self.object = obj
        self.probe = probe
        return

    # endregion --- reconstruction ---
