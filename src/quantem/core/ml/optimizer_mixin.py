from abc import abstractmethod
from typing import TYPE_CHECKING, Generator, Iterator, Sequence

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class OptimizerMixin:
    """
    Mixin class for handling optimizer and scheduler management.
    Each model (object, probe, dataset) can inherit from this to manage its own optimizers.
    """

    DEFAULT_OPTIMIZER_TYPE = "adam"

    def __init__(self, *args, **kwargs):
        """Initialize the optimizer mixin."""
        self._optimizer = None
        self._scheduler = None
        self._optimizer_params = {}
        self._scheduler_params = {}
        # Don't call super().__init__() in mixin classes to avoid MRO issues

    @property
    def optimizer(self) -> "torch.optim.Optimizer | None":
        """Get the optimizer for this model."""
        return self._optimizer

    @property
    def scheduler(self) -> "torch.optim.lr_scheduler.LRScheduler | None":
        """Get the scheduler for this model."""
        return self._scheduler

    @property
    def optimizer_params(self) -> dict:
        """Get the optimizer parameters."""
        return self._optimizer_params

    @optimizer_params.setter
    def optimizer_params(self, params: dict):
        """Set the optimizer parameters."""
        self._optimizer_params = params.copy() if params else {}

    @property
    def scheduler_params(self) -> dict:
        """Get the scheduler parameters."""
        return self._scheduler_params

    @scheduler_params.setter
    def scheduler_params(self, params: dict):
        """Set the scheduler parameters."""
        self._scheduler_params = params.copy() if params else {}

    @abstractmethod
    def get_optimization_parameters(
        self,
    ) -> "torch.Tensor | Sequence[torch.Tensor] | Iterator[torch.Tensor]":
        """Get the parameters that should be optimized for this model."""
        raise NotImplementedError("Subclasses must implement get_optimization_parameters")

    def set_optimizer(self, opt_params: dict | None = None) -> None:
        """Set the optimizer for this model."""
        if opt_params is not None:
            self.optimizer_params = opt_params

        if not self._optimizer_params:
            self._optimizer = None
            return

        params = self.get_optimization_parameters()
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, Generator):
            params = list(params)

        # Ensure parameters require gradients
        for p in params:
            p.requires_grad_(True)

        opt_params = self._optimizer_params.copy()
        opt_type = opt_params.pop("type", self.DEFAULT_OPTIMIZER_TYPE)

        if isinstance(opt_type, type):
            self._optimizer = opt_type(params, **opt_params)
        elif isinstance(opt_type, str):
            if opt_type.lower() == "adam":
                self._optimizer = torch.optim.Adam(params, **opt_params)
            elif opt_type.lower() == "adamw":
                self._optimizer = torch.optim.AdamW(params, **opt_params)
            elif opt_type.lower() == "sgd":
                self._optimizer = torch.optim.SGD(params, **opt_params)
            else:
                raise NotImplementedError(f"Unknown optimizer type: {opt_type}")
        else:
            raise TypeError(f"optimizer type must be string or type, got {type(opt_type)}")

    def set_scheduler(
        self, scheduler_params: dict | None = None, num_iter: int | None = None
    ) -> None:
        """Set the scheduler for this model."""
        if scheduler_params is not None:
            self.scheduler_params = scheduler_params

        if not self._scheduler_params or self._optimizer is None:
            self._scheduler = None
            return

        params = self._scheduler_params
        sched_type = params.get("type", "none").lower()
        optimizer = self._optimizer
        base_LR = optimizer.param_groups[0]["lr"]

        if sched_type == "none":
            self._scheduler = None
        elif sched_type == "cyclic":
            self._scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=params.get("base_lr", base_LR / 4),
                max_lr=params.get("max_lr", base_LR * 4),
                step_size_up=params.get("step_size_up", 100),
                step_size_down=params.get("step_size_down", params.get("step_size_up", 100)),
                mode=params.get("mode", "triangular2"),
                cycle_momentum=params.get("momentum", False),
            )
        elif sched_type.startswith(("plat", "reducelronplat")):
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 10),
                threshold=params.get("threshold", 1e-3),
                min_lr=params.get("min_lr", base_LR / 20),
                cooldown=params.get("cooldown", 50),
            )
        elif sched_type in ["exp", "gamma", "exponential"]:
            if "gamma" in params:
                gamma = params["gamma"]
            elif num_iter is not None:
                fac = params.get("factor", 0.01)
                gamma = fac ** (1.0 / num_iter)
            else:
                gamma = 0.999
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

    def step_optimizer(self) -> None:
        """Step the optimizer if it exists."""
        if self._optimizer is not None:
            self._optimizer.step()

    def zero_grad(self) -> None:
        """Zero gradients if optimizer exists."""
        if self._optimizer is not None:
            self._optimizer.zero_grad()

    def step_scheduler(self, loss: float | None = None) -> None:
        """Step the scheduler if it exists."""
        if self._scheduler is not None:
            if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if loss is not None:
                    self._scheduler.step(loss)
            else:
                self._scheduler.step()

    def has_optimizer(self) -> bool:
        """Check if this model has an active optimizer."""
        return self._optimizer is not None

    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        if self._optimizer is not None:
            return self._optimizer.param_groups[0]["lr"]
        return 0.0

    def reset_optimizer(self) -> None:
        """Reset the optimizer and scheduler."""
        self._optimizer = None
        self._scheduler = None
