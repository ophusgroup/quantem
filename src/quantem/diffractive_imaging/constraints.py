from abc import ABC, abstractmethod
from typing import Any

import torch

from quantem.core import config


class ConstraintLossMixin:
    """Mixin class for handling constraint loss accumulation across batches."""

    def accumulate_constraint_losses(
        self, batch_constraint_losses: dict[str, torch.Tensor]
    ) -> None:
        if not hasattr(self, "_epoch_constraint_losses"):
            self._epoch_constraint_losses = {}

        for loss_name, loss_value in batch_constraint_losses.items():
            if loss_name not in self._epoch_constraint_losses:
                self._epoch_constraint_losses[loss_name] = 0.0
            self._epoch_constraint_losses[loss_name] += loss_value.item()

    def get_epoch_constraint_losses(self) -> dict[str, float]:
        return getattr(self, "_epoch_constraint_losses", {})

    def reset_epoch_constraint_losses(self) -> None:
        self._epoch_constraint_losses = {}


class BaseConstraints(ConstraintLossMixin, ABC):
    """Base class for constraint management with common functionality."""

    # Subclasses should define their own DEFAULT_CONSTRAINTS
    DEFAULT_CONSTRAINTS: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._soft_constraint_loss = {}
        self._constraints = self.DEFAULT_CONSTRAINTS.copy()

    @property
    def constraints(self) -> dict[str, Any]:
        return self._constraints

    @constraints.setter
    def constraints(self, c: dict[str, Any]):
        allowed_keys = self.DEFAULT_CONSTRAINTS.keys()
        constraint_type = self.__class__.__name__.lower().replace("constraints", "")

        for key, value in c.items():
            if key not in allowed_keys:
                raise KeyError(
                    f"Invalid {constraint_type} constraint key '{key}', allowed keys are {list(allowed_keys)}"
                )
            self._constraints[key] = value

    @property
    def soft_constraint_loss(self) -> dict[str, torch.Tensor]:
        return self._soft_constraint_loss

    def add_constraint(self, key: str, value: Any):
        allowed_keys = self.DEFAULT_CONSTRAINTS.keys()
        constraint_type = self.__class__.__name__.lower().replace("constraints", "")

        if key not in allowed_keys:
            raise KeyError(
                f"Invalid {constraint_type} constraint key '{key}', allowed keys are {list(allowed_keys)}"
            )
        self._constraints[key] = value

    @abstractmethod
    def apply_soft_constraints(self, *args, **kwargs) -> torch.Tensor:
        """Apply soft constraints and return total constraint loss."""
        pass

    def _get_zero_loss_tensor(self) -> torch.Tensor:
        """Helper method to create a zero loss tensor with proper device and dtype."""
        device = getattr(self, "device", "cpu")
        return torch.tensor(0, device=device, dtype=getattr(torch, config.get("dtype_real")))
