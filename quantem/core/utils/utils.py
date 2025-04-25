from dataclasses import dataclass
from typing import Any, List, Optional, Type, Union

import numpy as np

from quantem.core import config

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


# --- Validator Base Class ---
@dataclass
class Validator:
    """Base class for all validators."""

    def __call__(self, value: Any) -> Any:
        """Apply the validation logic to the value."""
        raise NotImplementedError("Subclasses must implement __call__")


# --- Validator Descriptor ---
class ValidatedProperty:
    """
    A descriptor that applies a chain of validators to a property.

    Example:
        class Example:
            level = ValidatedProperty(
                EnsureInt(),
                IsPositive(),
                InsideRange(low=0, high=10),
            )

            def __init__(self, level):
                self.level = level
    """

    def __init__(self, *validators: Validator):
        self.validators = validators
        self.private_name = None

    def __set_name__(self, owner: Type, name: str) -> None:
        """Set the private name for the property."""
        self.private_name = f"_{name}"

    def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Any:
        """Get the property value."""
        if obj is None:
            return self
        return getattr(obj, self.private_name)

    def __set__(self, obj: Any, value: Any) -> None:
        """Set the property value after applying validators."""
        if obj is None:
            return

        # Apply each validator in sequence
        validated_value = value
        for validator in self.validators:
            validated_value = validator(validated_value)

        # Store the validated value
        setattr(obj, self.private_name, validated_value)


# --- Dataset-specific validators ---
@dataclass
class EnsureArray(Validator):
    """Convert value to numpy or cupy array."""

    def __call__(self, value: Any) -> Union[np.ndarray, cp.ndarray]:
        if isinstance(value, (np.ndarray, cp.ndarray)):
            return value
        return np.array(value)


@dataclass
class EnsureArrayDtype(Validator):
    """Ensure array has the correct dtype."""

    dtype: Optional[np.dtype] = None

    def __call__(
        self, value: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        if self.dtype is not None:
            return value.astype(self.dtype)
        return value


@dataclass
class EnsureNdinfo(Validator):
    """Convert value to numpy array for ndinfo fields (origin, sampling)."""

    def __call__(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)


@dataclass
class EnsureUnits(Validator):
    """Convert value to list of strings for units."""

    def __call__(self, value: Any) -> List[str]:
        if isinstance(value, (list, tuple)):
            return [str(unit) for unit in value]
        return [str(value)]


@dataclass
class EnsureStr(Validator):
    """Convert value to string."""

    def __call__(self, value: Any) -> str:
        return str(value)


@dataclass
class ValidateArrayDimensions(Validator):
    """Validate array dimensions match expected ndim."""

    ndim: Optional[int] = None

    def __call__(
        self, value: Union[np.ndarray, cp.ndarray]
    ) -> Union[np.ndarray, cp.ndarray]:
        if self.ndim is not None and value.ndim != self.ndim:
            raise ValueError(
                f"Array dimension {value.ndim} must equal expected dimension {self.ndim}"
            )
        return value


@dataclass
class ValidateNdinfoLength(Validator):
    """Validate ndinfo array length matches expected ndim."""

    ndim: Optional[int] = None

    def __call__(self, value: np.ndarray) -> np.ndarray:
        if self.ndim is not None and len(value) != self.ndim:
            raise ValueError(f"Length {len(value)} must match dimension {self.ndim}")
        return value
