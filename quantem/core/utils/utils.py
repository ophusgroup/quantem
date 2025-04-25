from functools import wraps
from typing import Any, Callable

import numpy as np

from quantem.core import config

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


# --- Validation Decorator ---
def validated(validator_factory: Callable) -> Callable:
    """
    Decorator for property-style validated fields.

    Example:
        @validated(lambda self: pipe(ensure_int, is_positive))
        def my_attr(self): ...
    """

    def decorator(func):
        attr_name = func.__name__
        private_name = f"_{attr_name}"

        @property
        def prop(self):
            return getattr(self, private_name)

        @prop.setter
        def prop(self, value):
            validator = validator_factory(self)
            validated_value = validator(value)
            setattr(self, private_name, validated_value)

        return prop

    return decorator


# --- Validator composition ---
def pipe(*funcs: Callable) -> Callable:
    """Compose multiple validator/converter functions into one."""

    def composed(val):
        for f in funcs:
            val = f(val)
        return val

    return composed


# --- Dataset-specific validators and converters ---
def ensure_array(value: Any) -> np.ndarray | cp.ndarray:
    """Convert value to numpy or cupy array."""
    if not isinstance(value, (np.ndarray, cp.ndarray)):
        raise ValueError("Value must be a numpy or cupy array")
    return value


def ensure_array_dtype(
    value: np.ndarray | cp.ndarray, dtype: np.dtype | None = None
) -> np.ndarray | cp.ndarray:
    """Ensure array has the correct dtype."""
    if dtype is not None:
        return value.astype(dtype)
    return value


def ensure_ndinfo(value: Any) -> np.ndarray:
    """Convert value to numpy array for ndinfo fields (origin, sampling)."""
    if not isinstance(value, (np.ndarray, tuple, list)):
        raise TypeError(f"Value should be a ndarray/list/tuple. Got type {type(value)}")
    return np.array(value)


def ensure_units(value: Any) -> list[str]:
    """Convert value to list of strings for units."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Units must be a list or tuple. Got type {type(value)}")
    return [str(unit) for unit in value]


def ensure_str(value: Any) -> str:
    """Convert value to string."""
    return str(value)


def validate_array_dimensions(
    value: np.ndarray | cp.ndarray, ndim: int | None = None
) -> np.ndarray | cp.ndarray:
    """Validate array dimensions match expected ndim."""
    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"Array dimension {value.ndim} must equal expected dimension {ndim}"
        )
    return value


def validate_ndinfo_length(value: np.ndarray, ndim: int | None = None) -> np.ndarray:
    """Validate ndinfo array length matches expected ndim."""
    if ndim is not None and len(value) != ndim:
        raise ValueError(f"Length {len(value)} must match dimension {ndim}")
    return value
