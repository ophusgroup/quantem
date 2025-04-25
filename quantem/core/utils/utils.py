from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Type, TypeVar, Union, cast

import numpy as np

from quantem.core import config

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


# Define a type variable for the property value
T = TypeVar("T")


# --- Validator Base Class ---
@dataclass
class Validator(Generic[T]):
    """Base class for all validators with generic type parameter."""

    def __call__(self, value: Any, instance: Any = None) -> T:
        """Apply the validation logic to the value."""
        raise NotImplementedError("Subclasses must implement __call__")


# --- Validator Descriptor ---
class ValidatedProperty(Generic[T]):
    """
    A descriptor that applies a chain of validators to a property.

    Example:
        class Example:
            level = ValidatedProperty[int](
                EnsureInt(),
                IsPositive(),
                InsideRange(low=0, high=10),
            )

            def __init__(self, level):
                self.level = level
    """

    def __init__(self, *validators: Validator[T]):
        self.validators = validators
        self.private_name = None

    def __set_name__(self, owner: Type, name: str) -> None:
        """Set the private name for the property."""
        self.private_name = f"_{name}"

    def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Any:
        """Get the property value."""
        if obj is None:
            return self
        if self.private_name is None:
            raise AttributeError("Property name not set")
        return getattr(obj, self.private_name)

    def __set__(self, obj: Any, value: Any) -> None:
        """Set the property value after applying validators."""
        if obj is None:
            return

        # Apply each validator in sequence
        validated_value = value
        for validator in self.validators:
            validated_value = validator(validated_value, obj)

        # Store the validated value
        if self.private_name is None:
            raise AttributeError("Property name not set")
        setattr(obj, self.private_name, validated_value)


# --- Dataset-specific validators ---
@dataclass
class EnsureArray(Validator[Union[np.ndarray, cp.ndarray]]):
    """Convert value to numpy or cupy array.

    This validator converts the input value to a numpy or cupy array if it isn't already one.
    If the value is already a numpy or cupy array, it is returned as is.
    """

    def __call__(
        self, value: Any, instance: Any = None
    ) -> Union[np.ndarray, cp.ndarray]:
        if isinstance(value, (np.ndarray, cp.ndarray)):
            return value
        return np.array(value)


@dataclass
class EnsureArrayDtype(Validator[Union[np.ndarray, cp.ndarray]]):
    """Ensure array has the correct dtype.

    This validator ensures that the array has the specified dtype by converting it if necessary.
    If no dtype is specified, the array is returned as is.
    """

    dtype: Optional[np.dtype] = None

    def __call__(
        self, value: Union[np.ndarray, cp.ndarray], instance: Any = None
    ) -> Union[np.ndarray, cp.ndarray]:
        if self.dtype is not None:
            return value.astype(self.dtype)
        return value


@dataclass
class EnsureNdinfo(Validator[np.ndarray]):
    """Convert value to numpy array for ndinfo fields (origin, sampling).

    This validator converts the input value to a numpy array if it isn't already one.
    If the value is already a numpy array, it is returned as is.
    """

    def __call__(self, value: Any, instance: Any = None) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)


@dataclass
class EnsureUnits(Validator[List[str]]):
    """Convert value to list of strings for units.

    This validator converts the input value to a list of strings.
    If the value is already a list or tuple, each element is converted to a string.
    If the value is a single item, it is converted to a string and wrapped in a list.
    """

    def __call__(self, value: Any, instance: Any = None) -> List[str]:
        if isinstance(value, (list, tuple)):
            return [str(unit) for unit in value]
        return [str(value)]


@dataclass
class EnsureStr(Validator[str]):
    """Convert value to string.

    This validator converts the input value to a string.
    """

    def __call__(self, value: Any, instance: Any = None) -> str:
        return str(value)


@dataclass
class ValidateArrayDimensions(Validator[Union[np.ndarray, cp.ndarray]]):
    """Validate array dimensions match expected ndim.

    This validator checks if the array dimensions match the expected number of dimensions.
    If the dimensions don't match, a ValueError is raised.
    """

    ndim: Optional[int] = None

    def __call__(
        self, value: Union[np.ndarray, cp.ndarray], instance: Any = None
    ) -> Union[np.ndarray, cp.ndarray]:
        if instance is not None and hasattr(instance, "ndim"):
            expected_ndim = instance.ndim
        else:
            expected_ndim = self.ndim

        if expected_ndim is not None and value.ndim != expected_ndim:
            raise ValueError(
                f"Array dimension {value.ndim} must equal expected dimension {expected_ndim}"
            )
        return value


@dataclass
class ValidateNdinfoLength(Validator[np.ndarray]):
    """Validate ndinfo array length matches expected ndim.

    This validator checks if the ndinfo array length matches the expected number of dimensions.
    If the length doesn't match, a ValueError is raised.
    """

    ndim: Optional[int] = None

    def __call__(self, value: np.ndarray, instance: Any = None) -> np.ndarray:
        if instance is not None and hasattr(instance, "ndim"):
            expected_ndim = instance.ndim
        else:
            expected_ndim = self.ndim

        if expected_ndim is not None and len(value) != expected_ndim:
            raise ValueError(
                f"Length {len(value)} must match dimension {expected_ndim}"
            )
        return value


@dataclass
class ValidateListLength(Validator[List[str]]):
    """Validate list length matches expected ndim.

    This validator checks if the list length matches the expected number of dimensions.
    If the length doesn't match, a ValueError is raised.
    """

    ndim: Optional[int] = None

    def __call__(self, value: List[str], instance: Any = None) -> List[str]:
        if instance is not None and hasattr(instance, "ndim"):
            expected_ndim = instance.ndim
        else:
            expected_ndim = self.ndim

        if expected_ndim is not None and len(value) != expected_ndim:
            raise ValueError(
                f"Length {len(value)} must match dimension {expected_ndim}"
            )
        return value
