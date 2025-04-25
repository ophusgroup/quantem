from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pytest

from quantem.core.datastructures.dataset import Dataset
from quantem.core.utils.utils import (
    EnsureArray,
    EnsureArrayDtype,
    EnsureInt,
    EnsureNdinfo,
    EnsureStr,
    EnsureUnits,
    InsideRange,
    IsPositive,
    ValidateArrayDimensions,
    ValidatedProperty,
    ValidateNdinfoLength,
    Validator,
)


# --- Additional Validators for Tests ---
@dataclass
class EnsureInt(Validator):
    """Convert value to integer."""

    def __call__(self, value: Any, instance: Any = None) -> int:
        return int(value)


@dataclass
class IsPositive(Validator):
    """Ensure value is positive."""

    def __call__(
        self, value: Union[int, float], instance: Any = None
    ) -> Union[int, float]:
        if value <= 0:
            raise ValueError(f"Value must be positive, got {value}")
        return value


@dataclass
class InsideRange(Validator):
    """Ensure value is inside a range."""

    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None

    def __call__(
        self, value: Union[int, float], instance: Any = None
    ) -> Union[int, float]:
        if self.low is not None and value < self.low:
            raise ValueError(f"Value {value} is below minimum {self.low}")
        if self.high is not None and value > self.high:
            raise ValueError(f"Value {value} is above maximum {self.high}")
        return value


# --- Test Classes ---
class Example:
    """A simple example class using the validation system."""

    # Define validated properties
    level = ValidatedProperty(
        EnsureInt(),
        IsPositive(),
        InsideRange(low=0, high=10),
    )

    name = ValidatedProperty(EnsureStr())

    def __init__(self, level: int, name: str):
        self.level = level
        self.name = name


# --- Tests ---
def test_dataset_validation():
    """Test that the Dataset validators are working correctly."""
    # Create a 4D dataset
    dataset = Dataset.from_array(
        np.random.rand(10, 10, 20, 20),
    )

    # Test that setting units to 4 values (matching ndim) succeeds
    dataset.units = ["pixels", "pixels", "pixels", "pixels"]
    assert len(dataset.units) == 4

    # Test that setting units to 2 values (not matching ndim) fails
    with pytest.raises(ValueError, match="Length 2 must match dimension 4"):
        dataset.units = ["pixels", "pixels"]

    # Test that setting origin to 2 values (not matching ndim) fails
    with pytest.raises(ValueError, match="Length 2 must match dimension 4"):
        dataset.origin = np.array([0, 0])

    # Test that setting origin to 4 values (matching ndim) succeeds
    dataset.origin = np.array([0, 0, 0, 0])
    assert len(dataset.origin) == 4


def test_example_class_validation():
    """Test that the Example class validators are working correctly."""
    # Create an example instance
    example = Example(level=5, name="Test")

    # Test that the level is set correctly
    assert example.level == 5

    # Test that setting level to a valid value succeeds
    example.level = 7
    assert example.level == 7

    # Test that setting level to an invalid value (negative) fails
    with pytest.raises(ValueError, match="Value must be positive"):
        example.level = -1

    # Test that setting level to an invalid value (too high) fails
    with pytest.raises(ValueError, match="Value 15 is above maximum 10"):
        example.level = 15

    # Test that setting level to a non-integer value is converted
    example.level = 3.7
    assert example.level == 3

    # Test that the name is set correctly
    assert example.name == "Test"

    # Test that setting name to a non-string value is converted
    example.name = 123
    assert example.name == "123"
