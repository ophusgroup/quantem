from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
import pytest

from quantem.core.datastructures.dataset import Dataset
from quantem.core.utils.utils import (
    EnsureArray,
    EnsureArrayDtype,
    EnsureNdinfo,
    EnsureStr,
    EnsureUnits,
    ValidateArrayDimensions,
    ValidatedProperty,
    ValidateListLength,
    ValidateNdinfoLength,
    Validator,
)


# --- Additional Validators for Tests ---
@dataclass
class EnsureInt(Validator[int]):
    """Convert value to integer."""

    def __call__(self, value: Any, instance: Any = None) -> int:
        return int(value)


@dataclass
class IsPositive(Validator[int]):
    """Ensure value is positive."""

    def __call__(self, value: int, instance: Any = None) -> int:
        if value <= 0:
            raise ValueError("Value must be positive")
        return value


@dataclass
class InsideRange(Validator[int]):
    """Ensure value is inside specified range."""

    low: int
    high: int

    def __call__(self, value: int, instance: Any = None) -> int:
        if not self.low <= value <= self.high:
            raise ValueError(f"Value must be between {self.low} and {self.high}")
        return value


# --- Test Classes ---
class Example:
    """A simple example class using the validation system."""

    # Define validated properties
    level = ValidatedProperty[int](
        EnsureInt(),
        IsPositive(),
        InsideRange(low=0, high=10),
    )

    name = ValidatedProperty[str](EnsureStr())

    def __init__(self, level: Union[int, str], name: Union[str, int]):
        self.level = level
        self.name = name


# --- Tests ---
def test_dataset_validation():
    """Test that the Dataset validators are working correctly."""
    # Test valid dataset creation
    array = np.zeros((10, 10))
    dataset = Dataset.from_array(
        array=array,
        name="test",
        origin=np.zeros(2),
        sampling=np.ones(2),
        units=["nm", "nm"],
        signal_units="counts",
    )

    assert isinstance(dataset.array, np.ndarray)
    assert dataset.name == "test"
    assert isinstance(dataset.origin, np.ndarray)
    assert isinstance(dataset.sampling, np.ndarray)
    assert isinstance(dataset.units, list)
    assert all(isinstance(unit, str) for unit in dataset.units)
    assert isinstance(dataset.signal_units, str)

    # Test invalid array dimensions
    with pytest.raises(ValueError):
        Dataset.from_array(
            array=np.zeros((10, 10, 10)),  # 3D array
            origin=np.zeros(2),  # 2D origin
            sampling=np.ones(2),
            units=["nm", "nm"],
        )

    # Test invalid units length
    with pytest.raises(ValueError):
        Dataset.from_array(
            array=array,
            origin=np.zeros(2),
            sampling=np.ones(2),
            units=["nm"],  # Wrong length
        )

    # Test invalid origin/sampling length
    with pytest.raises(ValueError):
        Dataset.from_array(
            array=array,
            origin=np.zeros(3),  # Wrong length
            sampling=np.ones(2),
            units=["nm", "nm"],
        )


def test_example_class_validation():
    """Test that the Example class validators are working correctly."""
    # Test valid initialization
    example = Example(level=5, name="test")
    assert example.level == 5
    assert example.name == "test"

    # Test type conversion
    example = Example(level="5", name=123)
    assert example.level == 5
    assert example.name == "123"

    # Test validation errors
    with pytest.raises(ValueError):
        Example(level=-1, name="test")  # Negative value

    with pytest.raises(ValueError):
        Example(level=11, name="test")  # Value too high

    with pytest.raises(ValueError):
        Example(level="abc", name="test")  # Invalid integer
