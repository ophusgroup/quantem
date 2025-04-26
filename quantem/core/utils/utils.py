from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from quantem.core import config

if config.get("has_cupy"):
    import cupy as cp  # type: ignore
else:
    import numpy as cp


# --- Dataset Validation Functions ---
def ensure_valid_array(
    array: Union[NDArray, Any], dtype: DTypeLike = None, ndim: int | None = None
) -> Union[NDArray, Any]:
    """Ensure input is a numpy array or cupy array (if available), converting if necessary."""
    if isinstance(array, (np.ndarray, cp.ndarray)):
        if dtype is not None:
            validated_array = array.astype(dtype)
        else:
            validated_array = array
    else:
        try:
            validated_array = np.array(array, dtype=dtype)
            if validated_array.ndim < 1:
                raise ValueError("Array must be at least 1D")
            elif not np.issubdtype(validated_array.dtype, np.number):
                raise ValueError("Array must contain numeric values")
        except Exception as e:
            raise TypeError(f"Input could not be converted to a NumPy array: {e}")
        if ndim is not None:
            if validated_array.ndim != ndim:
                raise ValueError(
                    f"Array ndim {validated_array.ndim} does not match expected ndim {ndim}"
                )
    return validated_array


def validate_ndinfo(
    value: Union[NDArray, tuple, list, float, int], ndim: int, name: str, dtype=None
) -> NDArray:
    """Validate and convert origin/sampling to a 1D numpy array of type dtype and correct length."""
    if np.isscalar(value):
        arr = np.full(ndim, value, dtype=dtype)
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{name} must contain numeric values")
        return arr
    elif not isinstance(value, (np.ndarray, tuple, list)):
        raise TypeError(
            f"{name} must be a numpy array, tuple, list, or scalar, got {type(value)}"
        )

    try:
        arr = np.array(value, dtype=dtype).flatten()
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert {name} to a 1D numeric NumPy array: {e}")

    if len(arr) != ndim:
        raise ValueError(f"Length of {name} ({len(arr)}) must match data ndim ({ndim})")

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")

    return arr


def validate_units(value: Union[List[str], tuple, list, str], ndim: int) -> List[str]:
    """Validate and convert units to a list of strings of correct length."""
    if isinstance(value, str):
        return [value] * ndim
    elif not isinstance(value, (list, tuple)):
        raise TypeError(f"Units must be a list, tuple, or string, got {type(value)}")
    elif len(value) != ndim:
        raise ValueError(
            f"Length of units ({len(value)}) must match data ndim ({ndim})"
        )

    return [str(unit) for unit in value]


# --- Vector Validation Functions ---
def validate_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Validate and convert shape to a tuple of integers.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape to validate

    Returns
    -------
    Tuple[int, ...]
        The validated shape

    Raises
    ------
    ValueError
        If shape contains non-positive integers
    TypeError
        If shape is not a tuple or contains non-integer values
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be a tuple, got {type(shape)}")

    validated = []
    for dim in shape:
        if not isinstance(dim, int):
            raise TypeError(f"Shape dimensions must be integers, got {type(dim)}")
        if dim <= 0:
            raise ValueError(f"Shape dimensions must be positive, got {dim}")
        validated.append(dim)

    return tuple(validated)


def validate_num_fields(num_fields: int, fields: Optional[List[str]] = None) -> int:
    """
    Validate number of fields.

    Parameters
    ----------
    num_fields : int
        The number of fields
    fields : Optional[List[str]]
        List of field names

    Returns
    -------
    int
        The validated number of fields

    Raises
    ------
    ValueError
        If num_fields is not positive or doesn't match fields length
    """
    if not isinstance(num_fields, int):
        raise TypeError(f"num_fields must be an integer, got {type(num_fields)}")
    if num_fields <= 0:
        raise ValueError(f"num_fields must be positive, got {num_fields}")
    if fields is not None and len(fields) != num_fields:
        raise ValueError(
            f"num_fields ({num_fields}) does not match length of fields ({len(fields)})"
        )
    return num_fields


def validate_fields(fields: List[str], num_fields: int) -> List[str]:
    """
    Validate field names.

    Parameters
    ----------
    fields : List[str]
        List of field names
    num_fields : int
        Expected number of fields

    Returns
    -------
    List[str]
        The validated field names

    Raises
    ------
    ValueError
        If fields has duplicate names or wrong length
    """
    if not isinstance(fields, (list, tuple)):
        raise TypeError(f"fields must be a list or tuple, got {type(fields)}")
    if len(fields) != num_fields:
        raise ValueError(
            f"Length of fields ({len(fields)}) must match num_fields ({num_fields})"
        )
    if len(set(fields)) != len(fields):
        raise ValueError("Duplicate field names are not allowed")
    return [str(field) for field in fields]


def validate_vector_units(units: List[str], num_fields: int) -> List[str]:
    """
    Validate units for fields.

    Parameters
    ----------
    units : List[str]
        List of units
    num_fields : int
        Expected number of fields

    Returns
    -------
    List[str]
        The validated units

    Raises
    ------
    ValueError
        If units has wrong length
    """
    if not isinstance(units, (list, tuple)):
        raise TypeError(f"units must be a list or tuple, got {type(units)}")
    if len(units) != num_fields:
        raise ValueError(
            f"Length of units ({len(units)}) must match num_fields ({num_fields})"
        )
    return [str(unit) for unit in units]


def validate_vector_data(
    data: List[Any], shape: Tuple[int, ...], num_fields: int
) -> List[Any]:
    """
    Validate data structure.

    Parameters
    ----------
    data : List[Any]
        The data to validate
    shape : Tuple[int, ...]
        Expected shape
    num_fields : int
        Expected number of fields

    Returns
    -------
    List[Any]
        The validated data

    Raises
    ------
    ValueError
        If data structure doesn't match shape or fields
    """
    # This is a placeholder - actual implementation would need to recursively
    # validate the nested structure matches the shape and contains arrays with
    # the correct number of fields
    return data
