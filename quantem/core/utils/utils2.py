from typing import Any, List, Union

import numpy as np

from quantem.core import config

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


# --- Dataset Validation Functions ---
def ensure_valid_array(
    array: np.ndarray | cp.ndarray, dtype: Any = None, ndim: int | None = None
) -> Union[np.ndarray, Any]:
    """Ensure input is a numpy array or cupy array (if available), converting if necessary."""
    if isinstance(array, (np.ndarray, cp.ndarray)):
        validated_array = array.astype(dtype)
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
    value: np.ndarray | tuple | list, ndim: int, name: str, dtype=float
) -> np.ndarray:
    """Validate and convert origin/sampling to a 1D numpy array of type dtype and correct length."""
    if not isinstance(value, (np.ndarray, tuple, list)):
        raise TypeError(
            f"{name} must be a numpy array, tuple, or list, got {type(value)}"
        )

    try:
        arr = np.array(value, dtype=dtype).flatten()
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert {name} to a 1D numeric NumPy array: {e}")

    if len(arr) != ndim:
        raise ValueError(f"Length of {name} ({len(arr)}) must match data ndim ({ndim})")

    return arr


def validate_units(value: Union[List[str], tuple, list], ndim: int) -> List[str]:
    """Validate and convert units to a list of strings of correct length."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Units must be a list or tuple, got {type(value)}")
    elif len(value) != ndim:
        raise ValueError(
            f"Length of units ({len(value)}) must match data ndim ({ndim})"
        )

    return [str(unit) for unit in value]
