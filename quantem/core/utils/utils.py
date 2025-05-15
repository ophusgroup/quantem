from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from quantem.core import config
from quantem.core.config import (
    device_id_to_int as device_id_to_int,  # just adding to namespace
)

if TYPE_CHECKING:
    import cupy as cp  # type: ignore
    import torch  # type: ignore
else:
    if config.get("has_cupy"):
        import cupy as cp
    if config.get("has_torch"):
        import torch


def get_array_module(array: NDArray):
    """Returns np or cp depending on the array type."""
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp
    if isinstance(array, np.ndarray):
        return np
    raise ValueError(f"Input is not a numpy array or cupy array: {type(array)}")


def get_tensor_module(tensor: NDArray):
    """
    This is like get_array_module but includes torch. It is kept explicitly separate as in most
    cases get_array_module is used, and that fails if given a torch.Tensor.
    """
    if config.get("has_torch"):
        if isinstance(tensor, torch.Tensor):
            return torch
    if config.get("has_cupy"):
        if isinstance(tensor, cp.ndarray):
            return cp
    if isinstance(tensor, np.ndarray):
        return np
    raise ValueError(
        f"Input is not a numpy array, cupy array, or torch tensor: {type(tensor)}"
    )


def as_numpy(array: Any) -> np.ndarray:
    """Convert a torch.Tensor or cupy.ndarray to a numpy.ndarray."""
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    if config.get("has_torch"):
        if isinstance(array, torch.Tensor):
            return array.cpu().numpy()
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)
