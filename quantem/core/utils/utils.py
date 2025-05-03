from typing import TYPE_CHECKING

import numpy as np
import math 

from quantem.core import config
from quantem.core.config import (
    device_id_to_int as device_id_to_int,  # just adding to namespace
)

if TYPE_CHECKING:
    import cupy as cp
    import torch
else:
    if config.get("has_cupy"):
        import cupy as cp
    if config.get("has_torch"):
        import torch


#region --- array module stuff ---

def get_array_module(array: "np.ndarray | cp.ndarray"):
    """Returns np or cp depending on the array type."""
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp
    if isinstance(array, np.ndarray):
        return np
    raise ValueError(f"Input is not a numpy array or cupy array: {type(array)}")


def get_tensor_module(tensor: "np.ndarray | cp.ndarray | torch.Tensor"):
    """
    This is like get_array_module but includes torch. It is kept explicitly seperate as in most
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


def as_numpy(array: np.ndarray | cp.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor or cupy.ndarray to a numpy.ndarray."""
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    if config.get("has_torch"):
        if isinstance(array, torch.Tensor):
            return array.cpu().detach().numpy()
    if isinstance(array, np.ndarray):
        return array
    try:    
        return np.asarray(array)
    except (ValueError, TypeError): 
        raise TypeError(f"Input is not a numpy array or convertible to one: {type(array)}")

#endregion


#region --- TEM ---

def electron_wavelength_angstrom(E_eV:float):
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34

    lam = h / math.sqrt(2 * m * e * E_eV) / math.sqrt(1 + e * E_eV / 2 / m / c**2) * 10**10
    return lam

#endregion

