import math
from itertools import product
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from tqdm import tqdm

from quantem.core import config

if TYPE_CHECKING:
    import cupy as cp
    import torch
else:
    if config.get("has_cupy"):
        import cupy as cp
    if config.get("has_torch"):
        import torch


# region --- array module stuff ---


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


def as_numpy(array: "np.ndarray | cp.ndarray | torch.Tensor") -> np.ndarray:
    """Convert a torch.Tensor or cupy.ndarray to a numpy.ndarray. Always returns
    a copy for consistency."""
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    if config.get("has_torch"):
        if isinstance(array, torch.Tensor):
            return array.cpu().detach().numpy()
    if isinstance(array, np.ndarray):
        return np.array(array)
    if isinstance(array, (list, tuple)):
        try:
            return np.array(array)
        except (ValueError, TypeError):
            ar2 = [as_numpy(i) for i in array]
            try:
                return np.array(ar2)
            except (ValueError, TypeError):
                pass
    try:
        return np.array(array)
    except (ValueError, TypeError):
        raise TypeError(
            f"Input is not a numpy array or convertible to one: {type(array)}"
        )


def to_cpu(arrs: Any) -> np.ndarray | Sequence:
    if config.get("has_cupy"):
        if isinstance(arrs, cp.ndarray):
            return cp.asnumpy(arrs)
    if config.get("has_torch"):
        if isinstance(arrs, torch.Tensor):
            return arrs.cpu().detach().numpy()
    if isinstance(arrs, np.ndarray):
        return np.array(arrs)
    elif isinstance(arrs, list):
        return [to_cpu(i) for i in arrs]
    elif isinstance(arrs, tuple):
        return tuple([to_cpu(i) for i in arrs])
    else:
        raise NotImplementedError(f"Unkown type: {type(arrs)}")


# endregion


# region --- TEM ---


def electron_wavelength_angstrom(E_eV: float):
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34

    lam = (
        h
        / math.sqrt(2 * m * e * E_eV)
        / math.sqrt(1 + e * E_eV / 2 / m / c**2)
        * 10**10
    )
    return lam


# endregion


# region --- generally useful bits ---


def tqdmnd(*iterables, **kwargs):
    return tqdm(list(product(*iterables)), **kwargs)


# endregion
