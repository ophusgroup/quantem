from typing import TYPE_CHECKING, Any, Generator

import numpy as np

from quantem.core import config

if TYPE_CHECKING:
    import cupy as cp
    import torch
else:
    if config.get("has_cupy"):
        import cupy as cp
    if config.get("has_torch"):
        import torch


def subdivide_into_batches(
    num_items: int, num_batches: int | None = None, max_batch: int | None = None
):
    """
    Split an n integer into m (almost) equal integers, such that the sum of smaller integers equals n.

    Parameters
    ----------
    n: int
        The integer to split.
    m: int
        The number integers n will be split into.

    Returns
    -------
    list of int
    """
    if (num_batches is not None) & (max_batch is not None):
        raise RuntimeError("num_batches and max_batch may not both be provided")

    if num_batches is None:
        if max_batch is not None:
            num_batches = (num_items + (-num_items % max_batch)) // max_batch
        else:
            raise RuntimeError(
                "max_batch must be provided if num_batches is not provided"
            )

    if num_items < num_batches:
        raise RuntimeError("num_batches may not be larger than num_items")

    elif num_items % num_batches == 0:
        return [num_items // num_batches] * num_batches
    else:
        v = []
        zp = num_batches - (num_items % num_batches)
        pp = num_items // num_batches
        for i in range(num_batches):
            if i >= zp:
                v = [pp + 1] + v
            else:
                v = [pp] + v
        return v


def generate_batches(
    num_items: int,
    num_batches: int | None = None,
    max_batch: int | None = None,
    start=0,
) -> Generator[tuple[int, int], Any, None]:
    for batch in subdivide_into_batches(num_items, num_batches, max_batch):
        end = start + batch
        yield start, end

        start = end


def get_array_module(array: "torch.Tensor | np.ndarray | cp.ndarray"):
    if config.get("has_torch"):
        if isinstance(array, torch.Tensor):
            return torch
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp
    if isinstance(array, np.ndarray):
        return np
    else:
        raise NotImplementedError


def fourier_shift(
    array: "torch.Tensor | np.ndarray | cp.ndarray", positions: "torch.Tensor | np.ndarray | cp.ndarray"
) -> "torch.Tensor | np.ndarray | cp.ndarray":
    """Fourier-shift array by flat array of positions."""
    xp = get_array_module(array)
    phase = fourier_translation_operator(positions, array.shape, device=array.device)
    fourier_array = xp.fft.fft2(array)
    shifted_fourier_array = fourier_array * phase

    return xp.fft.ifft2(shifted_fourier_array)


def fourier_translation_operator(
    positions: "torch.Tensor | np.ndarray",
    shape: tuple | np.ndarray,
    device: "str | torch.device" = "cpu",
) -> "torch.Tensor":
    """Returns phase ramp for fourier-shifting array of shape `shape`."""

    xp = get_array_module(positions)
    nx, ny = shape[-2:]
    x = positions[..., 0][:, None, None]
    y = positions[..., 1][:, None, None]
    if xp is torch:
        kx = torch.fft.fftfreq(nx, d=1.0, device=device)
        ky = torch.fft.fftfreq(ny, d=1.0, device=device)
        ramp_x = torch.exp(-2.0j * torch.pi * kx[None, :, None] * x)
        ramp_y = torch.exp(-2.0j * torch.pi * ky[None, None, :] * y)
    else:
        assert xp in [cp, np]
        kx = xp.fft.fftfreq(nx, d=1.0)
        ky = xp.fft.fftfreq(ny, d=1.0)
        ramp_x = xp.exp(-2.0j * xp.pi * kx[None, :, None] * x)
        ramp_y = xp.exp(-2.0j * xp.pi * ky[None, None, :] * y)

    ramp = ramp_x * ramp_y
    if len(shape) == 2:
        return ramp
    elif len(shape) == 3:
        return ramp[:, None]
    else:
        raise NotImplementedError


def sum_patches_base(patches, patch_row, patch_col, obj_shape):
    """Sums overlapping patches corner-centered at `positions`."""

    flat_weights = patches.ravel()
    indices = (patch_col + patch_row * obj_shape[1]).ravel()
    counts = np.bincount(indices, weights=flat_weights, minlength=len(obj_shape))
    counts = np.reshape(counts, obj_shape)

    return counts


def sum_patches(patches, patch_row, patch_col, obj_shape):
    """Sums overlapping patches corner-centered at `positions`."""

    if np.any(np.iscomplex(patches)):
        real = sum_patches_base(patches.real, patch_row, patch_col, obj_shape)
        imag = sum_patches_base(patches.imag, patch_row, patch_col, obj_shape)
        return real + 1.0j * imag
    else:
        return sum_patches_base(patches, patch_row, patch_col, obj_shape)
