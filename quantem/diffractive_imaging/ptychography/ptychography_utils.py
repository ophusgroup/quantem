from typing import Tuple

import torch


def sum_overlapping_patches(
    patches: torch.Tensor,
    positions_px: torch.Tensor,
    obj_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Sums overlapping patches into a global array using scatter_add, supports complex inputs.

    Parameters
    ----------
    patches : (N, sx, sy) torch.Tensor (real or complex)
        Array of N patches to be summed.
    positions_px : (N, 2) torch.Tensor
        Integer (x, y) positions for each patch.
    object_shape : (Hx, Hy)
        Shape of the full object array.

    Returns
    -------
    summed : (Hx, Hy) torch.Tensor
        Accumulated object array.
    """

    device = patches.device
    dtype = patches.dtype

    N, sx, sy = patches.shape
    Hx, Hy = obj_shape

    x0 = positions_px[:, 0].round().to(torch.long)
    y0 = positions_px[:, 1].round().to(torch.long)

    dx = torch.fft.fftfreq(sx, d=1 / sx, device=device).to(torch.long)
    dy = torch.fft.fftfreq(sy, d=1 / sy, device=device).to(torch.long)
    dx_grid, dy_grid = torch.meshgrid(dx, dy, indexing="ij")

    x_idx = (x0[:, None, None] + dx_grid[None, :, :]) % Hx
    y_idx = (y0[:, None, None] + dy_grid[None, :, :]) % Hy

    flat_indices = x_idx * Hy + y_idx
    flat_indices = flat_indices.reshape(-1)
    flat_weights = patches.reshape(-1)

    summed = torch.zeros(Hx * Hy, dtype=dtype, device=device)
    summed = summed.scatter_add(0, flat_indices, flat_weights)

    return summed.reshape(Hx, Hy)


def fourier_translation_operator(
    positions: torch.Tensor,
    shape: tuple,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Returns phase ramp for fourier-shifting array of shape `shape`."""

    nx, ny = shape[-2:]
    x = positions[..., 0][:, None, None]
    y = positions[..., 1][:, None, None]

    kx = torch.fft.fftfreq(nx, d=1.0, device=device)
    ky = torch.fft.fftfreq(ny, d=1.0, device=device)
    ramp_x = torch.exp(-2.0j * torch.pi * kx[None, :, None] * x)
    ramp_y = torch.exp(-2.0j * torch.pi * ky[None, None, :] * y)

    ramp = ramp_x * ramp_y
    return ramp


def fourier_shift(array: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Fourier-shift array by flat array of positions"""
    phase = fourier_translation_operator(positions, array.shape, device=array.device)
    fourier_array = torch.fft.fft2(array)
    shifted_fourier_array = fourier_array * phase

    return torch.fft.ifft2(shifted_fourier_array)
