from typing import Optional, Tuple

import torch

from quantem.core.utils.scattering_utils import electron_wavelength_angstrom


def return_patch_indices(
    positions_px: torch.Tensor,
    roi_shape: Tuple[int, int],
    obj_shape: Tuple[int, int],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute wrapped patch indices into the object array for each probe position.
    Note this assumes corner-centered probes.

    Parameters
    ----------
    positions_px : torch.Tensor
        Tensor of shape (N, 2), float32. Probe positions in pixels.
    roi_shape : tuple of int
        (Sx, Sy), the shape of the probe (patch).
    obj_shape : tuple of int
        (Hx, Hy), the shape of the object array.
    device : torch.device, optional
        Where to place tensors. If None, inferred from positions_px.

    Returns
    -------
    row : torch.Tensor
        (N, Sx, Sy), int64 tensor of row indices.
    col : torch.Tensor
        (N, Sx, Sy), int64 tensor of col indices.
    """
    if device is None:
        device = positions_px.device

    # Round and convert to int
    x0: torch.Tensor = torch.round(positions_px[:, 0]).to(torch.int64)
    y0: torch.Tensor = torch.round(positions_px[:, 1]).to(torch.int64)

    # Frequency-based index grid
    x_ind: torch.Tensor = torch.fft.fftfreq(
        roi_shape[0], d=1.0 / roi_shape[0], device=device
    ).to(torch.int64)
    y_ind: torch.Tensor = torch.fft.fftfreq(
        roi_shape[1], d=1.0 / roi_shape[1], device=device
    ).to(torch.int64)

    # Broadcast and wrap
    row: torch.Tensor = (x0[:, None, None] + x_ind[None, :, None]) % obj_shape[0]
    col: torch.Tensor = (y0[:, None, None] + y_ind[None, None, :]) % obj_shape[1]

    return row, col


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


def fourier_convolve_array(array: torch.Tensor, fourier_kernel: torch.Tensor):
    """ """
    return torch.fft.ifft2(torch.fft.fft2(array) * fourier_kernel)


def fourier_shift(array: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Fourier-shift array by flat array of positions"""
    phase = fourier_translation_operator(positions, array.shape, device=array.device)
    return fourier_convolve_array(array, phase)


def compute_propagator_array(
    energy: float,
    gpts: Tuple[int, int],
    sampling: Tuple[float, float],
    slice_thickness: float,
) -> torch.Tensor:
    """ " """
    kx = torch.fft.fftfreq(gpts[0], sampling[0])
    ky = torch.fft.fftfreq(gpts[1], sampling[1])

    wavelength = electron_wavelength_angstrom(energy)
    propagator = torch.exp(
        -1.0j * (torch.square(kx)[:, None] * torch.pi * wavelength * slice_thickness)
    ) * torch.exp(
        -1.0j * (torch.square(ky)[None, :] * torch.pi * wavelength * slice_thickness)
    )
    return propagator
