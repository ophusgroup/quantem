"""
A fully torch based implementation of the Radon transform based on compatible with CPU and GPU.

Implementation using scikit-image's radon and iradon functions.
Reference: van der Walt, S., et al. (2014). scikit-image: image processing in Python. PeerJ, 2, e453.
"""

import torch
import torch.nn.functional as F


def radon_torch(image, theta=None, device=None):
    """
    Radon transform implemented in PyTorch.

    Parameters
    ----------
    image : torch.Tensor
        2D tensor representing the grayscale input image.
    theta : array-like
        Projection angles in degrees. Default is torch.arange(180).
    circle : bool
        If True, restricts the transform to the inscribed circle.
    preserve_range : bool
        Included for compatibility; no-op in this version.
    device : torch.device, optional
        Device to perform computation on.

    Returns
    -------
    radon_image : torch.Tensor
        Sinogram of shape [N_pixels, N_angles]
    """
    if device is None:
        device = image.device

    if image.ndim != 2:
        raise ValueError("Input must be a 2D image.")

    if theta is None:
        theta = torch.arange(180)
    # theta = torch.tensor(theta, dtype=torch.float32, device=device)

    H, W = image.shape

    shape_min = min(H, W)
    radius = shape_min // 2
    center = torch.tensor([H // 2, W // 2], device=device)

    Y, X = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    dist2 = (X - center[1]) ** 2 + (Y - center[0]) ** 2
    image = image.clone()
    image[dist2 > radius**2] = 0

    # Crop to square
    excess = torch.tensor([H, W], device=device) - shape_min
    slices = tuple(
        slice(int((e.item() + 1) // 2), int((e.item() + 1) // 2 + shape_min))
        if e > 0
        else slice(None)
        for e in excess
    )
    image = image[slices]

    N = image.shape[0]
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image must be square after padding or cropping.")
    center = N // 2
    radon_image = torch.zeros((len(theta), N), dtype=image.dtype, device=device)

    # Create grid for affine grid sample
    grid_y, grid_x = torch.meshgrid(
        torch.arange(N, dtype=torch.float32, device=device),
        torch.arange(N, dtype=torch.float32, device=device),
        indexing="ij",
    )
    coords = torch.stack((grid_x - center, grid_y - center), dim=-1)  # shape (N, N, 2)

    for i, angle in enumerate(theta):
        angle_rad = torch.deg2rad(angle)
        rot_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [-torch.sin(angle_rad), -torch.cos(angle_rad)],
            ],
            device=device,
        )

        rotated_coords = coords @ rot_matrix.T  # shape (N, N, 2)
        rotated_coords += center

        # Normalize coordinates to [-1, 1] for grid_sample
        grid = 2 * rotated_coords / (N - 1) - 1
        grid = grid.unsqueeze(0)  # Add batch dimension

        image_batch = image.unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(
            image_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        projection = sampled.squeeze().sum(0)
        radon_image[i, :] = projection

    return radon_image


def iradon_torch(
    sinogram,
    theta=None,
    output_size=None,
    filter_name="ramp",
    circle=True,
    device=None,
):
    """
    Inverse Radon transform (filtered backprojection) using PyTorch.
    sinogram: shape [N_angles, N_pixels]
    """

    if sinogram.ndim != 2:
        raise ValueError("Input sinogram must be 2D")

    if theta is None:
        theta = torch.linspace(0, 180, steps=sinogram.shape[0], device=device)

    num_angles, N = sinogram.shape
    if theta.shape[0] != num_angles:
        raise ValueError("theta does not match number of projections")

    if output_size is None:
        output_size = N if circle else int(torch.floor(torch.sqrt(torch.tensor(N) ** 2 / 2.0)))

    device = sinogram.device if device is None else device
    sinogram = sinogram.to(dtype=torch.float32, device=device)

    # Padding for FFT
    padded_size = max(
        64, int(2 ** torch.ceil(torch.log2(torch.tensor(2 * N, dtype=torch.float32))))
    )
    pad_y = padded_size - N
    padded = F.pad(sinogram, (0, pad_y))  # pad vertically

    # Apply Fourier filter
    f_filter = get_fourier_filter_torch(padded_size, filter_name, device=device)
    spectrum = torch.fft.fft(padded, dim=1)  # Changed from dim=0 to dim=1
    filtered = torch.real(torch.fft.ifft(spectrum * f_filter, dim=1))[
        :, :N
    ]  # Changed from dim=0 to dim=1

    # Reconstruct by backprojection
    recon = torch.zeros((output_size, output_size), dtype=filtered.dtype, device=device)
    radius = output_size // 2
    y, x = torch.meshgrid(
        torch.arange(output_size, device=device) - radius,
        torch.arange(output_size, device=device) - radius,
        indexing="ij",
    )
    x = x.flatten()
    y = y.flatten()

    for i, angle in enumerate(torch.deg2rad(theta)):
        t = (x * torch.cos(angle) - y * torch.sin(angle)).reshape(output_size, output_size)
        t_idx = t + (N // 2)

        # Linear interpolation
        t0 = torch.floor(t_idx).long().clamp(0, N - 2)
        t1 = t0 + 1
        w = t_idx - t0.float()
        val0 = filtered[i, t0]  # Changed indexing to match new dimensions
        val1 = filtered[i, t1]  # Changed indexing to match new dimensions
        proj = (1 - w) * val0 + w * val1
        recon += proj

    if circle:
        mask = (
            x.reshape(output_size, output_size) ** 2 + y.reshape(output_size, output_size) ** 2
        ) > radius**2
        recon[mask] = 0.0

    return recon * torch.pi / (2 * num_angles)


def get_fourier_filter_torch(size, filter_name="ramp", device=None, dtype=torch.float32):
    """
    Construct the Fourier filter in PyTorch.
    """
    if size % 2 != 0:
        raise ValueError("Filter size must be even")

    n = torch.cat(
        [
            torch.arange(1, size // 2 + 1, 2, device=device),
            torch.arange(size // 2 - 1, 0, -2, device=device),
        ]
    )
    f = torch.zeros(size, device=device, dtype=dtype)
    f[0] = 0.25
    f[1::2] = -1.0 / (torch.pi * n.float()) ** 2

    fourier_filter = 2 * torch.real(torch.fft.fft(f))

    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        omega = torch.pi * torch.fft.fftfreq(size, device=device)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega
    elif filter_name == "cosine":
        freq = torch.linspace(0, torch.pi, steps=size, device=device)
        fourier_filter *= torch.fft.fftshift(torch.sin(freq))
    elif filter_name == "hamming":
        hamming = torch.hamming_window(size, periodic=False, dtype=dtype, device=device)
        fourier_filter *= torch.fft.fftshift(hamming)
    elif filter_name == "hann":
        hann = torch.hann_window(size, periodic=False, dtype=dtype, device=device)
        fourier_filter *= torch.fft.fftshift(hann)
    elif filter_name is None:
        fourier_filter[:] = 1.0
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    # Reshape filter for broadcasting with sinogram
    return fourier_filter.unsqueeze(0)  # Shape: [1, size] for broadcasting with [num_angles, size]
