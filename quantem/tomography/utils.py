import torch
import numpy as np

def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.) -> torch.Tensor:
    radius = np.ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = torch.distributions.Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())

def gaussian_filter_2d(img: torch.Tensor, sigma: float, kernel_1d: torch.Tensor) -> torch.Tensor: #Add kernel_1d as an argument
    # kernel_1d = gaussian_kernel_1d(sigma)  # Create 1D Gaussian kernel - Moved outside function
    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    img = img.unsqueeze(0).unsqueeze_(0)  # Make copy, make 4D for ``conv2d()``
    # Convolve along columns and rows
    img = torch.nn.functional.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    img = torch.nn.functional.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    return img.squeeze_(0).squeeze_(0)  # Make 2D again


def gaussian_filter_2d_stack(stack: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D Gaussian blur to each slice stack[:, i, :] in a vectorized way.

    Args:
        stack (torch.Tensor): Tensor of shape (H, N, W) where N is num_sinograms
        kernel_1d (torch.Tensor): 1D Gaussian kernel

    Returns:
        torch.Tensor: Blurred stack of same shape (H, N, W)
    """
    H, N, W = stack.shape
    padding = len(kernel_1d) // 2

    # Reshape to (N, 1, H, W) for conv2d
    stack_reshaped = stack.permute(1, 0, 2).unsqueeze(1)  # (N, 1, H, W)

    # Apply separable conv2d: vertical then horizontal
    out = torch.nn.functional.conv2d(stack_reshaped, kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    out = torch.nn.functional.conv2d(out, kernel_1d.view(1, 1, 1, -1), padding=(0, padding))

    # Restore shape to (H, N, W)
    return out.squeeze(1).permute(1, 0, 2)

# Circular mask

def torch_phase_cross_correlation(im1, im2):
    f1 = torch.fft.fft2(im1)
    f2 = torch.fft.fft2(im2)
    cc = torch.fft.ifft2(f1 * torch.conj(f2))
    cc_abs = torch.abs(cc)

    max_idx = torch.argmax(cc_abs)
    shifts = torch.tensor(np.unravel_index(max_idx.item(), im1.shape), device=im1.device).float()

    for i, dim in enumerate(im1.shape):
        if shifts[i] > dim // 2:
            shifts[i] -= dim

    # return shifts.flip(0)  # (dx, dy)
    return shifts