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

def circular_mask(shape, radius, center=None, dtype=torch.float32, device='cpu'):
    """Generate a 2D circular mask of given shape and radius."""
    H, W = shape
    if center is None:
        center = (H // 2, W // 2)
    y = torch.arange(H, dtype=dtype, device=device).view(-1, 1)
    x = torch.arange(W, dtype=dtype, device=device).view(1, -1)
    dist_sq = (x - center[1])**2 + (y - center[0])**2
    return (dist_sq <= radius**2).to(dtype)

def apply_circular_masks_all_axes(volume, radii):
    """
    Apply 2D circular masks along all three axes of a 3D volume.
    
    Args:
        volume (torch.Tensor): 3D tensor of shape (H, W, D)
        radii (tuple): (r0, r1, r2) for axes 0, 1, 2
    Returns:
        masked_volume: tensor with all masks applied
    """
    H, W, D = volume.shape
    device = volume.device
    dtype = volume.dtype

    # Masks for each axis
    mask0 = circular_mask((W, D), radii[0], dtype=dtype, device=device).unsqueeze(0)      # shape (1, W, D)
    mask1 = circular_mask((H, D), radii[1], dtype=dtype, device=device).unsqueeze(1)      # shape (H, 1, D)
    mask2 = circular_mask((H, W), radii[2], dtype=dtype, device=device).unsqueeze(2)      # shape (H, W, 1)

    # Broadcast and multiply all masks together
    total_mask = mask0 * mask1 * mask2  # shape (H, W, D)

    return volume * total_mask