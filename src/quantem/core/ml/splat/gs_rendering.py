import torch


def rasterization_2dgs(
    positions: torch.Tensor,
    sigmas,
    intensities,
    grids: tuple[torch.Tensor, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    """
    TODO - future break this into two functions, 1) projection and 2) rasterization
    """

    grid_y, grid_x = grids

    z, y, x = positions.T

    dx = grid_x.unsqueeze(0) - x.unsqueeze(1).unsqueeze(2)  # Shape: (num_gaussians, H, W)
    dy = grid_y.unsqueeze(0) - y.unsqueeze(1).unsqueeze(2)  # Shape: (num_gaussians, H, W)
    dist_squared = dx**2 + dy**2  # Shape: (num_gaussians, H, W)

    # Compute Gaussian contributions
    amps = intensities * ((2 * torch.pi) ** 0.5) * sigmas
    gaussians = amps[:, None, None] * torch.exp(-dist_squared / (2 * sigmas[:, None, None] ** 2))

    image = gaussians.sum(dim=0)  # Sum over all Gaussians, Shape: (H, W)

    return image[None,]
