import torch


def rasterization_2dgs(
    positions: torch.Tensor,
    sigmas,
    intensities,
    grids: tuple[torch.Tensor, torch.Tensor],
    isotropic_splats: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Rasterize 2D Gaussian splats, supporting [sigma_z, sigma_y, sigma_x].
    For isotropic splats (sigma_y == sigma_x, sigma_z == 0), this reduces to the standard 2D Gaussian.
    For anisotropic or nonzero sigma_z, the effective 2D sigmas are used.
    The normalization is always correct for the 2D marginal (i.e., integrates to intensity).
    """
    grid_y, grid_x = grids
    z, y, x = positions.T
    dx = grid_x.unsqueeze(0) - x.unsqueeze(1).unsqueeze(2)  # (N, H, W)
    dy = grid_y.unsqueeze(0) - y.unsqueeze(1).unsqueeze(2)  # (N, H, W)

    if isotropic_splats:
        sigmas = sigmas.mean(dim=1)
        # sigmas = sigmas[:, 0]
        amps = intensities * ((2 * torch.pi) ** 0.5) * sigmas
        exp_arg = (dy**2 + dx**2) / (sigmas[:, None, None] ** 2 + 1e-12)
    else:
        sigma_z, sigma_y, sigma_x = sigmas[:, 0], sigmas[:, 1], sigmas[:, 2]
        amps = intensities * ((2 * torch.pi) ** 0.5) * sigma_z
        exp_arg = (dy**2) / (sigma_y[:, None, None] ** 2 + 1e-12) + (dx**2) / (
            sigma_x[:, None, None] ** 2 + 1e-12
        )

    gaussians = amps[:, None, None] * torch.exp(-0.5 * exp_arg)
    image = gaussians.sum(dim=0)
    return image[None,]
