import torch
import torch.nn.functional as F


def random_quaternion(
    size: tuple, device: str | torch.device = "cpu", generator: torch.Generator | None = None
) -> torch.Tensor:
    """Generate random unit quaternions for uniform random rotations using Shoemake's method."""
    u = torch.rand((*size, 3), device=device, generator=generator)
    u1, u2, u3 = u.unbind(-1)
    s1, s2 = torch.sqrt(u1), torch.sqrt(1 - u1)
    t2, t3 = 2 * torch.pi * u2, 2 * torch.pi * u3
    return torch.stack(
        [s2 * torch.sin(t2), s2 * torch.cos(t2), s1 * torch.sin(t3), s1 * torch.cos(t3)], dim=-1
    )


def quaternion_to_2d_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Extract 2D rotation angle around Z-axis from quaternions."""
    q = F.normalize(quaternion, dim=-1)
    return 2 * torch.atan2(q[..., 3], q[..., 0])  # 2 * atan2(z, w)


def quaternion_to_rotation_matrix_2d(
    quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternions to 2D rotation matrix components."""
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    return (1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * y + z * w), 1 - 2 * (x**2 + z**2))


def rasterization_2dgs(
    positions: torch.Tensor,
    sigmas: torch.Tensor,
    intensities: torch.Tensor,
    grids: tuple[torch.Tensor, torch.Tensor],
    isotropic_splats: bool = True,
    quaternions: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Rasterize 2D Gaussian splats, supporting [sigma_z, sigma_y, sigma_x].
    For isotropic splats (sigma_y == sigma_x, sigma_z == 0), this reduces to the standard 2D Gaussian.
    For anisotropic or nonzero sigma_z, the effective 2D sigmas are used.
    The normalization is always correct for the 2D marginal (i.e., integrates to intensity).

    If quaternions are provided for anisotropic splats, rotation will be applied.
    Quaternion format: [w, x, y, z] where w is the scalar part and [x,y,z] is the vector part.
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

        if quaternions is not None:
            # Use quaternions to apply rotation to the Gaussian ellipse
            # Convert quaternions to 2D rotation matrix components
            r00, r01, r10, r11 = quaternion_to_rotation_matrix_2d(quaternions)

            # Apply rotation to each point
            dx_rot = r00[:, None, None] * dx + r01[:, None, None] * dy
            dy_rot = r10[:, None, None] * dx + r11[:, None, None] * dy

            # Compute the Gaussian exponent using the rotated coordinates
            exp_arg = (dx_rot**2) / (sigma_x[:, None, None] ** 2 + 1e-12) + (dy_rot**2) / (
                sigma_y[:, None, None] ** 2 + 1e-12
            )
        else:
            # Without rotation, just use the standard formula
            exp_arg = (dy**2) / (sigma_y[:, None, None] ** 2 + 1e-12) + (dx**2) / (
                sigma_x[:, None, None] ** 2 + 1e-12
            )

    gaussians = amps[:, None, None] * torch.exp(-0.5 * exp_arg)
    image = gaussians.sum(dim=0)
    return image[None,]
