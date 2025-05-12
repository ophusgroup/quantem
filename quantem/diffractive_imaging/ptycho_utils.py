from typing import TYPE_CHECKING, Any, Generator, Literal

import numpy as np
from scipy.optimize import curve_fit

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
    array: "torch.Tensor | np.ndarray | cp.ndarray",
    positions: "torch.Tensor | np.ndarray | cp.ndarray",
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
    nh, nw = shape[-2:]
    h = positions[..., 0][:, None, None]
    w = positions[..., 1][:, None, None]
    if xp is torch:
        kx = torch.fft.fftfreq(nh, d=1.0, device=device)
        ky = torch.fft.fftfreq(nw, d=1.0, device=device)
        ramp_x = torch.exp(-2.0j * torch.pi * kx[None, :, None] * h)
        ramp_y = torch.exp(-2.0j * torch.pi * ky[None, None, :] * w)
    else:
        assert xp in [cp, np]
        kx = xp.fft.fftfreq(nh, d=1.0)
        ky = xp.fft.fftfreq(nw, d=1.0)
        ramp_x = xp.exp(-2.0j * xp.pi * kx[None, :, None] * h)
        ramp_y = xp.exp(-2.0j * xp.pi * ky[None, None, :] * w)

    ramp = ramp_x * ramp_y
    if len(shape) == 2:
        return ramp
    elif len(shape) == 3:
        return ramp[:, None]
    else:
        raise NotImplementedError


def sum_patches_base(
    patches: np.ndarray, patch_row: np.ndarray, patch_col: np.ndarray, obj_shape: tuple
):
    """Sums overlapping patches corner-centered at `positions`."""

    flat_weights = patches.ravel()
    indices = (patch_col + patch_row * obj_shape[1]).ravel()
    counts = np.bincount(indices, weights=flat_weights, minlength=np.prod(obj_shape))
    counts = np.reshape(counts, obj_shape)

    return counts


def sum_patches(
    patches: np.ndarray, patch_row: np.ndarray, patch_col: np.ndarray, obj_shape: tuple
):
    """Sums overlapping patches corner-centered at `positions`."""
    if np.any(np.iscomplex(patches)):
        real = sum_patches_base(patches.real, patch_row, patch_col, obj_shape)
        imag = sum_patches_base(patches.imag, patch_row, patch_col, obj_shape)
        return real + 1.0j * imag
    else:
        return sum_patches_base(patches, patch_row, patch_col, obj_shape)


def get_shifted_array(ar, hshift, wshift, periodic=True, bilinear=False, device="cpu"):
    """
        Shifts array ar by the shift vector (hshift,wshift), using the either
    the Fourier shift theorem (i.e. with sinc interpolation), or bilinear
    resampling. Boundary conditions can be periodic or not.

    Args:
            ar (float): input array
            hshift (float): shift along axis 0 (x) in pixels
            wshift (float): shift along axis 1 (y) in pixels
            periodic (bool): flag for periodic boundary conditions
            bilinear (bool): flag for bilinear image shifts
            device(str): calculation device will be perfomed on. Must be 'cpu' or 'gpu'
        Returns:
            (array) the shifted array
    """
    if device == "gpu":
        xp = cp
    else:
        xp = np

    ar = xp.asarray(ar)

    # Apply image shift
    if bilinear is False:
        nh, nw = xp.shape(ar)
        qh, qw = make_Fourier_coords2D(nh, nw, 1)
        qh = xp.asarray(qh)
        qw = xp.asarray(qw)

        p = xp.exp(-(2j * xp.pi) * ((wshift * qw) + (hshift * qh)))
        shifted_ar = xp.real(xp.fft.ifft2((xp.fft.fft2(ar)) * p))

    else:
        xF = xp.floor(hshift).astype(int).item()
        yF = xp.floor(wshift).astype(int).item()
        wx = hshift - xF
        wy = wshift - yF

        shifted_ar = (
            xp.roll(ar, (xF, yF), axis=(0, 1)) * ((1 - wx) * (1 - wy))
            + xp.roll(ar, (xF + 1, yF), axis=(0, 1)) * ((wx) * (1 - wy))
            + xp.roll(ar, (xF, yF + 1), axis=(0, 1)) * ((1 - wx) * (wy))
            + xp.roll(ar, (xF + 1, yF + 1), axis=(0, 1)) * ((wx) * (wy))
        )

    if periodic is False:
        # Rounded coordinates for boundaries
        xR = (xp.round(hshift)).astype(int)
        yR = (xp.round(wshift)).astype(int)

        if xR > 0:
            shifted_ar[0:xR, :] = 0
        elif xR < 0:
            shifted_ar[xR:, :] = 0
        if yR > 0:
            shifted_ar[:, 0:yR] = 0
        elif yR < 0:
            shifted_ar[:, yR:] = 0

    return shifted_ar


def make_Fourier_coords2D(Nx: int, Ny: int, pixelSize: float | tuple[float, float] = 1):
    """
    Generates Fourier coordinates for a (Nx,Ny)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
    """
    if isinstance(pixelSize, (tuple, list)):
        assert len(pixelSize) == 2, "pixelSize must either be a scalar or have length 2"
        pixelSize_x = pixelSize[0]
        pixelSize_y = pixelSize[1]
    else:
        pixelSize_x = pixelSize
        pixelSize_y = pixelSize

    qh = np.fft.fftfreq(Nx, pixelSize_x)
    qw = np.fft.fftfreq(Ny, pixelSize_y)
    qw, qh = np.meshgrid(qw, qh)
    return qh, qw


######## Fitting


def _plane(xy, mx, my, b):
    return mx * xy[0] + my * xy[1] + b


def _parabola(xy, c0, cx1, cx2, cy1, cy2, cxy):
    return (
        c0
        + cx1 * xy[0]
        + cy1 * xy[1]
        + cx2 * xy[0] ** 2
        + cy2 * xy[1] ** 2
        + cxy * xy[0] * xy[1]
    )


def _bezier_two(xy, c00, c01, c02, c10, c11, c12, c20, c21, c22):
    return (
        c00 * ((1 - xy[0]) ** 2) * ((1 - xy[1]) ** 2)
        + c10 * 2 * (1 - xy[0]) * xy[0] * ((1 - xy[1]) ** 2)
        + c20 * (xy[0] ** 2) * ((1 - xy[1]) ** 2)
        + c01 * 2 * ((1 - xy[0]) ** 2) * (1 - xy[1]) * xy[1]
        + c11 * 4 * (1 - xy[0]) * xy[0] * (1 - xy[1]) * xy[1]
        + c21 * 2 * (xy[0] ** 2) * (1 - xy[1]) * xy[1]
        + c02 * ((1 - xy[0]) ** 2) * (xy[1] ** 2)
        + c12 * 2 * (1 - xy[0]) * xy[0] * (xy[1] ** 2)
        + c22 * (xy[0] ** 2) * (xy[1] ** 2)
    )


# TODO -- testing this
def fit_origin(
    data: np.ndarray | tuple[np.ndarray, np.ndarray],
    mask: np.ndarray | None = None,
    fit_function: Literal["plane", "parabola", "bezier_two", "constant"] = "plane",
    robust=False,
    robust_steps=3,
    robust_thresh=2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the origin of diffraction space using the specified method."""

    qh0_meas, qw0_meas = data

    if fit_function == "plane":
        f = _plane
    elif fit_function == "parabola":
        f = _parabola
    elif fit_function == "bezier_two":
        f = _bezier_two
    elif fit_function == "constant":
        qh0_fit = np.mean(qh0_meas) * np.ones_like(qh0_meas)
        qw0_fit = np.mean(qw0_meas) * np.ones_like(qw0_meas)
        qh0_residuals = qh0_meas - qh0_fit
        qw0_residuals = qw0_meas - qw0_fit
        return qh0_fit, qw0_fit, qh0_residuals, qw0_residuals
    else:
        raise ValueError(
            "fit_function must be one of 'plane', 'parabola', 'bezier_two', 'constant'"
        )
    shape = qh0_meas.shape
    h, w = np.indices(shape)
    h1D = h.reshape(1, np.prod(shape))
    w1D = w.reshape(1, np.prod(shape))
    hw = np.vstack((h1D, w1D))

    if mask is not None:
        qh0_meas_masked = qh0_meas[mask]
        qw0_meas_masked = qw0_meas[mask]
        mask1D = mask.reshape(1, np.prod(shape))
        hw_masked = np.vstack((h1D * mask1D, w1D * mask1D))

        popt_x, _ = curve_fit(f, hw_masked, qh0_meas_masked)
        popt_y, _ = curve_fit(f, hw_masked, qw0_meas_masked)

        if robust:
            popt_x = perform_robust_fitting(
                f, hw_masked, qh0_meas_masked, popt_x, robust_steps, robust_thresh
            )
            popt_y = perform_robust_fitting(
                f, hw_masked, qw0_meas_masked, popt_y, robust_steps, robust_thresh
            )
    else:
        popt_x, _ = curve_fit(f, hw, qh0_meas)
        popt_y, _ = curve_fit(f, hw, qw0_meas)

        if robust:
            popt_x = perform_robust_fitting(
                f, hw, qh0_meas, popt_x, robust_steps, robust_thresh
            )
            popt_y = perform_robust_fitting(
                f, hw, qw0_meas, popt_y, robust_steps, robust_thresh
            )

    qh0_fit = f(hw, *popt_x).reshape(shape)
    qw0_fit = f(hw, *popt_y).reshape(shape)
    qh0_residuals = qh0_meas - qh0_fit
    qw0_residuals = qw0_meas - qw0_fit

    return qh0_fit, qw0_fit, qh0_residuals, qw0_residuals


def perform_robust_fitting(func, hw, data, initial_guess, robust_steps, robust_thresh):
    """Performs robust fitting by iteratively rejecting outliers."""
    popt = initial_guess
    for k in range(robust_steps):
        fit_vals = func(hw, *popt)
        rmse = np.sqrt(np.mean((fit_vals - data) ** 2))
        mask = np.abs(fit_vals - data) <= robust_thresh * rmse
        hw = np.vstack((hw[0][mask], hw[1][mask]))
        data = data[mask]
        popt, _ = curve_fit(func, hw, data, p0=popt)
    return popt


class AffineTransform:
    """
    Affine Transform Class.

    Simplified version of AffineTransform from tike:
    https://github.com/AdvancedPhotonSource/tike/blob/f9004a32fda5e49fa63b987e9ffe3c8447d59950/src/tike/ptycho/position.py

    AffineTransform() -> Identity

    Parameters
    ----------
    scale0: float
        x-scaling
    scale1: float
        y-scaling
    shear1: float
        \\gamma shear
    angle: float
        \\theta rotation angle
    t0: float
        x-translation
    t1: float
        y-translation
    dilation: float
        Isotropic expansion (multiplies scale0 and scale1)
    """

    def __init__(
        self,
        scale0: float = 1.0,
        scale1: float = 1.0,
        shear1: float = 0.0,
        angle: float = 0.0,
        t0: float = 0.0,
        t1: float = 0.0,
        dilation: float = 1.0,
    ):
        self.scale0 = scale0 * dilation
        self.scale1 = scale1 * dilation
        self.shear1 = shear1
        self.angle = angle
        self.t0 = t0
        self.t1 = t1

    @classmethod
    def from_array(cls, T: np.ndarray):
        """
        Return an Affine Transfrom from a 2x2 matrix.
        Use decomposition method from Graphics Gems 2 Section 7.1
        """
        R = T[:2, :2].copy()
        scale0 = np.linalg.norm(R[0])
        if scale0 <= 0:
            return cls()
        R[0] /= scale0
        shear1 = R[0] @ R[1]
        R[1] -= shear1 * R[0]
        scale1 = np.linalg.norm(R[1])
        if scale1 <= 0:
            return cls()
        R[1] /= scale1
        shear1 /= scale1
        angle = np.arccos(R[0, 0])

        if T.shape[0] > 2:
            t0, t1 = T[2]
        else:
            t0 = t1 = 0.0

        return cls(
            scale0=float(scale0),
            scale1=float(scale1),
            shear1=float(shear1),
            angle=float(angle),
            t0=t0,
            t1=t1,
        )

    def asarray(self):
        """
        Return an 2x2 matrix of scale, shear, rotation.
        This matrix is scale @ shear @ rotate from left to right.
        """
        cosx = np.cos(self.angle)
        sinx = np.sin(self.angle)
        return (
            np.array(
                [
                    [self.scale0, 0.0],
                    [0.0, self.scale1],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [1.0, 0.0],
                    [self.shear1, 1.0],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [+cosx, -sinx],
                    [+sinx, +cosx],
                ],
                dtype="float32",
            )
        )

    def asarray3(self):
        """
        Return an 3x2 matrix of scale, shear, rotation, translation.
        This matrix is scale @ shear @ rotate from left to right.
        Expects a homogenous (z) coordinate of 1.
        """
        T = np.empty((3, 2), dtype="float32")
        T[2] = (self.t0, self.t1)
        T[:2, :2] = self.asarray()
        return T

    def astuple(self):
        """Return the constructor parameters in a tuple."""
        return (
            self.scale0,
            self.scale1,
            self.shear1,
            self.angle,
            self.t0,
            self.t1,
        )

    def __call__(self, x: np.ndarray, origin=(0, 0), xp=np) -> np.ndarray:
        origin = xp.asarray(origin, dtype=xp.float32)
        tf_matrix = self.asarray()
        tf_matrix = xp.asarray(tf_matrix, dtype=xp.float32)
        tf_translation = xp.array((self.t0, self.t1)) + origin
        return ((x - origin) @ tf_matrix) + tf_translation

    def __str__(self):
        return (
            "AffineTransform( \n"
            f"  scale0 = {self.scale0:.4f}, scale1 = {self.scale1:.4f}, \n"
            f"  shear1 = {self.shear1:.4f}, angle = {self.angle:.4f}, \n"
            f"  t0 = {self.t0:.4f}, t1 = {self.t1:.4f}, \n"
            ")"
        )

    def __repr__(self):
        return (
            "AffineTransform( \n"
            f"  scale0 = {self.scale0:.4f}, scale1 = {self.scale1:.4f}, \n"
            f"  shear1 = {self.shear1:.4f}, angle = {self.angle:.4f}, \n"
            f"  t0 = {self.t0:.4f}, t1 = {self.t1:.4f}, \n"
            ")"
        )


def center_crop_arr(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """
    Crop an array to a given shape, centered along all axes.

    Parameters
    ----------
    arr : np.ndarray
        The input n-dimensional array to be cropped.
    shape : tuple[int, ...]
        The desired output shape. Must have the same number of dimensions as arr,
        and each dimension must be less than or equal to the corresponding dimension of arr.
    """
    if len(shape) != arr.ndim:
        raise ValueError(
            f"Shape must have the same number of dimensions as arr. "
            f"Got shape with {len(shape)} dimensions and arr with {arr.ndim} dimensions."
        )

    for i, (s, a) in enumerate(zip(shape, arr.shape)):
        if s > a:
            raise ValueError(
                f"Dimension {i} of shape ({s}) is larger than dimension {i} of arr ({a})."
            )

    slices = []
    for i, (s, a) in enumerate(zip(shape, arr.shape)):
        start = (a - s) // 2
        end = start + s
        slices.append(slice(start, end))

    # Return the cropped array
    return arr[tuple(slices)]
