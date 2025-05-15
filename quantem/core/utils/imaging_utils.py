# Utilities for processing images

from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def cross_correlation_shift(
    im_ref,
    im,
    upsample_factor: int = 1,
    max_shift=None,
    return_shifted_image: bool = False,
    fft_input: bool = False,
    fft_output: bool = False,
    device: str = "cpu",
):
    """
    Estimate subpixel shift between two 2D images using Fourier cross-correlation.

    Parameters
    ----------
    im_ref : ndarray
        Reference image or its FFT if fft_input=True
    im : ndarray
        Image to align or its FFT if fft_input=True
    upsample_factor : int
        Subpixel upsampling factor (must be > 1 for subpixel accuracy)
    fft_input : bool
        If True, assumes im_ref and im are already in Fourier space
    return_shifted_image : bool
        If True, return the shifted version of `im` aligned to `im_ref`
    device : str
        'cpu' or 'gpu' (requires CuPy)

    Returns
    -------
    shifts : tuple of float
        (row_shift, col_shift) to align `im` to `im_ref`
    image_shifted : ndarray (optional)
        Shifted image in real space, only returned if return_shifted_image=True
    """
    if device == "gpu":
        import cupy as cp  # type: ignore

        xp = cp
    else:
        xp = np

    # Fourier transforms
    F_ref = im_ref if fft_input else xp.fft.fft2(im_ref)
    F_im = im if fft_input else xp.fft.fft2(im)

    # Correlation
    cc = F_ref * xp.conj(F_im)
    cc_real = xp.real(xp.fft.ifft2(cc))

    if max_shift is not None:
        x = np.fft.fftfreq(cc.shape[0], 1 / cc.shape[0])
        y = np.fft.fftfreq(cc.shape[1], 1 / cc.shape[1])
        mask = x[:, None] ** 2 + y[None, :] ** 2 >= max_shift**2
        cc_real[mask] = 0.0

    # Coarse peak
    peak = xp.unravel_index(xp.argmax(cc_real), cc_real.shape)
    x0, y0 = peak

    # Parabolic refinement
    x_inds = xp.mod(x0 + xp.arange(-1, 2), cc.shape[0]).astype(int)
    y_inds = xp.mod(y0 + xp.arange(-1, 2), cc.shape[1]).astype(int)

    vx = cc_real[x_inds, y0]
    vy = cc_real[x0, y_inds]

    def parabolic_peak(v):
        return (v[2] - v[0]) / (4 * v[1] - 2 * v[2] - 2 * v[0])

    dx = parabolic_peak(vx)
    dy = parabolic_peak(vy)

    x0 = (x0 + dx) % cc.shape[0]
    y0 = (y0 + dy) % cc.shape[1]

    if upsample_factor <= 1:
        shifts = (x0, y0)
    else:
        # Local DFT upsampling
        def dft_upsample(F, up, shift):
            M, N = F.shape
            du = np.ceil(1.5 * up).astype(int)
            row = np.arange(-du, du + 1)
            col = np.arange(-du, du + 1)
            r_shift = shift[0] - M // 2
            c_shift = shift[1] - N // 2

            kern_row = np.exp(
                -2j
                * np.pi
                / (M * up)
                * np.outer(row, xp.fft.ifftshift(xp.arange(M)) - M // 2 + r_shift)
            )
            kern_col = np.exp(
                -2j
                * np.pi
                / (N * up)
                * np.outer(xp.fft.ifftshift(xp.arange(N)) - N // 2 + c_shift, col)
            )
            return xp.real(kern_row @ F @ kern_col)

        local = dft_upsample(cc, upsample_factor, (x0, y0))
        peak = np.unravel_index(xp.argmax(local), local.shape)

        try:
            lx, ly = peak
            icc = local[lx - 1 : lx + 2, ly - 1 : ly + 2]
            if icc.shape == (3, 3):
                dxf = parabolic_peak(icc[:, 1])
                dyf = parabolic_peak(icc[1, :])
            else:
                raise ValueError("Subarray too close to edge")
        except (IndexError, ValueError):
            dxf = dyf = 0.0

        shifts = (
            np.array([x0, y0]) + (np.array(peak) - upsample_factor) / upsample_factor
        )
        shifts += np.array([dxf, dyf]) / upsample_factor

    shifts = (shifts + 0.5 * np.array(cc.shape)) % cc.shape - 0.5 * np.array(cc.shape)

    if not return_shifted_image:
        return shifts

    # Fourier shift image (F_im assumed to be FFT)
    kx = xp.fft.fftfreq(F_im.shape[0])[:, None]
    ky = xp.fft.fftfreq(F_im.shape[1])[None, :]
    phase_ramp = xp.exp(-2j * np.pi * (kx * shifts[0] + ky * shifts[1]))
    F_im_shifted = F_im * phase_ramp
    if fft_output:
        image_shifted = F_im_shifted
    else:
        image_shifted = xp.real(xp.fft.ifft2(F_im_shifted))

    return shifts, image_shifted


def bilinear_kde(
    xa: np.ndarray,
    ya: np.ndarray,
    intensities: np.ndarray,
    output_shape: Tuple[int, int],
    kde_sigma: float,
    pad_value: float = 0.0,
    threshold: float = 1e-3,
    lowpass_filter: bool = False,
    max_batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Compute a bilinear kernel density estimate (KDE) with smooth threshold masking.

    Parameters
    ----------
    xa : np.ndarray
        Vertical (row) coordinates of input points.
    ya : np.ndarray
        Horizontal (col) coordinates of input points.
    intensities : np.ndarray
        Weights for each (xa, ya) point.
    output_shape : tuple of int
        Output image shape (rows, cols).
    kde_sigma : float
        Standard deviation of Gaussian KDE smoothing.
    pad_value : float, default = 1.0
        Value to return when KDE support is too low.
    threshold : float, default = 1e-3
        Minimum counts_KDE value for trusting the output signal.
    lowpass_filter : bool, optional
        If True, apply sinc-based inverse filtering to deconvolve the kernel.
    max_batch_size : int or None, optional
        Max number of points to process in one batch.

    Returns
    -------
    np.ndarray
        The estimated KDE image with threshold-masked output.
    """
    rows, cols = output_shape
    xF = np.floor(xa.ravel()).astype(int)
    yF = np.floor(ya.ravel()).astype(int)
    dx = xa.ravel() - xF
    dy = ya.ravel() - yF
    w = intensities.ravel()

    pix_count = np.zeros(rows * cols, dtype=np.float32)
    pix_output = np.zeros(rows * cols, dtype=np.float32)

    if max_batch_size is None:
        max_batch_size = xF.shape[0]

    for start, end in generate_batches(xF.shape[0], max_batch=max_batch_size):
        for dx_off, dy_off, weight in [
            (0, 0, (1 - dx) * (1 - dy)),
            (1, 0, dx * (1 - dy)),
            (0, 1, (1 - dx) * dy),
            (1, 1, dx * dy),
        ]:
            inds = [xF[start:end] + dx_off, yF[start:end] + dy_off]
            inds_1D = np.ravel_multi_index(inds, dims=output_shape, mode="wrap")
            weights = weight[start:end]

            pix_count += np.bincount(inds_1D, weights=weights, minlength=rows * cols)
            pix_output += np.bincount(
                inds_1D, weights=weights * w[start:end], minlength=rows * cols
            )

    # Reshape to 2D and apply Gaussian KDE
    pix_count = pix_count.reshape(output_shape)
    pix_output = pix_output.reshape(output_shape)

    pix_count = gaussian_filter(pix_count, kde_sigma)
    pix_output = gaussian_filter(pix_output, kde_sigma)

    # Final image
    weight = np.minimum(pix_count / threshold, 1.0)
    image = pad_value * (1.0 - weight) + weight * (
        pix_output / np.maximum(pix_count, 1e-8)
    )

    if lowpass_filter:
        f_img = np.fft.fft2(image)
        fx = np.fft.fftfreq(rows)
        fy = np.fft.fftfreq(cols)
        f_img /= np.sinc(fx)[:, None]  # type: ignore
        f_img /= np.sinc(fy)[None, :]  # type: ignore
        image = np.real(np.fft.ifft2(f_img))

    return image


def subdivide_batches(
    num_items: int,
    num_batches: Optional[int] = None,
    max_batch: Optional[int] = None,
) -> List[int]:
    """
    Split `num_items` into a list of batch sizes.

    Parameters
    ----------
    num_items : int
        Total number of items to split into batches.
    num_batches : int, optional
        Number of desired batches. Cannot be used with `max_batch`.
    max_batch : int, optional
        Maximum batch size. Cannot be used with `num_batches`.

    Returns
    -------
    List[int]
        List of batch sizes that sum to `num_items`.
    """
    if num_batches is not None and max_batch is not None:
        raise RuntimeError("Specify only one of `num_batches` or `max_batch`.")

    if num_batches is None:
        if max_batch is None:
            raise RuntimeError("Must provide either `num_batches` or `max_batch`.")
        num_batches = (num_items + max_batch - 1) // max_batch

    if num_items < num_batches:
        raise ValueError("`num_batches` may not exceed `num_items`.")

    base_size = num_items // num_batches
    remainder = num_items % num_batches

    return [base_size + 1] * remainder + [base_size] * (num_batches - remainder)


def generate_batches(
    num_items: int,
    num_batches: Optional[int] = None,
    max_batch: Optional[int] = None,
    start_index: int = 0,
) -> Iterator[Tuple[int, int]]:
    """
    Yield (start, end) index tuples for each batch.

    Parameters
    ----------
    num_items : int
        Total number of items to batch.
    num_batches : int, optional
        Number of batches. Cannot be used with `max_batch`.
    max_batch : int, optional
        Maximum size of each batch. Cannot be used with `num_batches`.
    start_index : int, default = 0
        Optional offset to start indexing from.

    Yields
    ------
    (int, int)
        Tuple of (start, end) indices for each batch.
    """
    batch_sizes = subdivide_batches(num_items, num_batches, max_batch)
    idx = start_index
    for size in batch_sizes:
        yield idx, idx + size
        idx += size
