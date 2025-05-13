# Utilities for processing images

from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


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

    # Smooth transition mask around threshold
    # alpha = 1000.0  # Controls steepness of logistic transition
    # weight = 1.0 / (1.0 + np.exp(-alpha * (pix_count - threshold)))
    weight = np.minimum(pix_count / threshold, 1.0)

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(figsize=(12,12))
    # ax.imshow(
    #     pix_count,
    #     vmin = 0,
    #     vmax = 1,
    #     )
    # print(np.min(pix_count),np.max(pix_count))

    # Final image
    image = pad_value * (1.0 - weight) + weight * (
        pix_output / np.maximum(pix_count, 1e-8)
    )

    if lowpass_filter:
        f_img = np.fft.fft2(image)
        fx = np.fft.fftfreq(rows)
        fy = np.fft.fftfreq(cols)
        f_img /= np.sinc(fx)[:, None]
        f_img /= np.sinc(fy)[None, :]
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
