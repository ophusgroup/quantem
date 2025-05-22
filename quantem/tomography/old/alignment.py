from numpy.typing import NDArray
from typing import Tuple

import numpy as np

from quantem.core.utils.imaging_utils import cross_correlation_shift

from tqdm.auto import tqdm

def tilt_series_cross_cor_align(
    tilt_series: NDArray,
    upsample_factor: int = 1,
) -> Tuple[NDArray, NDArray]:
    """
    Aligns full tilt series using cross-correlation.
    """
    
    aligned_tilt_series = [tilt_series[0]]
    shifts = []
    num_imgs = tilt_series.shape[0]
    
    pbar = tqdm(range(num_imgs - 1), desc = "Cross-correlation alignment")
    
    for i in pbar:
        
        shift, aligned_img = cross_correlation_shift(
            tilt_series[i],
            tilt_series[i + 1],
            return_shifted_image=True,
        )
        
        aligned_tilt_series.append(aligned_img)
        shifts.append(shift)
        
    return np.array(aligned_tilt_series), np.array(shifts)

def compute_com_tilt_series(
    tilt_series: NDArray,

) -> NDArray:
    """
    Computes the center of mass of a tilt series.
    """
    
    x = np.arange(tilt_series.shape[1])[:, None]
    x0 = np.zeros(tilt_series.shape[0])

    for a0 in range(tilt_series.shape[0]):
        im = tilt_series[a0]
        x0[a0] = np.sum(im * x) / np.sum(im)
        
    return np.mean(x0), np.std(x0)