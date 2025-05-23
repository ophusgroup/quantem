from typing import Any, Self, Union, Self

import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset3d import Dataset3d

from quantem.core.utils.validators import ensure_valid_array
from quantem.core.utils.compound_validators import validate_list_of_dataset2d

from quantem.tomography.alignment import tilt_series_cross_cor_align, compute_com_tilt_series
from quantem.tomography.tilt_series_dataset import Tilt_Series

class Tomography(AutoSerialize):
    """Class for handling tomography data and operations.

    This class provides methods for aligning tilt series, computing the center of mass,
    and other tomography-related operations.

    Attributes
    ----------
    None beyond base Dataset.
    """
    
    _token = object()

    def __init__(
        self,
        tilt_series: Tilt_Series,
        recon_volume: Dataset3d | None,
        _token: object | None = None,
    ):
        """Initialize a Tomography object.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 3D array data
        name : str
            A descriptive name for the dataset
        """
        
        if _token is not self._token:
            raise RuntimeError(
                "This class is not meant to be instantiated directly. Use the from_data method."
            )
        
        self.tilt_series = tilt_series
        self.recon_volume = recon_volume
        
    @classmethod
    def from_data(
        cls,
        tilt_series: Dataset3d | NDArray,
        tilt_angles: list | NDArray = None,
        recon_volume: Dataset3d | NDArray = None,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        


        tilt_series_dataset = Tilt_Series.from_array(
            array = tilt_series,
            tilt_angles = tilt_angles,
            name = name,
            origin = origin,
            sampling = sampling,
            units = units,
            signal_units= signal_units,
        )
        
        if recon_volume is not None:
            validated_recon_volume = ensure_valid_array(recon_volume, ndim=3)
            recon_volume = Dataset3d.from_array(
                array = validated_recon_volume,
                name = name,
                origin = origin,
                sampling = sampling,
                units = units,
                signal_units= signal_units,
            )
        
        
        return cls(
            tilt_series = tilt_series_dataset,
            recon_volume = recon_volume,
            _token = cls._token,
        )
        
    # --- Preprocessing Here ---
    def align_tilt_series(self, upsample_factor: int = 1, overwrite: bool = False) -> None:
        # TODO: Should we return the predicted shifts?
        aligned_tilt_series, shifts = tilt_series_cross_cor_align(self.tilt_series.array, upsample_factor=upsample_factor)
        
        old_mean_com, old_std_com = compute_com_tilt_series(self.array)
        new_mean_com, new_std_com = compute_com_tilt_series(aligned_tilt_series)
        
        print(f"Old COM: {old_mean_com:.2f} ± {old_std_com:.2f}")
        print(f"New COM: {new_mean_com:.2f} ± {new_std_com:.2f}")
        
        print(new_mean_com/new_std_com)
        print(old_mean_com/old_std_com)
        
        if overwrite:
            self.tilt_series.array = aligned_tilt_series
        else:
            print("Set overwrite=True to overwrite the original tilt series.")
            
    
    # --- Reconstruction Algorithms ---
    
    def sirt_reconstruct(
        self,
        num_iterations: int = 10,
        reset: bool = True,
        step_size: float = 0.25,
        enforce_positivity: bool = True,
        smoothing_sigma: float | None = None,
        inline_alignment: bool = False,
    ) -> None:
        """Runs the SIRT reconstruction for the specified number of iterations.

        Parameters
        ----------
        num_iterations : int, optional
            The number of iterations to run the reconstruction (default is 10).
        reset : bool, optional
            If True, resets the reconstruction volume before starting (default is True).
        step_size : float, optional
            The step size for the reconstruction algorithm (default is 0.25).
        enforce_positivity : bool, optional
            If True, enforces positivity in the reconstruction (default is True).
        smoothing_sigma : float, optional
            The standard deviation for Gaussian smoothing (default is None).
        inline_alignment : bool, optional
            If True, performs inline alignment during reconstruction (default is False).
        """
        
        
        for a0 in 