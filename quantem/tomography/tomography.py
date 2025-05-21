from typing import Any, Self, Union

import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset3d import Dataset3d

from quantem.core.utils.validators import ensure_valid_array, validate_volumetric_data
from quantem.core.utils.compound_validators import validate_list_of_dataset2d

from quantem.tomography.alignment import tilt_series_cross_cor_align, compute_com_tilt_series
from quantem.tomography.dataset_tomo import Tilt_Series

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
    ) -> Tilt_Series:
        
        # validated_tilt_series = validate_list_of_dataset2d(tilt_series) # Perhaps I don't need this
        validated_tilt_series = ensure_valid_array(tilt_series, ndim=3)
        if tilt_angles is not None:
            validated_tilt_angles = ensure_valid_array(tilt_angles)     
        if recon_volume is not None:
            # TODO: Not implemented yet
            validated_recon_volume = validate_volumetric_data(recon_volume)
        else:
            validated_recon_volume = None

        tilt_series_dataset = Tilt_Series.from_array(
            array = validated_tilt_series,
            tilt_angles = validated_tilt_angles,
            name = name,
            origin = origin,
            sampling = sampling,
            units = units,
            signal_units= signal_units,
        )
        
        return cls(
            tilt_series = tilt_series_dataset,
            recon_volume = validated_recon_volume,
            _token = cls._token,
        )
        
    # --- Rough cross-correlation alignment of raw tilt series ---
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
            
        