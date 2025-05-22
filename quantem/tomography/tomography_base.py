from typing import Any, Self, Union, Self

import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset3d import Dataset3d

from quantem.core.utils.validators import ensure_valid_array
from quantem.core.utils.compound_validators import validate_list_of_dataset2d

from quantem.tomography.tilt_series_dataset import TiltSeries

class TomographyBase(AutoSerialize):
    
    _token = object()
    
    
    def __init__(
        self,
        tilt_series: Dataset3d,
        recon_volume: Dataset3d | None,
        device: str = "cuda",
        # ABF/HAADF property
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
        
        self._device = device
        self._tilt_series = tilt_series
        self._recon_volume = recon_volume
        self._loss = []
        self._mode = []
        
        
        
    @classmethod
    def from_tilt_series(
        cls,
        device: str,
        tilt_series: NDArray | Dataset3d| Any,
        tilt_angles: list | NDArray = None,
        recon_volume: NDArray | Dataset3d| None = None,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ):
        
        
        device = device.lower()
        if "cuda" not in device or "gpu" not in device:
            raise NotImplementedError("Tomography not currently supported on CPU.")
        
        tilt_series = TiltSeries.from_array(
            array = tilt_series,
            tilt_angles = tilt_angles,
            name = name,
            origin = origin,
            sampling = sampling,
            units = units,
            signal_units = signal_units,
        )
        
        if recon_volume is not None:
            if not isinstance(recon_volume, Dataset3d):
                recon_volume = Dataset3d.from_array(
                    array = recon_volume,
                    name = name,
                    origin = origin,
                    sampling = sampling,
                    units = units,
                    signal_units = signal_units,
                )
        else:
            recon_volume = None
            
        return cls(
            tilt_series = tilt_series,
            recon_volume = recon_volume,
            device = device,
            _token = cls._token,
        )
        
    # --- Properties ---
    @property
    def tilt_series(self) -> TiltSeries:
        """Tilt series dataset."""
        
        return self._tilt_series
    
    
    @tilt_series.setter
    def tilt_series(
        self, 
        tilt_series: Dataset3d| NDArray | TiltSeries,
        tilt_angles: list | NDArray = None,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ):
        """Set the tilt series dataset."""
        
        if not isinstance(tilt_series, TiltSeries):
            
            tilt_series = TiltSeries.from_array(
                array = tilt_series,
                tilt_angles = tilt_angles,
                name = name,
                origin = origin,
                sampling = sampling,
                units = units,
                signal_units = signal_units,
            )
        
        self._tilt_series = tilt_series
        
    @property
    def recon_volume(self) -> Dataset3d | None:
        """Reconstruction volume dataset."""
        
        return self._recon_volume
    @recon_volume.setter
    def recon_volume(self, recon_volume: Dataset3d | NDArray):
        """Set the reconstruction volume dataset."""
        
        if not isinstance(recon_volume, Dataset3d):
            recon_volume = Dataset3d.from_array(
                array = recon_volume,
                name = self._tilt_series.name,
                origin = self._tilt_series.origin,
                sampling = self._tilt_series.sampling,
                units = self._tilt_series.units,
                signal_units = self._tilt_series.signal_units,
            )
        
        self._recon_volume = recon_volume
        
    @property
    def device(self) -> str:
        """Computation device."""
        
        return self._device
    @device.setter
    def device(self, device: str):
        """Set the computation device."""
        
        if "cuda" not in device or "gpu" not in device:
            raise NotImplementedError("Tomography not currently supported on CPU.")
        
        self._device = device
        
    @property
    def loss(self) -> list:
        """List of loss values during reconstruction."""
        
        return self._loss
    
    @loss.setter
    def loss(self, loss: list):
        """Set the loss values during reconstruction."""
        
        if not isinstance(loss, list):
            raise TypeError("Loss must be a list.")
        
        self._loss = loss
        
    @property
    def mode(self) -> list:
        """List of modes used during reconstruction."""
        
        return self._mode
    
    
    # --- Preprocessing ---
    
    """
    TODO
    1. Implement tilt series cross-correlation alignment
    2. Background subtraction (for ABF)
    3. COM Alignment
    4. Masking
    5. Drift Correction
    """