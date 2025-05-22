
import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset3d import Dataset3d

from quantem.core.utils.validators import ensure_valid_array
from quantem.core.utils.compound_validators import validate_list_of_dataset2d

from typing import Any, Self, Union, Self
from quantem.tomography.tilt_series_dataset import TiltSeries
from quantem.tomography.tomography_base import TomographyBase

class TomographyConv(TomographyBase):
    """
    Class for handling conventional reconstruction methods of tomography data.
    """
    
    # def __init__(
    #     self,
    #     reconstruction_method: str,
    # ):
    #     super().__init__()
        
    # --- Reconstruction Methods ---
    """
    TODO
    Implement _run_epoch
    """
    def reconstruct(
        self, 
        reconstruction_method: None,
        num_iterations: int = 10, 
        step_size: float = 0.25,
        reset: bool = True,
        inline_alingment: bool = False,
        enforce_positivity: bool = True,
        smoothing_sigma: float = None,
    ):
        
        if reset:
            self.loss = []
            self.mode = []
            self.recon_volume = None
        
        if self._reconstruction_method == "SIRT":
            reconstructed_volume, loss = SIRT_Recon(
                tilt_series=self.tilt_series,
                num_iterations=num_iterations,
            )
        else:
            raise NotImplementedError(
                f"Reconstruction method '{self._reconstruction_method}' is not implemented."
            )
    # --- Properties ---
    # @property
    # def reconstruction_method(self) -> str:
    #     """Get the reconstruction method."""
    #     return self._reconstruction_method
    # @reconstruction_method.setter
    # def reconstruction_method(self, value: str):
    #     """Set the reconstruction method."""
    #     if value not in ["SIRT", "FBP"]:
    #         raise ValueError("Invalid reconstruction method. Choose 'SIRT' or 'FBP'.")
    #     self._reconstruction_method = value