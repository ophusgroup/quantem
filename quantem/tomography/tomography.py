
import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset3d import Dataset3d

from quantem.core.utils.validators import ensure_valid_array
from quantem.core.utils.compound_validators import validate_list_of_dataset2d

from typing import Any, Self, Union, Self
from quantem.tomography.tilt_series_dataset import TiltSeries
from quantem.tomography.tomography_conv import TomographyConv
from quantem.tomography.tomography_ml import TomographyML
from quantem.tomography.tomography_base import TomographyBase

class Tomography(TomographyConv, TomographyML, TomographyBase):
    """
    Top level class for either using conventional or ML-based reconstruction methods
    for tomography.
    """
    
    def __init__(
        self,
        tilt_series,
        recon_volume,
        device,
        _token,
    ):
        super().__init__(tilt_series, recon_volume, device, _token)
        
    # --- Reconstruction Method ---
    
    def reconstruct(
        self,
    ):
        
