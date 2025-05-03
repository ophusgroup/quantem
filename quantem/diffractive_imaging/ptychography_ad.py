from typing import TYPE_CHECKING

import numpy as np
from tqdm import trange

from quantem.core import config
from quantem.core.datastructures import Dataset, Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.diffractive_imaging.ptycho_utils import (
    fourier_shift,
    fourier_translation_operator,
    generate_batches,
)
from quantem.diffractive_imaging.ptychography_base import PtychographyBase

if TYPE_CHECKING:
    import cupy as cp
    import torch
else: 
    if config.get("has_torch"):
        import torch


class PtychographyAD(PtychographyBase):
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        self._object_padding_force_power2: bool = True # TODO might have to overload padding? 
        self._object_padding_force_power2_level: int = 3
        
        
