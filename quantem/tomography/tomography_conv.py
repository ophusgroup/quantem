
import numpy as np
import torch

from numpy.typing import NDArray



from quantem.core.io.serialize import AutoSerialize
from quantem.core.datastructures.dataset3d import Dataset3d

from quantem.core.utils.validators import ensure_valid_array
from quantem.core.utils.compound_validators import validate_list_of_dataset2d

from typing import Any, Self, Union, Self
from quantem.tomography.tilt_series_dataset import TiltSeries
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.utils import gaussian_filter_2d_stack, torch_phase_cross_correlation

from torch_radon.radon import ParallelBeam as Radon

class TomographyConv(TomographyBase):
    """
    Class for handling conventional reconstruction methods of tomography data.
    """
        
    # --- Reconstruction Methods ---
    """
    TODO
    Implement _run_epoch
    """
    
    def _sirt_run_epoch(
        self,
        radon: Radon,
        stack_recon: torch.Tensor,
        stack_torch: torch.Tensor,
        proj_forward: torch.Tensor,
        step_size: float = 0.25,
        gaussian_kernel: torch.Tensor = None,
        inline_alignment = True,
        enforce_positivity = True,
    ):
        
        loss = 0
        
        if inline_alignment:
            for ind in range(len(self.tilt_series.tilt_angles)):
                im_proj = proj_forward[:, ind, :]
                im_meas = stack_torch[:, ind, :]
                
                shift = torch_phase_cross_correlation(im_proj, im_meas)
                if torch.linalg.norm(shift) <= 32:
                    shifted = torch.fft.ifft2(torch.fft.fft2(im_meas) * torch.exp(
                        -2j * np.pi * (
                            shift[0] * torch.fft.fftfreq(im_meas.shape[0], device=im_meas.device).unsqueeze(1) +
                            shift[1] * torch.fft.fftfreq(im_meas.shape[1], device=im_meas.device)
                        )
                    )).real
                    
                    stack_torch[:, ind, :] = shifted
                
        
        proj_forward = radon.forward(stack_recon)
        
        proj_diff = stack_torch - proj_forward
        
        loss = torch.mean(torch.abs(proj_diff))
        
        recon_slice_update = radon.backward(
            radon.filter_sinogram(
                proj_diff,
            )
        )
        
        stack_recon += step_size * recon_slice_update
        if enforce_positivity:
            stack_recon = torch.clamp(stack_recon, min=0)
        
        if gaussian_kernel is not None:
            stack_recon = gaussian_filter_2d_stack(
                stack_recon,
                gaussian_kernel,
            )
        
        return stack_recon, loss
        
    
    def _sirt_serial_run_epoch(
        self,
        radon: Radon,
        stack_recon: torch.Tensor,
        stack_torch: torch.Tensor,
        proj_forward: torch.Tensor,
        step_size: float = 0.25,
        gaussian_kernel: torch.Tensor = None,
        inline_alignment = True,
        enforce_positivity = True,
    ):
        
        recon_slice_update = torch.zeros_like(stack_recon).to(self.device)
        
        loss = 0
        
        for i in range(stack_recon.shape[0]):
            proj_forward[i] = radon.forward(stack_recon[i])
            
        proj_diff = stack_torch - proj_forward
        
        loss = torch.mean(torch.abs(proj_diff))
        
        for i in range(stack_recon.shape[0]):
            recon_slice_update[i] = radon.backward(
                radon.filter_sinogram(
                    proj_diff[i],
                )
            )
            
        stack_recon += step_size * recon_slice_update
        
        if enforce_positivity:
            stack_recon = torch.clamp(stack_recon, min=0)
            
            
        return stack_recon, loss
    
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