from numpy.typing import NDArray
from typing import Optional

import torch
import numpy as np

# from torch_radon.radon import BaseRadon as Radon
from torch_radon.radon import ParallelBeam as Radon
from tqdm.auto import tqdm

from quantem.tomography.tilt_series_dataset import Tilt_Series
from quantem.tomography.utils import gaussian_kernel_1d, gaussian_filter_2d_stack, apply_circular_masks_all_axes
from quantem.core.utils.imaging_utils import cross_correlation_shift

# TODO: Maybe add some filtering after the reconstruction? I.e, masking.

class SIRT_Recon:
    """
    Implements the Simultaneous Iterative Reconstruction Technique (SIRT) for tomographic reconstruction.

    This class reconstructs a 3D volume from a tilt series using the SIRT algorithm and the Radon transform.
    It supports GPU acceleration via PyTorch and provides access to the reconstructed volume and loss history.

    Parameters
    ----------
    dataset : Tilt_Series
        The tomographic dataset containing the tilt series and tilt angles.
    device : str, optional
        The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
    num_iterations : int, optional
        The default number of SIRT iterations to perform. Default is 200.

    Attributes
    ----------
    recon : np.ndarray
        The reconstructed 3D volume as a NumPy array.
    loss : list of float
        The loss history (mean absolute error per iteration).

    Methods
    -------
    reconstruct(num_iterations=10, reset=True)
        Runs the SIRT reconstruction for the specified number of iterations.
    """
    
    def __init__(
        self,
        dataset: Tilt_Series,
        device: str = "cpu",
    ):
        """
        Initializes the reconstruction object for tomography.
        Args:
            dataset (Tilt_Series): The tomography dataset containing the tilt series and tilt angles.
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Defaults to "cpu".
        Raises:
            TypeError: If the provided dataset is not an instance of Tilt_Series.
        Attributes initialized:
            _device (str): The computation device.
            _loss (list): List to store loss values during reconstruction.
            _tilt_series (torch.Tensor): The tilt series data as a float tensor on the specified device.
            _recon (torch.Tensor): The reconstruction volume initialized to zeros.
            _tilt_angles (torch.Tensor): The tilt angles as a float tensor on the specified device.
            _num_angles (int): Number of tilt angles.
            _num_rows (int): Number of rows in each projection.
            _num_sinograms (int): Number of sinograms (projections per angle).
        """
        
        if not isinstance(dataset, Tilt_Series):
            raise TypeError("dataset must be a Tilt_Series instance") # Make into a setter
        if "cuda" not in device:
            raise NotImplementedError("Only CUDA is supported for now")
        
        self._device = device
        self._loss = []
        
        
        # self._tilt_series = torch.from_numpy(np.transpose(dataset.array, axes = (2, 0, 1))).float().to(self._device)
        self._tilt_series = torch.from_numpy(dataset.array).float().to(self._device)
        
        self._num_angles, self._num_rows, self._num_sinograms = dataset.array.shape
        
        self._recon = torch.zeros(
            (self._num_rows, self._num_rows, self._num_sinograms),
            dtype = torch.float32,
            device = self._device,
        )
        self._tilt_angles = torch.from_numpy(dataset.tilt_angles_rad).float().to(self._device)
        
        
    
    def reconstruct(
            self, 
            num_iterations: int = 10, 
            reset: bool = True,
            step_size: float = 0.25,
            enforce_positivity: bool = True,
            smoothing_sigma: Optional[float] = None,
            inline_alignment: bool = False,
            batched_recon: bool = False, # TODO: Fix batched recon
            circular_mask: bool = False,
            radii: Optional[tuple[float, float, float]] = None,
        ):
        """
        Performs iterative tomographic reconstruction using the SIRT algorithm.
        Args:
            num_iterations (int, optional): Number of SIRT iterations to perform. Defaults to 10.
            reset (bool, optional): If True, resets the reconstruction and loss history before starting. Defaults to True.
        Description:
            Initializes the reconstruction volume and loss history if `reset` is True. 
            Sets up the Radon transform operator with the current configuration.
            Iteratively updates the reconstruction for the specified number of iterations,
            tracking and displaying the loss at each step.
        Side Effects:
            Updates `self._recon` with the reconstructed volume.
            Appends loss values to `self._loss` after each iteration.
        """
        
        if inline_alignment:
            raise NotImplementedError("Inline alignment is not yet implemented.")
        
        if circular_mask:
            if radii is None:
                raise ValueError("Radii must be provided for circular masking.")
            if len(radii) != 3:
                raise ValueError("Radii must be a tuple of three values for each axis.")
            
        
        if reset:
            self._recon = torch.zeros(
                (self._num_rows, self._num_rows, self._num_sinograms),
                dtype = torch.float32,
                device = self._device,
            )
            self._loss = []
        
        # Instantiate the Radon transform
        radon = Radon(
            det_count = self._num_rows,
            angles = self._tilt_angles,
            # clip_to_circle=True,
            # det_count = self._num_rows,
        )
        
        # Instantiate Gaussian kernel
        if smoothing_sigma is not None:
            kernel_1d = gaussian_kernel_1d(smoothing_sigma).to(self._device)
        else:
            kernel_1d = None
        
        pbar = tqdm(range(num_iterations), desc="SIRT Iterations")
        
        for a0 in pbar:
            
            if inline_alignment and a0 >= 2:
                self._run_epoch(radon, 
                                step_size=step_size,
                                enforce_positivity=enforce_positivity,
                                kernel_1d = kernel_1d, 
                                inline_alignment=True,
                                batched_recon=batched_recon,
                                )
            else:
                self._run_epoch(radon, 
                                step_size=step_size, 
                                enforce_positivity=enforce_positivity, 
                                kernel_1d = kernel_1d, 
                                inline_alignment=False,
                                batched_recon=batched_recon,
                                )

            pbar.set_description(
                f"SIRT Iteration {a0+1}/{num_iterations} - Loss: {self._loss[-1]:.4f}"
            )
            
        # Apply circular mask if specified
        if circular_mask:
            self._recon = apply_circular_masks_all_axes(self._recon, radii)
        
    def _run_epoch(
        self, 
        radon: Radon, 
        step_size: float, 
        enforce_positivity: bool , 
        kernel_1d: Optional[torch.Tensor],
        inline_alignment: bool, # TODO: Implement inline alignment
        batched_recon: bool = False, # TODO: Fix batched recon
        ):
        """
        Performs a single epoch of iterative reconstruction for all sinograms.
        Args:
            radon (Radon): An instance of the Radon transform class, providing forward and backprojection operations.
            step_size (float, optional): The step size for the reconstruction update. Defaults to 0.25.
        Updates:
            self._recon: The reconstructed volume is updated in-place for each slice.
            self._loss: Appends the average loss (mean absolute error) for the epoch.
        Process:
            - For each sinogram slice:
                - Computes the forward projection of the current reconstruction.
                - Calculates the difference between the measured and forward-projected sinogram.
                - Accumulates the mean absolute error as loss.
                - Applies filtering and backprojection to the difference.
                - Updates the reconstruction slice using the computed update and step size.
            - Clamps the reconstructed values to be non-negative.
            - Stores the average loss for the epoch.
        """
        
        # TODO: Incorporate inline alignment
        #   - Would this mean I have to split the backprojection into different for loops?
        #   - I wonder if there's a better way to do this vectorized.
        
        loss = 0
        
        # --- Batching whole sinogram in ---
        if batched_recon:
            proj_forward = radon.forward(self._recon)
            
            proj_diff = self._tilt_series - proj_forward
            loss += torch.mean(torch.abs(proj_diff))
            recon_update = radon.backward(
                radon.filter_sinogram(
                    proj_diff,
                )
            )
            self._recon += step_size * recon_update
        
        # --- Looping through each sinogram ---
        else:
            for i in range(self._num_sinograms):
                proj_forward = radon.forward(self._recon[:, :, i])

                proj_diff = self._tilt_series[:, :, i] - proj_forward
                loss += torch.mean(torch.abs(proj_diff))
                recon_update = radon.backward(
                    radon.filter_sinogram(
                        proj_diff,
                        filter_name = 'hamming'
                    )
                )
                self._recon[:, :, i] += step_size * recon_update
            
            loss /= self._num_sinograms

        if kernel_1d is not None:
            self._recon = gaussian_filter_2d_stack(self._recon, kernel_1d)
        
        if enforce_positivity:
            self._recon = torch.clamp(self._recon, min=0)
        
        # Circular mask
        
        
        # Appending to loss
        self._loss.append(loss.detach().cpu().numpy())
    

    # --- Properties ---
    @property
    def recon(self) -> NDArray:
        """Get the reconstructed dataset."""
        # return np.transpose(self._recon.cpu().numpy(), axes = (1, 2, 0))
        return self._recon.cpu().numpy()
    
    
    @property
    def loss(self) -> NDArray:
        return self._loss
    
    

