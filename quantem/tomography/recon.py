from numpy.typing import NDArray
from typing import Optional

import torch
import numpy as np

from torch_radon import Radon
from tqdm.auto import tqdm

from quantem.tomography.dataset_tomo import DatasetTomo
from quantem.tomography.utils import gaussian_kernel_1d, gaussian_filter_2d

class SIRT_Recon:
    """
    Implements the Simultaneous Iterative Reconstruction Technique (SIRT) for tomographic reconstruction.

    This class reconstructs a 3D volume from a tilt series using the SIRT algorithm and the Radon transform.
    It supports GPU acceleration via PyTorch and provides access to the reconstructed volume and loss history.

    Parameters
    ----------
    dataset : DatasetTomo
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
        dataset: DatasetTomo,
        device: str = "cpu",
    ):
        """
        Initializes the reconstruction object for tomography.
        Args:
            dataset (DatasetTomo): The tomography dataset containing the tilt series and tilt angles.
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Defaults to "cpu".
        Raises:
            TypeError: If the provided dataset is not an instance of DatasetTomo.
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
        
        if not isinstance(dataset, DatasetTomo):
            raise TypeError("dataset must be a DatasetTomo instance") # Make into a setter
        if "cuda" not in device:
            raise NotImplementedError("Only CUDA is supported for now")
        
        self._device = device
        self._loss = []
        
        self._tilt_series = torch.from_numpy(dataset.array).float().to(self._device)
        self._recon = torch.zeros(
            (self._num_rows, self._num_rows, self._num_sinograms),
            dtype = torch.float32,
            device = self._device,
        )
        self._tilt_angles = torch.from_numpy(dataset.tilt_angles_rad).float().to(self._device)
        self._num_angles, self._num_rows, self._num_sinograms = self._tilt_series.shape
        
        
    
    def reconstruct(
            self, 
            num_iterations: int = 10, 
            reset: bool = True,
            step_size: float = 0.25,
            enforce_positivity: bool = True,
            smoothing_sigma: Optional[float] = None,
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
        
        if reset:
            self._recon = torch.zeros(
                (self._num_rows, self._num_rows, self._num_sinograms),
                dtype = torch.float32,
                device = self._device,
            )
            self._loss = []
        
        # Instantiate the Radon transform
        radon = Radon(
            resolution = self._num_rows,
            angles = self._tilt_angles,
            clip_to_circle=True,
            det_count = self._num_rows,
        )
        
        pbar = tqdm(range(num_iterations), desc="SIRT Iterations")
        
        for a0 in pbar:
            self._run_epoch(radon, step_size=step_size, enforce_positivity=enforce_positivity, smoothing_sigma=smoothing_sigma)

            pbar.set_description(
                f"SIRT Iteration {a0+1}/{num_iterations} - Loss: {self._loss[-1]:.4f}"
            )
        
        
    def _run_epoch(self, radon: Radon, step_size: float = 0.25, enforce_positivity: bool = True, smoothing_sigma: Optional[float] = None):
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
        loss = 0
        
        for ind in range(self._num_sinograms):
            
            proj_forward = radon.forward(self._recon[:, :, ind])
            proj_diff = self._tilt_series[:, :, ind] - proj_forward
            
            loss += torch.mean(torch.abs(proj_diff))

            recon_slice_update = radon.backprojection(
                radon.filter_sinogram(
                    proj_diff,
                )
            )
            self._recon[:, :, ind] += step_size * recon_slice_update
            
            # Applying Gaussian smoothing if specified
            if smoothing_sigma is not None:
                kernel_1d = gaussian_kernel_1d(smoothing_sigma).to(self._device)
                self._recon[:, :, ind] = gaussian_filter_2d(self._recon[:, :, ind], smoothing_sigma, kernel_1d)
            
        if enforce_positivity:
            self._recon = torch.clamp(self._recon, min=0)
            
        # Appending to loss
        self._loss.append(loss.detach().cpu().numpy()/self._num_sinograms)
    
    
    # --- Properties ---
    @property
    def recon(self) -> NDArray:
        """Get the reconstructed dataset."""
        return self._recon.cpu().numpy()
    
    @property
    def loss(self) -> NDArray:
        return self._loss
    
    
