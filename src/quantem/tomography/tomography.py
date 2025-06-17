import torch

# from torch_radon.radon import ParallelBeam as Radon
from tqdm.auto import tqdm

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_conv import TomographyConv
from quantem.tomography.tomography_ml import TomographyML
from quantem.tomography.utils import gaussian_kernel_1d


class Tomography(TomographyConv, TomographyML, TomographyBase):
    """
    Top level class for either using conventional or ML-based reconstruction methods
    for tomography.
    """

    def __init__(
        self,
        tilt_series,
        volume_obj,
        device,
        _token,
    ):
        super().__init__(tilt_series, volume_obj, device, _token)

    # --- Reconstruction Method ---

    def sirt_recon(
        self,
        num_iterations: int = 10,
        inline_alignment: bool = False,
        enforce_positivity: bool = True,
        volume_shape: tuple = None,
        reset: bool = True,
        smoothing_sigma: float = None,
        shrinkage: float = None,
    ):
        num_slices, num_angles, num_rows = self.tilt_series.array.shape

        if volume_shape is None:
            volume_shape = (num_rows, num_rows, num_rows)
        else:
            D, H, W = volume_shape

        if reset:
            volume = torch.zeros((D, H, W), device=self.device, dtype=torch.float32)
            self.loss = []
        else:
            volume = torch.tensor(
                self.volume_obj.array,
                device=self.device,
                dtype=torch.float32,
            )

        tilt_series_torch = torch.tensor(
            self.tilt_series.array,
            device=self.device,
            dtype=torch.float32,
        )

        tilt_angles_torch = torch.tensor(
            self.tilt_series.tilt_angles,
            device=self.device,
            dtype=torch.float32,
        )

        proj_forward = torch.zeros_like(tilt_series_torch)

        pbar = tqdm(range(num_iterations), desc="SIRT Reconstruction")

        if smoothing_sigma is not None:
            gaussian_kernel = gaussian_kernel_1d(smoothing_sigma).to(self.device)
        else:
            gaussian_kernel = None

        for iter in pbar:
            if iter > 0 and inline_alignment:
                volume, proj_forward, loss = self._sirt_run_epoch(
                    volume=volume,
                    tilt_series=tilt_series_torch,
                    proj_forward=proj_forward,
                    angles=tilt_angles_torch,
                    inline_alignment=True,
                    enforce_positivity=enforce_positivity,
                    shrinkage=shrinkage,
                    gaussian_kernel=gaussian_kernel,
                )
            else:
                volume, proj_forward, loss = self._sirt_run_epoch(
                    volume=volume,
                    tilt_series=tilt_series_torch,
                    proj_forward=proj_forward,
                    angles=tilt_angles_torch,
                    inline_alignment=False,
                    enforce_positivity=enforce_positivity,
                    shrinkage=shrinkage,
                    gaussian_kernel=gaussian_kernel,
                )

            pbar.set_description(f"SIRT Reconstruction | Loss: {loss.item():.4f}")

            self.loss.append(loss.item())

        self.volume_obj = Dataset3d.from_array(
            array=volume.cpu().numpy(),
            name=self.tilt_series.name,
            origin=self.tilt_series.origin,
            sampling=self.tilt_series.sampling,
            units=self.tilt_series.units,
            signal_units=self.tilt_series.signal_units,
        )

    def voxel_wise_recon(
        self,
    ):
        if not isinstance(self.volume_obj):
            raise NotImplementedError()
        raise NotImplementedError(
            "Voxel-wise reconstruction is not implemented yet. Please use the SIRT method for now."
        )

    def recon_ML(
        self,
    ):
        raise NotImplementedError(
            "ML-based reconstruction is not implemented yet. Please use the SIRT method for now."
        )
