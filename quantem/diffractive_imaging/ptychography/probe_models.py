from typing import Optional, Self, Tuple

import torch

from quantem.core.datastructures import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.diffractive_imaging.converged_probe import ConvergedProbe
from quantem.diffractive_imaging.ptychography.ptychography_utils import fourier_shift


class ProbeModelBase(AutoSerialize):
    """
    Base class for all ProbeModels to inherit from.
    """

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    @property
    def tensor(self) -> torch.Tensor:
        return self.dataset.array


class PixelatedProbeModel(ProbeModelBase):
    """ """

    _token = object()

    def __init__(
        self,
        probe_dataset: Dataset2d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use SingleProbeModel.from_array() or SingleProbeModel.from_aberration_coefficients() to instantiate this class."
            )

        self.dataset = probe_dataset

    @classmethod
    def from_array(
        cls,
        array: torch.Tensor,
        mean_diffraction_intensity: float,
        sampling: Tuple[int, int],
    ) -> Self:
        """ """

        probe_array = torch.as_tensor(array).to(torch.cfloat)
        probe_intensity = torch.sum(
            torch.square(torch.abs(torch.fft.fft2(probe_array, norm="ortho")))
        )
        normalized_probe = probe_array * torch.sqrt(
            mean_diffraction_intensity / probe_intensity
        )

        probe_dataset = Dataset2d.from_array(
            normalized_probe,
            name="ptychographic probe",
            sampling=sampling,
            units=("A", "A"),
        )

        return cls(
            probe_dataset,
            cls._token,
        )

    @classmethod
    def from_aberration_coefficients(
        cls,
        energy: float,
        gpts: Tuple[int, int],
        sampling: Tuple[int, int],
        mean_diffraction_intensity: float,
        semiangle_cutoff: float = torch.inf,
        soft_aperture: bool = True,
        vacuum_probe_intensity: Optional[torch.Tensor] = None,
        aberration_coefficients: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> Self:
        """ """
        probe = (
            ConvergedProbe(
                energy,
                gpts,
                sampling,
                semiangle_cutoff,
                soft_aperture,
                vacuum_probe_intensity,
                aberration_coefficients,
                **kwargs,
            )
            .build()
            ._array
        )

        # Normalize probe to match mean diffraction intensity
        return cls.from_array(probe, mean_diffraction_intensity, sampling)

    def forward(
        self,
        fractional_positions_px: torch.Tensor,
    ):
        shifted_probes = fourier_shift(
            self.tensor,
            fractional_positions_px,
        )

        return shifted_probes

    def backward(
        self,
        gradient_array,
        object_array,
    ):
        if self.tensor.requires_grad:
            obj_normalization = torch.sum(torch.square(torch.abs(object_array)), dim=0)

            probe_gradient = (
                torch.sum(gradient_array * torch.conj(object_array), dim=0)
                / obj_normalization
            )

            self.tensor.grad = probe_gradient.clone().detach()

            return probe_gradient
