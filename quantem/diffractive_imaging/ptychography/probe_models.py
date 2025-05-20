from typing import Optional, Self, Tuple

import torch

from quantem.core.datastructures import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.diffractive_imaging.converged_probe import ConvergedProbe


def fourier_translation_operator(
    positions: torch.Tensor,
    shape: tuple,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Returns phase ramp for fourier-shifting array of shape `shape`."""

    nx, ny = shape[-2:]
    x = positions[..., 0][:, None, None]
    y = positions[..., 1][:, None, None]

    kx = torch.fft.fftfreq(nx, d=1.0, device=device)
    ky = torch.fft.fftfreq(ny, d=1.0, device=device)
    ramp_x = torch.exp(-2.0j * torch.pi * kx[None, :, None] * x)
    ramp_y = torch.exp(-2.0j * torch.pi * ky[None, None, :] * y)

    ramp = ramp_x * ramp_y
    return ramp


def fourier_shift(array: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Fourier-shift array by flat array of positions"""
    phase = fourier_translation_operator(positions, array.shape, device=array.device)
    fourier_array = torch.fft.fft2(array)
    shifted_fourier_array = fourier_array * phase

    return torch.fft.ifft2(shifted_fourier_array)


class ProbeModelBase(AutoSerialize):
    """
    Base class for all ProbeModels to inherit from.
    """

    def initialize(self, *args):
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    @property
    def tensor(self) -> torch.Tensor:
        return self.dataset.array


class SingleProbeModel(ProbeModelBase):
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
            torch.square(torch.abs(torch.fft.fft2(probe_array)))
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
            probe_gradient = torch.mean(
                gradient_array * torch.conj(object_array), dim=0
            )

            self.tensor.grad = probe_gradient.clone().detach()

            return probe_gradient
