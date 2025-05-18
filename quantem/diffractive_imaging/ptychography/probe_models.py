from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from quantem.core.datastructures import Dataset2d
from quantem.diffractive_imaging.converged_probe import ConvergedProbe


class ProbeModelBase(ABC):
    """
    Base class for all ProbeModels to inherit from.
    """

    @abstractmethod
    def initialize_probe(self, *args):
        pass

    @abstractmethod
    def forward_probe(self, *args):
        pass

    @abstractmethod
    def backward_probe(self, *args):
        pass


class SingleProbeModel(ProbeModelBase):
    """ """

    def __init__(
        self,
        energy: float,
        gpts: Tuple[int, int],
        sampling: Tuple[int, int],
        semiangle_cutoff: float = torch.inf,
        soft_aperture: bool = True,
        vacuum_probe_intensity: Optional[torch.Tensor] = None,
        aberration_coefficients: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        self.dataset = Dataset2d.from_array(
            self.initialize_probe(
                energy,
                gpts,
                sampling,
                semiangle_cutoff,
                soft_aperture,
                vacuum_probe_intensity,
                aberration_coefficients,
                **kwargs,
            ),
            name="ptychographic probe",
            sampling=sampling,
            units=("A", "A"),
        )

    def initialize_probe(
        self,
        energy: float,
        gpts: Tuple[int, int],
        sampling: Tuple[int, int],
        semiangle_cutoff: float = torch.inf,
        soft_aperture: bool = True,
        vacuum_probe_intensity: Optional[torch.Tensor] = None,
        aberration_coefficients: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        probe = ConvergedProbe(
            energy,
            gpts,
            sampling,
            semiangle_cutoff,
            soft_aperture,
            vacuum_probe_intensity,
            aberration_coefficients,
            **kwargs,
        ).build()

        return probe._array

    def forward_probe(
        self,
        fractional_positions_px: torch.Tensor,
    ):
        shifted_probes = fourier_shift(
            self.dataset.array,
            fractional_positions_px,
        )

        return shifted_probes

    def backward_probe(self):
        pass


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
