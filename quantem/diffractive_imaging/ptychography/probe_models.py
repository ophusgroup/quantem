from typing import List, Optional, Self, Tuple

import torch
import torch.nn as nn

from quantem.core.datastructures import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.diffractive_imaging.converged_probe import ConvergedProbe
from quantem.diffractive_imaging.ptychography.ptychography_utils import (
    fourier_shift,
    fourier_translation_operator,
)

# region --- Pixelated Probe ---


class PixelatedProbeModel(AutoSerialize):
    """ """

    _token = object()

    def __init__(
        self,
        probe_dataset: Dataset3d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use PixelatedProbeModel.from_array() or PixelatedProbeModel.from_aberration_coefficients() to instantiate this class."
            )

        self.dataset = probe_dataset
        self.roi_shape = self.dataset.shape[-2:]
        self.sampling = self.dataset.sampling[-2:]
        self.num_probes = self.dataset.shape[0]

    @classmethod
    def from_array(
        cls,
        array: torch.Tensor,
        mean_diffraction_intensity: float,
        sampling: Tuple[int, int],
    ) -> Self:
        """ """

        probe_array = torch.as_tensor(array).to(torch.cfloat)

        if probe_array.ndim == 2:
            return cls.from_single_probe(
                probe_array,
                num_probes=1,
                mean_diffraction_intensity=mean_diffraction_intensity,
                sampling=sampling,
            )

        probe_intensity = torch.sum(
            torch.square(torch.abs(torch.fft.fft2(probe_array, norm="ortho"))),
        )
        normalized_probe = probe_array * torch.sqrt(
            mean_diffraction_intensity / probe_intensity + 1e-8
        )

        probe_dataset = Dataset3d.from_array(
            normalized_probe,
            name="ptychographic probe",
            sampling=(1,) + tuple(sampling),
            units=("index", "A", "A"),
        )

        return cls(
            probe_dataset,
            cls._token,
        )

    @classmethod
    def from_single_probe(
        cls,
        array: torch.Tensor,
        num_probes: int,
        mean_diffraction_intensity: float,
        sampling: Tuple[int, int],
    ) -> Self:
        """ """

        gpts = array.shape
        probe = torch.empty((num_probes,) + gpts, dtype=torch.cfloat)
        probe[0] = torch.as_tensor(array).to(torch.cfloat)

        phase_ramps = fourier_translation_operator(
            torch.rand(num_probes, 2) - 0.5, gpts
        )
        for s in range(1, num_probes):
            probe[s] = probe[s - 1] * phase_ramps[s]

        # Normalize probe to match mean diffraction intensity
        return cls.from_array(probe, mean_diffraction_intensity, sampling)

    @classmethod
    def from_aberration_coefficients(
        cls,
        num_probes: int,
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
                soft_aperture,
                semiangle_cutoff,
                vacuum_probe_intensity,
                mean_diffraction_intensity,
                aberration_coefficients,
                **kwargs,
            )
            .build()
            ._array
        )

        return cls.from_single_probe(
            probe, num_probes, mean_diffraction_intensity, sampling
        )

    def forward(
        self,
        fractional_positions_px: torch.Tensor,
    ):
        shifted_probes = fourier_shift(
            self.tensor,
            fractional_positions_px[:, None, :],
        )

        return shifted_probes

    def backward(
        self,
        gradient_array,
        object_array,
    ):
        if self.tensor.requires_grad:
            obj_normalization = torch.sum(
                torch.square(torch.abs(object_array)), dim=(0, 1)
            )

            probe_gradient = (
                torch.sum(gradient_array * torch.conj(object_array), dim=0)
                / obj_normalization[None]
            )

            self.tensor.grad = probe_gradient.clone().detach()

            return probe_gradient

    def orthogonalize(
        self,
    ):
        """ """
        num_probes = self.num_probes
        probe = self.tensor.clone()
        pairwise_dot_product = torch.empty((num_probes, num_probes), dtype=probe.dtype)

        for i in range(num_probes):
            for j in range(num_probes):
                pairwise_dot_product[i, j] = torch.sum(probe[i].conj() * probe[j])

        _, evecs = torch.linalg.eigh(pairwise_dot_product, UPLO="U")
        probe = torch.tensordot(evecs.T, probe, dims=1)

        intensities = torch.sum(probe.abs().square(), dim=(-2, -1))
        intensities_order = torch.argsort(intensities).flip(0)

        self.tensor.data = probe[intensities_order]
        return self.tensor

    @property
    def tensor(self) -> torch.Tensor:
        return self.dataset.array

    def parameters(self) -> List[torch.Tensor]:
        return [self.tensor]


# endregion --- Pixelated Probe ---


# region --- Parametrized Probe ---


class ParametrizedProbeModel(PixelatedProbeModel):
    """ """

    _token = object()

    def __init__(
        self,
        energy: float,
        gpts: Tuple[int, int],
        sampling: Tuple[int, int],
        mean_diffraction_intensity: float,
        semiangle_cutoff: float,
        aberration_coefficients: nn.ParameterDict,
        _token: object | None = None,
    ):
        """ """

        if _token is not self._token:
            raise RuntimeError(
                "Use ParametrizedProbeModel.from_aberration_coefficients() to instantiate this class."
            )

        self.energy = energy
        self.sampling = sampling
        self.roi_shape = gpts
        self.num_probes = 1

        self._parameters = (
            nn.ParameterDict(
                {
                    "mean_diffraction_intensity": nn.Parameter(
                        torch.as_tensor(mean_diffraction_intensity, dtype=torch.float)
                    ),
                    "semiangle_cutoff": nn.Parameter(
                        torch.as_tensor(semiangle_cutoff, dtype=torch.float)
                    ),
                }
            )
            | aberration_coefficients
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
        if vacuum_probe_intensity is not None:
            raise NotImplementedError(
                "Use PixelatedProbeModel.from_aberration_coefficients() instead."
            )

        if soft_aperture is not True:
            raise NotImplementedError()

        probe = ConvergedProbe(
            energy,
            gpts,
            sampling,
            True,
            semiangle_cutoff,
            vacuum_probe_intensity,
            mean_diffraction_intensity,
            aberration_coefficients,
            **kwargs,
        )

        return cls(
            energy,
            gpts,
            sampling,
            mean_diffraction_intensity,
            semiangle_cutoff,
            probe._aberration_coefficients,
            cls._token,
        )

    def build_probe(self) -> torch.Tensor:
        """Rebuild probe from current parameters."""
        parameters = self.parameters()
        semiangle_cutoff = parameters["semiangle_cutoff"]
        mean_diffraction_intensity = parameters["mean_diffraction_intensity"]
        aberrations = {
            k: v
            for k, v in parameters.items()
            if k.startswith("phi") or k.startswith("C")
        }

        probe = (
            ConvergedProbe(
                self.energy,
                self.roi_shape,
                self.sampling,
                soft_aperture=True,
                semiangle_cutoff=semiangle_cutoff,
                vacuum_probe_intensity=None,
                fourier_intensity_normalization=mean_diffraction_intensity,
                aberration_coefficients=aberrations,
            )
            .build()
            ._array
        )

        return probe[None]

    def backward(self, *args, **kwargs):
        raise NotImplementedError("Use autograd for ParametrizedProbeModel.")

    @property
    def tensor(self) -> torch.Tensor:
        return self.build_probe()

    def parameters(self) -> List[nn.Parameter]:
        return self._parameters


# endregion --- Parametrized Probe ---
