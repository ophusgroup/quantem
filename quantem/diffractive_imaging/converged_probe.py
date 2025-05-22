from typing import Optional, Tuple

import torch
import torch.nn as nn
from numpy.typing import NDArray

from quantem.core.utils.scattering_utils import electron_wavelength_angstrom

# fmt: off
polar_symbols = (
    "C10", "C12"  , "phi12",
    "C21", "phi21", "C23"  , "phi23",
    "C30", "C32"  , "phi32", "C34"  , "phi34",
    "C41", "phi41", "C43"  , "phi43", "C45"  , "phi45",
    "C50", "C52"  , "phi52", "C54"  , "phi54", "C56"  , "phi56",
)

polar_aliases = {
    "defocus": "C10",
    "astigmatism": "C12",
    "astigmatism_angle": "phi12",
    "coma": "C21",
    "coma_angle": "phi21",
    "Cs": "C30",
    "C5": "C50",
}
# fmt: on


class ConvergedProbe:
    """ """

    def __init__(
        self,
        energy: float,
        gpts: Tuple[int, int],
        sampling: Tuple[float, float],
        soft_aperture: bool = True,
        semiangle_cutoff: float | torch.Tensor = torch.inf,
        vacuum_probe_intensity: Optional[torch.Tensor | NDArray] = None,
        fourier_intensity_normalization: float | torch.Tensor = 1.0,
        aberration_coefficients: Optional[
            dict[str, float] | nn.ParameterDict[str, torch.Tensor]
        ] = None,
        **kwargs,
    ):
        """ """

        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()):
                raise ValueError("{} not a recognized parameter".format(key))

        if aberration_coefficients is None:
            aberration_coefficients: dict[str, float | torch.Tensor] = {}
        aberration_coefficients.update(kwargs)

        self._aberration_coefficients = torch.nn.ParameterDict(
            (k, 0.0) for k in polar_symbols
        )
        self._aberration_coefficients.update(
            self.standardize_aberration_coefficients(aberration_coefficients)
        )

        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._fourier_intensity_normalization = fourier_intensity_normalization
        self._semiangle_cutoff = semiangle_cutoff
        self._soft_aperture = soft_aperture
        self._energy = energy
        self._wavelength = electron_wavelength_angstrom(energy)
        self._gpts = gpts
        self._sampling = sampling
        self._angular_sampling = tuple(
            self._wavelength * 1e3 / s / n for s, n in zip(self._sampling, self._gpts)
        )

    @staticmethod
    def standardize_aberration_coefficients(
        aberration_coefficients: dict[str, float | torch.Tensor],
    ) -> dict[str, float | torch.Tensor]:
        """
        Parameters
        ----------
        aberration_coefficients: dict
            Mapping from aberration symbols to their corresponding values.
        """

        keys_to_pop = []
        for symbol, value in aberration_coefficients.items():
            if symbol in polar_symbols:
                pass
            elif symbol == "defocus":
                keys_to_pop.append(symbol)
                aberration_coefficients[polar_aliases[symbol]] = -value
            elif symbol in polar_aliases.keys():
                keys_to_pop.append(symbol)
                aberration_coefficients[polar_aliases[symbol]] = value
            else:
                raise ValueError("{} not a recognized parameter".format(symbol))

        for key in keys_to_pop:
            aberration_coefficients.pop(key)

        return aberration_coefficients

    def get_scattering_angles(self):
        """ """
        gpts = self._gpts
        sampling = self._sampling
        wavelength = self._wavelength

        kx, ky = tuple(torch.fft.fftfreq(n, s) for n, s in zip(gpts, sampling))
        alpha = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) * wavelength
        phi = torch.arctan2(ky[None, :], kx[:, None])

        return alpha, phi

    def evaluate_aperture(
        self,
        alpha: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """ """
        semiangle_cutoff = self._semiangle_cutoff
        soft_aperture = self._soft_aperture
        vacuum_probe_intensity = self._vacuum_probe_intensity
        angular_sampling = self._angular_sampling

        if vacuum_probe_intensity is not None:
            vacuum_probe_intensity = torch.as_tensor(
                vacuum_probe_intensity, dtype=torch.float
            ).clip(0)
            return torch.sqrt(vacuum_probe_intensity + 1e-8)

        if semiangle_cutoff == torch.inf:
            return torch.ones_like(alpha)

        if soft_aperture:
            denominator = torch.sqrt(
                (torch.cos(phi) * angular_sampling[0] * 1e-3) ** 2
                + (torch.sin(phi) * angular_sampling[1] * 1e-3) ** 2
                + 1e-8
            )
            array = torch.clip(
                (semiangle_cutoff / 1000 - alpha) / denominator + 0.5, 0.0, 1.0
            )
        else:
            array = (alpha < semiangle_cutoff / 1000).to(torch.float)

        return array

    def evaluate_chi(
        self,
        alpha: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """ """
        p = self._aberration_coefficients
        wavelength = self._wavelength
        alpha2 = torch.square(alpha)

        array = torch.zeros_like(alpha)
        if any([p[symbol] != 0.0 for symbol in ("C10", "C12", "phi12")]):
            array += (
                1
                / 2
                * alpha2
                * (p["C10"] + p["C12"] * torch.cos(2 * (phi - p["phi12"])))
            )

        if any([p[symbol] != 0.0 for symbol in ("C21", "phi21", "C23", "phi23")]):
            array += (
                1
                / 3
                * alpha2
                * alpha
                * (
                    p["C21"] * torch.cos(phi - p["phi21"])
                    + p["C23"] * torch.cos(3 * (phi - p["phi23"]))
                )
            )

        if any(
            [p[symbol] != 0.0 for symbol in ("C30", "C32", "phi32", "C34", "phi34")]
        ):
            array += (
                1
                / 4
                * torch.square(alpha2)
                * (
                    p["C30"]
                    + p["C32"] * torch.cos(2 * (phi - p["phi32"]))
                    + p["C34"] * torch.cos(4 * (phi - p["phi34"]))
                )
            )

        if any(
            [
                p[symbol] != 0.0
                for symbol in ("C41", "phi41", "C43", "phi43", "C45", "phi41")
            ]
        ):
            array += (
                1
                / 5
                * torch.square(alpha2)
                * alpha
                * (
                    p["C41"] * torch.cos((phi - p["phi41"]))
                    + p["C43"] * torch.cos(3 * (phi - p["phi43"]))
                    + p["C45"] * torch.cos(5 * (phi - p["phi45"]))
                )
            )

        if any(
            [
                p[symbol] != 0.0
                for symbol in ("C50", "C52", "phi52", "C54", "phi54", "C56", "phi56")
            ]
        ):
            array += (
                1
                / 6
                * torch.pow(alpha, 3.0)
                * (
                    p["C50"]
                    + p["C52"] * torch.cos(2 * (phi - p["phi52"]))
                    + p["C54"] * torch.cos(4 * (phi - p["phi54"]))
                    + p["C56"] * torch.cos(6 * (phi - p["phi56"]))
                )
            )

        array = 2 * torch.pi / wavelength * array
        return array

    def evaluate(self) -> torch.Tensor:
        """ """
        alpha, phi = self.get_scattering_angles()
        array = torch.exp(-1.0j * self.evaluate_chi(alpha, phi))
        array = array * self.evaluate_aperture(alpha, phi)

        return array

    def build(self) -> "ConvergedProbe":
        """ """
        fourier_array = self.evaluate()
        fourier_intensity = torch.sum(torch.square(torch.abs(fourier_array)))
        normalization = torch.sqrt(
            fourier_intensity / self._fourier_intensity_normalization
        )
        self._fourier_array = fourier_array / normalization
        self._array = torch.fft.ifft2(self._fourier_array, norm="ortho")

        return self
