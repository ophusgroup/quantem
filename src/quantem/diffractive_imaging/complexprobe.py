from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from quantem.core import config

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


polar_symbols = (
    "C10",
    "C12",
    "phi12",
    "C21",
    "phi21",
    "C23",
    "phi23",
    "C30",
    "C32",
    "phi32",
    "C34",
    "phi34",
    "C41",
    "phi41",
    "C43",
    "phi43",
    "C45",
    "phi45",
    "C50",
    "C52",
    "phi52",
    "C54",
    "phi54",
    "C56",
    "phi56",
)

#: Aliases for the most commonly used optical aberrations.
polar_aliases = {
    "defocus": "C10",
    "astigmatism": "C12",
    "astigmatism_angle": "phi12",
    "coma": "C21",
    "coma_angle": "phi21",
    "Cs": "C30",
    "C5": "C50",
}


def electron_wavelength_angstrom(E_eV):
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34

    lam = h / np.sqrt(2 * m * e * E_eV) / np.sqrt(1 + e * E_eV / 2 / m / c**2) * 10**10
    return lam


def _asnumpy(ar) -> NDArray[Any]:
    if config.get("has_cupy"):
        if isinstance(ar, cp.ndarray):
            return ar.get()
    return np.asarray(ar)


class ComplexProbe:
    """
    Complex Probe Class.

    Simplified version of CTF and Probe from abTEM:
    https://github.com/abTEM/abTEM/blob/master/abtem/transfer.py
    https://github.com/abTEM/abTEM/blob/master/abtem/waves.py

    Parameters
    ----------
    energy: float
        The electron energy of the wave functions this contrast transfer function will be applied to [eV].
    semiangle_cutoff: float
        The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].
    gpts : tuple[int,int]
        Number of grid points describing the wave functions.
    sampling : tuple[float,float]
        Lateral sampling of wave functions in Å
    device: str, optional
        Device to perform calculations on. Must be either 'cpu' or 'gpu'
    rolloff: float, optional
        Tapers the cutoff edge over the given angular range [mrad].
    vacuum_probe_intensity: np.ndarray, optional
        Squared of corner-centered aperture amplitude to use, instead of semiangle_cutoff + rolloff
    force_spatial_frequencies: np.ndarray, optional
        Corner-centered spatial frequencies. Useful for creating shifted probes necessary in direct ptychography.
    focal_spread: float, optional
        The 1/e width of the focal spread due to chromatic aberration and lens current instability [Å].
    angular_spread: float, optional
        The 1/e width of the angular deviations due to source size [mrad].
    gaussian_spread: float, optional
        The 1/e width image deflections due to vibrations and thermal magnetic noise [Å].
    phase_shift : float, optional
        A constant phase shift [radians].
    parameters: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in Å
        and angles should be given in radians.
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    def __init__(
        self,
        energy: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        semiangle_cutoff: float = np.inf,
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray | None = None,
        force_spatial_frequencies: tuple[np.ndarray, np.ndarray] | None = None,
        device: str = "cpu",
        focal_spread: float = 0.0,
        angular_spread: float = 0.0,
        gaussian_spread: float = 0.0,
        phase_shift: float = 0.0,
        parameters: dict[str, float] | None = None,
        **kwargs,
    ):
        if device == "cpu":
            self._xp = np
        elif device == "gpu" and config.get("has_cupy"):
            self._xp = cp
        else:
            raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()):
                raise ValueError("{} not a recognized parameter".format(key))

        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._force_spatial_frequencies = force_spatial_frequencies
        self._semiangle_cutoff = semiangle_cutoff
        self._rolloff = rolloff
        self._focal_spread = focal_spread
        self._angular_spread = angular_spread
        self._gaussian_spread = gaussian_spread
        self._phase_shift = phase_shift
        self._energy = energy
        self._wavelength = electron_wavelength_angstrom(energy)
        self._gpts = gpts
        self._sampling = sampling
        self._device = device

        self._parameters: dict[str, float] = dict(
            zip(polar_symbols, [0.0] * len(polar_symbols))
        )

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)
        self.set_parameters(parameters)

    def set_parameters(self, parameters: dict):
        """
        Set the phase of the phase aberration.
        Parameters
        ----------
        parameters: dict
            Mapping from aberration symbols to their corresponding values.
        """

        for symbol, value in parameters.items():
            if symbol in self._parameters.keys():
                self._parameters[symbol] = value

            elif symbol == "defocus":
                self._parameters[polar_aliases[symbol]] = -value

            elif symbol in polar_aliases.keys():
                self._parameters[polar_aliases[symbol]] = value

            else:
                raise ValueError("{} not a recognized parameter".format(symbol))

        return parameters

    def evaluate_aperture(
        self, alpha: float | np.ndarray, phi: float | np.ndarray | None = None
    ) -> float | np.ndarray:
        xp = self._xp
        semiangle_cutoff = self._semiangle_cutoff / 1000

        if self._vacuum_probe_intensity is not None:
            if self._force_spatial_frequencies is not None:
                vacuum_probe_intensity = get_shifted_ar(
                    xp.asarray(self._vacuum_probe_intensity, dtype=xp.float32),
                    self._origin[0],
                    self._origin[1],
                    bilinear=False,
                    device=self._device,
                )
            else:
                vacuum_probe_intensity = xp.asarray(
                    self._vacuum_probe_intensity, dtype=xp.float32
                )
            vacuum_probe_amplitude = xp.sqrt(xp.maximum(vacuum_probe_intensity, 0))
            return vacuum_probe_amplitude

        if self._semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        if self._rolloff > 0.0:
            rolloff = self._rolloff / 1000.0  # * semiangle_cutoff
            array = 0.5 * (
                1 + xp.cos(np.pi * (alpha - semiangle_cutoff + rolloff) / rolloff)
            )
            array[alpha > semiangle_cutoff] = 0.0
            array = xp.where(
                alpha > semiangle_cutoff - rolloff,
                array,
                xp.ones_like(alpha, dtype=xp.float32),
            )
        else:
            array = xp.array(alpha < semiangle_cutoff).astype(xp.float32)
        return array

    def evaluate_temporal_envelope(
        self, alpha: float | np.ndarray
    ) -> float | np.ndarray:
        xp = self._xp
        return xp.exp(
            -((0.5 * xp.pi / self._wavelength * self._focal_spread * alpha**2) ** 2)
        ).astype(xp.float32)

    def evaluate_gaussian_envelope(
        self, alpha: float | np.ndarray
    ) -> float | np.ndarray:
        xp = self._xp
        return xp.exp(-0.5 * self._gaussian_spread**2 * alpha**2 / self._wavelength**2)

    def evaluate_spatial_envelope(
        self, alpha: float | np.ndarray, phi: float | np.ndarray
    ) -> float | np.ndarray:
        xp = self._xp
        p = self._parameters
        dchi_dk = (
            2
            * xp.pi
            / self._wavelength
            * (
                (p["C12"] * xp.cos(2.0 * (phi - p["phi12"])) + p["C10"]) * alpha
                + (
                    p["C23"] * xp.cos(3.0 * (phi - p["phi23"]))
                    + p["C21"] * xp.cos(1.0 * (phi - p["phi21"]))
                )
                * alpha**2
                + (
                    p["C34"] * xp.cos(4.0 * (phi - p["phi34"]))
                    + p["C32"] * xp.cos(2.0 * (phi - p["phi32"]))
                    + p["C30"]
                )
                * alpha**3
                + (
                    p["C45"] * xp.cos(5.0 * (phi - p["phi45"]))
                    + p["C43"] * xp.cos(3.0 * (phi - p["phi43"]))
                    + p["C41"] * xp.cos(1.0 * (phi - p["phi41"]))
                )
                * alpha**4
                + (
                    p["C56"] * xp.cos(6.0 * (phi - p["phi56"]))
                    + p["C54"] * xp.cos(4.0 * (phi - p["phi54"]))
                    + p["C52"] * xp.cos(2.0 * (phi - p["phi52"]))
                    + p["C50"]
                )
                * alpha**5
            )
        )

        dchi_dphi = (
            -2
            * xp.pi
            / self._wavelength
            * (
                1 / 2.0 * (2.0 * p["C12"] * xp.sin(2.0 * (phi - p["phi12"]))) * alpha
                + 1
                / 3.0
                * (
                    3.0 * p["C23"] * xp.sin(3.0 * (phi - p["phi23"]))
                    + 1.0 * p["C21"] * xp.sin(1.0 * (phi - p["phi21"]))
                )
                * alpha**2
                + 1
                / 4.0
                * (
                    4.0 * p["C34"] * xp.sin(4.0 * (phi - p["phi34"]))
                    + 2.0 * p["C32"] * xp.sin(2.0 * (phi - p["phi32"]))
                )
                * alpha**3
                + 1
                / 5.0
                * (
                    5.0 * p["C45"] * xp.sin(5.0 * (phi - p["phi45"]))
                    + 3.0 * p["C43"] * xp.sin(3.0 * (phi - p["phi43"]))
                    + 1.0 * p["C41"] * xp.sin(1.0 * (phi - p["phi41"]))
                )
                * alpha**4
                + 1
                / 6.0
                * (
                    6.0 * p["C56"] * xp.sin(6.0 * (phi - p["phi56"]))
                    + 4.0 * p["C54"] * xp.sin(4.0 * (phi - p["phi54"]))
                    + 2.0 * p["C52"] * xp.sin(2.0 * (phi - p["phi52"]))
                )
                * alpha**5
            )
        )

        return xp.exp(
            -xp.sign(self._angular_spread)
            * (self._angular_spread / 2 / 1000) ** 2
            * (dchi_dk**2 + dchi_dphi**2)
        )

    def evaluate_chi(
        self, alpha: float | np.ndarray, phi: float | np.ndarray
    ) -> float | np.ndarray:
        xp = self._xp
        p = self._parameters

        alpha2 = alpha**2
        alpha = xp.array(alpha)

        array = xp.zeros(alpha.shape, dtype=np.float32)
        if any([p[symbol] != 0.0 for symbol in ("C10", "C12", "phi12")]):
            array += (
                1 / 2 * alpha2 * (p["C10"] + p["C12"] * xp.cos(2 * (phi - p["phi12"])))
            )

        if any([p[symbol] != 0.0 for symbol in ("C21", "phi21", "C23", "phi23")]):
            array += (
                1
                / 3
                * alpha2
                * alpha
                * (
                    p["C21"] * xp.cos(phi - p["phi21"])
                    + p["C23"] * xp.cos(3 * (phi - p["phi23"]))
                )
            )

        if any(
            [p[symbol] != 0.0 for symbol in ("C30", "C32", "phi32", "C34", "phi34")]
        ):
            array += (
                1
                / 4
                * alpha2**2
                * (
                    p["C30"]
                    + p["C32"] * xp.cos(2 * (phi - p["phi32"]))
                    + p["C34"] * xp.cos(4 * (phi - p["phi34"]))
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
                * alpha2**2
                * alpha
                * (
                    p["C41"] * xp.cos((phi - p["phi41"]))
                    + p["C43"] * xp.cos(3 * (phi - p["phi43"]))
                    + p["C45"] * xp.cos(5 * (phi - p["phi45"]))
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
                * alpha2**3
                * (
                    p["C50"]
                    + p["C52"] * xp.cos(2 * (phi - p["phi52"]))
                    + p["C54"] * xp.cos(4 * (phi - p["phi54"]))
                    + p["C56"] * xp.cos(6 * (phi - p["phi56"]))
                )
            )

        array = 2 * xp.pi / self._wavelength * array + self._phase_shift
        return array

    def evaluate_aberrations(
        self, alpha: float | np.ndarray, phi: float | np.ndarray
    ) -> float | np.ndarray:
        xp = self._xp
        return xp.exp(-1.0j * self.evaluate_chi(alpha, phi))

    def evaluate(
        self, alpha: float | np.ndarray, phi: float | np.ndarray
    ) -> float | np.ndarray:
        array = self.evaluate_aberrations(alpha, phi)

        if self._semiangle_cutoff < np.inf or self._vacuum_probe_intensity is not None:
            array *= self.evaluate_aperture(alpha, phi)

        if self._focal_spread > 0.0:
            array *= self.evaluate_temporal_envelope(alpha)

        if self._angular_spread > 0.0:
            array *= self.evaluate_spatial_envelope(alpha, phi)

        if self._gaussian_spread > 0.0:
            array *= self.evaluate_gaussian_envelope(alpha)

        return array

    def _evaluate_ctf(self):
        alpha, phi = self.get_scattering_angles()

        array = self.evaluate(alpha, phi)
        return array

    def get_scattering_angles(self):
        kx, ky = self.get_spatial_frequencies()
        alpha, phi = self.polar_coordinates(
            kx * self._wavelength, ky * self._wavelength
        )
        return alpha, phi

    def get_spatial_frequencies(self):
        xp = self._xp
        if self._force_spatial_frequencies is None:
            kx, ky = spatial_frequencies(self._gpts, self._sampling, xp)
        else:
            kx, ky = self._force_spatial_frequencies
            kx = xp.asarray(kx).astype(xp.float32)
            ky = xp.asarray(ky).astype(xp.float32)

            def find_zero_crossing(x):
                n = x.shape[0]
                y0, y1 = np.argsort(np.abs(x))[:2]
                x0, x1 = x[y0], x[y1]
                y = (y0 * x1 - y1 * x0) / (x1 - x0)
                dy = np.mod(y + n / 2, n) - n / 2
                return dy

            self._origin = tuple(find_zero_crossing(k) for k in [kx, ky])

        return kx, ky

    def polar_coordinates(self, x, y):
        """Calculate a polar grid for a given Cartesian grid."""
        xp = self._xp
        alpha = xp.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
        phi = xp.arctan2(y[None, :], x[:, None])
        return alpha, phi

    def build(self):
        """Builds corner-centered complex probe in the center of the region of interest."""
        xp = self._xp
        array = xp.fft.ifft2(self._evaluate_ctf())
        array = array / xp.sqrt((xp.abs(array) ** 2).sum())
        self._array = array
        return self

    def visualize(self, **kwargs):
        """Plots the probe intensity."""
        xp = self._xp

        cmap = kwargs.get("cmap", "Greys_r")
        kwargs.pop("cmap", None)

        plt.imshow(
            _asnumpy(xp.abs(xp.fft.ifftshift(self._array)) ** 2),
            cmap=cmap,
            **kwargs,
        )
        return


def spatial_frequencies(gpts: tuple[int, int], sampling: tuple[float, float], xp=np):
    """
    Calculate spatial frequencies of a grid.

    Parameters
    ----------
    gpts: tuple of int
        Number of grid points.
    sampling: tuple of float
        Sampling of the potential [1 / Å].

    Returns
    -------
    tuple of arrays
    """

    return tuple(
        xp.fft.fftfreq(n, d).astype(xp.float32) for n, d in zip(gpts, sampling)
    )


def get_shifted_ar(ar, xshift, yshift, periodic=True, bilinear=False, device="cpu"):
    """
        Shifts array ar by the shift vector (xshift,yshift), using the either
    the Fourier shift theorem (i.e. with sinc interpolation), or bilinear
    resampling. Boundary conditions can be periodic or not.

    Args:
            ar (float): input array
            xshift (float): shift along axis 0 (x) in pixels
            yshift (float): shift along axis 1 (y) in pixels
            periodic (bool): flag for periodic boundary conditions
            bilinear (bool): flag for bilinear image shifts
            device(str): calculation device will be perfomed on. Must be 'cpu' or 'gpu'
        Returns:
            (array) the shifted array
    """
    if device == "gpu":
        xp = cp
    else:
        xp = np

    ar = xp.asarray(ar)

    # Apply image shift
    if bilinear is False:
        nx, ny = xp.shape(ar)
        qx, qy = make_Fourier_coords2D(nx, ny, 1)
        qx = xp.asarray(qx)
        qy = xp.asarray(qy)

        w = xp.exp(-(2j * xp.pi) * ((yshift * qy) + (xshift * qx)))
        shifted_ar = xp.real(xp.fft.ifft2((xp.fft.fft2(ar)) * w))

    else:
        xF = xp.floor(xshift).astype(int).item()
        yF = xp.floor(yshift).astype(int).item()
        wx = xshift - xF
        wy = yshift - yF

        shifted_ar = (
            xp.roll(ar, (xF, yF), axis=(0, 1)) * ((1 - wx) * (1 - wy))
            + xp.roll(ar, (xF + 1, yF), axis=(0, 1)) * ((wx) * (1 - wy))
            + xp.roll(ar, (xF, yF + 1), axis=(0, 1)) * ((1 - wx) * (wy))
            + xp.roll(ar, (xF + 1, yF + 1), axis=(0, 1)) * ((wx) * (wy))
        )

    if periodic is False:
        # Rounded coordinates for boundaries
        xR = (xp.round(xshift)).astype(int)
        yR = (xp.round(yshift)).astype(int)

        if xR > 0:
            shifted_ar[0:xR, :] = 0
        elif xR < 0:
            shifted_ar[xR:, :] = 0
        if yR > 0:
            shifted_ar[:, 0:yR] = 0
        elif yR < 0:
            shifted_ar[:, yR:] = 0

    return shifted_ar


def make_Fourier_coords2D(Nx: int, Ny: int, pixelSize: float | tuple[float, float] = 1):
    """
    Generates Fourier coordinates for a (Nx,Ny)-shaped 2D array.
        Specifying the pixelSize argument sets a unit size.
    """
    if isinstance(pixelSize, (tuple, list)):
        assert len(pixelSize) == 2, "pixelSize must either be a scalar or have length 2"
        pixelSize_x = pixelSize[0]
        pixelSize_y = pixelSize[1]
    else:
        pixelSize_x = pixelSize
        pixelSize_y = pixelSize

    qx = np.fft.fftfreq(Nx, pixelSize_x)
    qy = np.fft.fftfreq(Ny, pixelSize_y)
    qy, qx = np.meshgrid(qy, qx)
    return qx, qy
