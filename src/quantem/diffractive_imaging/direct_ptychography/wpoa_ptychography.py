import math
from typing import Tuple

import torch
from numpy.typing import NDArray

from quantem.core.datastructures import Dataset2d, Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.scattering_utils import electron_wavelength_angstrom
from quantem.core.utils.utils import SimpleTorchBatcher
from quantem.core.utils.validators import ensure_valid_tensor
from quantem.diffractive_imaging.aberration_surfaces import chi_taylor_expansion
from quantem.diffractive_imaging.origin_models import CenterOfMassOriginModel


class WPOAPtychography(AutoSerialize):
    """ """

    _token = object()

    def __init__(
        self,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float,
        _token: object | None = None,
    ):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use WPOAPtychography.from_dataset4dstem() or WPOAPtychography.from_virtual_bfs() to instantiate this class."
            )

        self.vbf_stack = vbf_dataset.array
        self.bf_mask = bf_mask_dataset.array

        self.scan_sampling = vbf_dataset.sampling[-2:]
        self.scan_gpts = vbf_dataset.shape[-2:]
        self.num_bf = vbf_dataset.shape[0]

        self.reciprocal_sampling = bf_mask_dataset.sampling
        self.semiangle_cutoff = semiangle_cutoff
        self.gpts = bf_mask_dataset.shape
        self.sampling = tuple(
            1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts)
        )
        self.wavelength = electron_wavelength_angstrom(energy)

        self.rotation_angle = rotation_angle
        self.coefs = aberration_coefs

    @classmethod
    def from_virtual_bfs(
        cls,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float,
    ):
        """ """

        return cls(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            _token=cls._token,
        )

    @classmethod
    def from_dataset4dstem(
        cls,
        dataset,
        energy: float,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float,
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bicubic",
        force_measured_origin: Tuple[float, float]
        | torch.Tensor
        | NDArray
        | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float = 0.5,
    ):
        """ """

        origin = CenterOfMassOriginModel.from_dataset(dataset)

        # measure and fit origin
        if force_fitted_origin is None:
            if force_measured_origin is None:
                origin.calculate_origin(max_batch_size)
            else:
                origin.origin_measured = force_measured_origin
            origin.fit_origin_background(fit_method=fit_method)
        else:
            origin.origin_fitted = force_fitted_origin

        # shift to origin
        origin.shift_origin_to(
            max_batch_size=max_batch_size,
            mode=mode,
        )
        shifted_tensor = origin.shifted_tensor

        # bf_mask
        mean_dp = shifted_tensor.mean(dim=(0, 1))
        bf_mask = mean_dp > mean_dp.max() * intensity_threshold

        bf_mask_dataset = Dataset2d.from_array(
            bf_mask.to(torch.int),
            name="BF mask",
            units=("A^-1", "A^-1"),
            sampling=dataset.sampling[-2:],
        )

        # vbf_stack
        vbf_stack = shifted_tensor[..., bf_mask]
        vbf_stack = vbf_stack / vbf_stack.mean((0, 1)) - 1
        vbf_stack = torch.moveaxis(vbf_stack, (0, 1, 2), (1, 2, 0))

        vbf_dataset = Dataset3d.from_array(
            vbf_stack,
            name="vBF stack",
            units=("index", "A", "A"),
            sampling=(1,) + tuple(dataset.sampling[:2]),
        )

        return cls(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            _token=cls._token,
        )

    @property
    def vbf_stack(self) -> torch.Tensor:
        return self._vbf_stack

    @vbf_stack.setter
    def vbf_stack(self, value: torch.Tensor):
        self._vbf_stack = ensure_valid_tensor(value, dtype=torch.float)

    @property
    def bf_mask(self) -> torch.Tensor:
        return self._bf_mask

    @bf_mask.setter
    def bf_mask(self, value: torch.Tensor):
        self._bf_mask = ensure_valid_tensor(value, dtype=torch.bool)

    @property
    def rotation_angle(self) -> float:
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value: float):
        self._rotation_angle = float(value)

    @property
    def coefs(self) -> dict:
        return self._coefs

    @coefs.setter
    def coefs(self, value: dict):
        self._coefs = dict(value)

    def preprocess(
        self,
    ):
        """ """

        self._bf_inds_i, self._bf_inds_j = torch.where(self.bf_mask)
        self._vbf_fourier = torch.fft.fft2(self.vbf_stack, norm="ortho")

        self._kxa, self._kya = torch.meshgrid(
            torch.fft.fftfreq(self.gpts[0], self.sampling[0]),
            torch.fft.fftfreq(self.gpts[1], self.sampling[1]),
            indexing="ij",
        )

        self._k_probe = self.semiangle_cutoff * 1e-3 / self.wavelength

        self.corrected_stack = None

        return self

    def _return_upsampled_qgrid(
        self,
        upsampling_factor=None,
    ):
        """
        Assumes integer upsampling factor.
        """

        if upsampling_factor is None:
            scan_gpts = self.scan_gpts
            scan_sampling = self.scan_sampling
        else:
            scan_gpts = tuple(n * upsampling_factor for n in self.scan_gpts)
            scan_sampling = tuple(s / upsampling_factor for s in self.scan_sampling)

        qxa, qya = torch.meshgrid(
            torch.fft.fftfreq(scan_gpts[0], scan_sampling[0]),
            torch.fft.fftfreq(scan_gpts[1], scan_sampling[1]),
            indexing="ij",
        )

        return qxa, qya

    def _parallax_correction_quadratic(
        self,
        coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
        max_batch_size=None,
    ):
        """ """
        if coefs is None:
            coefs = self.coefs
        coefs = {key: coefs[key] for key in ("C10", "C12", "phi12") if key in coefs}

        if rotation_angle is None:
            rotation_angle = self.rotation_angle

        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)

        if coefs:
            _, dx, dy = chi_taylor_expansion(
                self._kxa,
                self._kya,
                self.wavelength,
                rotation_angle,
                coefs,
                include_gradient=True,
                include_hessian=False,
            )
            grad_k = torch.stack((dx[self.bf_mask], dy[self.bf_mask]), -1)

            qxa, qya = self._return_upsampled_qgrid(upsampling_factor)
            qvec = torch.stack((qxa, qya), 0)
            grad_kq = torch.einsum("na,amp->nmp", grad_k, qvec)
            operator = torch.exp(-1j * grad_kq)

            if max_batch_size is None:
                max_batch_size = self.num_bf

            batcher = SimpleTorchBatcher(
                torch.arange(self.num_bf), batch_size=max_batch_size, shuffle=False
            )

            corrected_stack = torch.empty((self.num_bf,) + qxa.shape)
            for batch_idx in batcher:
                vbf_fourier = torch.tile(
                    self._vbf_fourier[batch_idx],
                    (1, upsampling_factor, upsampling_factor),
                )
                corrected_stack[batch_idx] = (
                    torch.fft.ifft2(
                        vbf_fourier * operator[batch_idx], norm="ortho"
                    ).real
                    * upsampling_factor
                )

            self.corrected_stack = corrected_stack
        return self

    def _parallax_correction_higher_order(
        self,
        coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
        max_batch_size=None,
    ):
        """ """
        if coefs is None:
            coefs = self.coefs

        if rotation_angle is None:
            rotation_angle = self.rotation_angle

        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)

        if coefs:
            _, dx, dy, hxx, hxy, hyy = chi_taylor_expansion(
                self._kxa,
                self._kya,
                self.wavelength,
                rotation_angle,
                coefs,
                include_gradient=True,
                include_hessian=True,
            )
            grad_k = torch.stack((dx[self.bf_mask], dy[self.bf_mask]), -1)
            hess_k = torch.stack(
                [
                    torch.stack([hxx[self.bf_mask], hxy[self.bf_mask]], dim=-1),
                    torch.stack([hxy[self.bf_mask], hyy[self.bf_mask]], dim=-1),
                ],
                dim=-2,
            )

            qxa, qya = self._return_upsampled_qgrid(upsampling_factor)
            qvec = torch.stack((qxa, qya), 0)
            grad_kq = torch.einsum("na,amp->nmp", grad_k, qvec)

            hess_kq = torch.einsum("amp,nab,bmp->nmp", qvec, hess_k, qvec) / 2

            (chi_q,) = chi_taylor_expansion(
                qxa,
                qya,
                self.wavelength,
                0,  # note fixed coordinate system
                coefs,
                include_gradient=False,
                include_hessian=False,
            )
            operator = torch.exp(-1j * (grad_kq + hess_kq - chi_q))

            if max_batch_size is None:
                max_batch_size = self.num_bf

            batcher = SimpleTorchBatcher(
                torch.arange(self.num_bf), batch_size=max_batch_size, shuffle=False
            )

            corrected_stack = torch.empty((self.num_bf,) + qxa.shape)
            for batch_idx in batcher:
                vbf_fourier = torch.tile(
                    self._vbf_fourier[batch_idx],
                    (1, upsampling_factor, upsampling_factor),
                )
                corrected_stack[batch_idx] = (
                    torch.fft.ifft2(
                        vbf_fourier * operator[batch_idx], norm="ortho"
                    ).real
                    * upsampling_factor
                )

            self.corrected_stack = corrected_stack
        return self

    def parallax_correction(
        self,
        coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
        quadratic_approximation=True,
        max_batch_size=None,
    ):
        """ """
        if quadratic_approximation:
            self._parallax_correction_quadratic(
                coefs, upsampling_factor, rotation_angle, max_batch_size
            )
        else:
            self._parallax_correction_higher_order(
                coefs, upsampling_factor, rotation_angle, max_batch_size
            )
        return self

    @staticmethod
    def _evaluate_probe(
        qxa,
        qya,
        wavelength,
        q_probe,
        reciprocal_sampling,
        rotation_angle,
        coefs,
        normalize=False,
    ):
        """soft aperture + chi"""
        q = torch.sqrt(qxa**2 + qya**2)
        aperture = torch.sqrt(
            torch.clip(
                (q_probe - q) / reciprocal_sampling + 0.5,
                0,
                1,
            ),
        )
        if normalize:
            aperture /= aperture.abs().square().sum().sqrt()

        (chi,) = chi_taylor_expansion(
            qxa,
            qya,
            wavelength,
            rotation_angle,
            coefs,
            include_gradient=False,
            include_hessian=False,
        )
        return aperture * torch.exp(-1j * chi)

    @staticmethod
    def _evaluate_gamma_factor(
        qmks,
        qpks,
        cmplx_probe_at_k,
        wavelength,
        q_probe,
        reciprocal_sampling,
        rotation_angle,
        coefs,
        normalize=False,
    ):
        """ """
        qmkxa, qmkya = qmks
        qpkxa, qpkya = qpks

        probe_m = WPOAPtychography._evaluate_probe(
            qmkxa,
            qmkya,
            wavelength,
            q_probe,
            reciprocal_sampling,
            rotation_angle,
            coefs,
            normalize=normalize,
        )

        probe_p = WPOAPtychography._evaluate_probe(
            qpkxa,
            qpkya,
            wavelength,
            q_probe,
            reciprocal_sampling,
            rotation_angle,
            coefs,
            normalize=normalize,
        )

        gamma = probe_m * cmplx_probe_at_k.conj() - probe_p.conj() * cmplx_probe_at_k
        gamma /= gamma.abs().clamp(min=1e-8)
        return gamma.conj()

    def single_sideband_correction(
        self,
        coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
    ):
        if coefs is None:
            coefs = self.coefs

        if rotation_angle is None:
            rotation_angle = self.rotation_angle

        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)

        cmplx_probe = self._evaluate_probe(
            self._kxa,
            self._kya,
            self.wavelength,
            self._k_probe,
            self.reciprocal_sampling[0],
            rotation_angle,
            coefs,
            normalize=False,
        )

        cos_angle = math.cos(rotation_angle)
        sin_angle = math.sin(rotation_angle)
        qxa, qya = self._return_upsampled_qgrid(upsampling_factor)
        qxa, qya = qxa * cos_angle + qya * sin_angle, -qxa * sin_angle + qya * cos_angle

        corrected_stack = torch.empty((self.num_bf,) + qxa.shape)
        for n, (ind_i, ind_j) in enumerate(zip(self._bf_inds_i, self._bf_inds_j)):
            qmks = qxa - self._kxa[ind_i, ind_j], qya - self._kya[ind_i, ind_j]
            qpks = qxa + self._kxa[ind_i, ind_j], qya + self._kya[ind_i, ind_j]
            cmplx_probe_at_k = cmplx_probe[ind_i, ind_j]

            gamma = self._evaluate_gamma_factor(
                qmks,
                qpks,
                cmplx_probe_at_k,
                self.wavelength,
                self._k_probe,
                self.reciprocal_sampling[0],
                rotation_angle,
                coefs,
                normalize=False,
            )

            vbf_fourier = torch.tile(
                self._vbf_fourier[n], (upsampling_factor, upsampling_factor)
            )

            corrected_stack[n] = (
                torch.fft.ifft2(vbf_fourier * gamma, norm="ortho").imag
                * upsampling_factor
            )

        self.corrected_stack = corrected_stack
        return self
