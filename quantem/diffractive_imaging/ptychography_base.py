from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from mpl_toolkits.axes_grid1 import ImageGrid

from quantem.core import config
from quantem.core.datastructures import Dataset, Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import as_numpy, electron_wavelength_angstrom, tqdmnd
from quantem.core.utils.validators import (
    validate_arr_gt,
    validate_array,
    validate_dict_keys,
    validate_gt,
    validate_int,
    validate_np_len,
    validate_xplike,
)
from quantem.diffractive_imaging.complexprobe import (
    POLAR_ALIASES,
    POLAR_SYMBOLS,
    ComplexProbe,
)
from quantem.diffractive_imaging.ptycho_utils import (
    AffineTransform,
    center_crop_arr,
    fit_origin,
    fourier_shift,
    generate_batches,
    get_array_module,
    get_shifted_array,
    sum_patches,
)

if TYPE_CHECKING:
    import cupy as cp
    import torch
else:
    if config.get("has_cupy"):
        import cupy as cp
    if config.get("has_torch"):
        import torch

"""
large scale design patterns:
    - all returns will be numpy arrays
    - hidden attributes, e.g. _object, will be numpy/cupy
        - for ptychography_AD, will overload object to allow for torch tensors to continue 
          with same optimizer # TODO see if this is necessary
    - objects are always 3D, if doing a singleslice recon, the shape is just [1, :, :]
    - likewise probes are always stacks for mixed state, if single probe, then shape is [1, :, :]
    - all preprocessing will be done with numpy/cupy arrays 
    - devices will be handled with config.get("device"), nothing stored internally 
        - probably need to do some checking for if a cupy is passed and device=cpu
        - but if device=cuda:0 and a np array is passed, can convert to cupy 
        - also going to default/hardcode to storage on cpu, e.g. for recon_iters and initial object
    - just going to store the initial probe as a numpy array, if giving the probe params then will 
      create the ComplexProbe and only keep the np array 
"""

# TODO
# forward pass (be sure to have multislice and mixed state intrinsic)
# ptychoAD
# ptychoGD
# testing for cases


class PtychographyBase(AutoSerialize):
    DEFAULT_PROBE_PARAMS = {
        "energy": None,
        "defocus": None,
        "semiangle_cutoff": None,
        "rolloff": 2,
        "polar_parameters": {},
    }

    """
    A base class for performing phase retrieval using the Ptychography algorithm.

    This class provides a basic framework for performing phase retrieval using the Ptychography algorithm.
    It is designed to be subclassed by specific Ptychography algorithms.

    Attributes:
        probe (Dataset): The probe function to be used in the algorithm
        data (Dataset4dstem): The data to be used in the algorithm
        device (str): The device to be used in the algorithm
    """

    def __init__(  # TODO prevent direct instantiation
        self,
        dset: Dataset4dstem,
        object_type: Literal["complex", "pure_phase", "potential"] = "complex",
        num_probes: int = 1,
        num_slices: int = 1,
        slice_thicknesses: float | None = None,
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        self.verbose = verbose
        self.dset = dset
        self.object_type = object_type
        self.num_probes = num_probes
        self.num_slices = num_slices
        self.slice_thicknesses = slice_thicknesses
        self.rng = rng

        # initializing attributes
        self._probe_params = self.DEFAULT_PROBE_PARAMS
        self._preprocessed: bool = False
        self._losses: list[float] = []  # losses/errors across epochs
        self._recon_types: list[
            str
        ] = []  # tracking different recon types (e.g. pixelwise, DIP, etc.)
        self._lrs: list[float] = []  # LRs/step_sizes across epochs

        # forces padding to be divisble by 2^n
        self._object_padding_force_power2_level: int = 0
        self._store_iterations: bool = False
        self._store_iterations_every: int = 1
        self._recon_iterations: list = []

    @classmethod
    def from_dataset4dstem(
        cls,
        dset: Dataset4dstem,
        object_type: Literal["complex", "pure_phase", "potential"] = "complex",
        probe_params: dict | None = None,
        probe: np.ndarray | None = None,
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
        num_probes: int = 1,
        num_slices: int = 1,
        slice_thicknesses: float | None = None,
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        slf = cls(
            dset=dset,
            object_type=object_type,
            num_probes=num_probes,
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            verbose=verbose,
            rng=rng,
        )
        slf.set_initial_probe(probe_params, probe, vacuum_probe_intensity)
        return slf

    # region --- preprocessing ---
    def set_initial_probe(
        self,
        probe_params: dict | None,
        probe: np.ndarray | None = None,
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
    ):
        self.vacuum_probe_intensity = vacuum_probe_intensity
        if probe_params is not None:
            if probe is not None:
                raise ValueError("cannot probe both directly and with probe_params")
            self.probe_params = probe_params
            prb = ComplexProbe(
                gpts=tuple(self.roi_shape),
                sampling=tuple(self.sampling),
                energy=self.probe_params["energy"],
                semiangle_cutoff=self.probe_params["semiangle_cutoff"],
                defocus=self.probe_params["defocus"],
                rolloff=self.probe_params["rolloff"],
                vacuum_probe_intensity=self.vacuum_probe_intensity,
                parameters=self.probe_params["polar_parameters"],
                device="cpu",
            )
            probes = prb.build()._array
            pass
        elif probe is not None:
            if isinstance(probe, ComplexProbe):
                # currently don't really support this, would need to store the initial complex
                # probe because gets re-set during preprocess if padding diffraction shape
                probes = probe.build()._array
            elif isinstance(probe, (np.ndarray, cp.ndarray)):
                probes = probe
            else:
                raise NotImplementedError(
                    f"setting probe currently supports np.ndarray or ComplexProbe, got {type(probe)}"
                )
            pass
        else:
            raise ValueError("must provide either probe_params or probe")

        if probes.ndim != 3:
            probes = probes[None]
        if probes.shape[0] != self.num_probes:
            probes = np.tile(probes, (self.num_probes, 1, 1))

        # apply random phase shifts for initializing mixed state
        for a0 in range(1, self.num_probes):
            shift_y = np.exp(
                -2j
                * np.pi
                * (self.rng.random() - 0.5)
                * np.fft.fftfreq(self.roi_shape[0])
            ).astype(config.get("dtype_complex"))
            shift_x = np.exp(
                -2j
                * np.pi
                * (self.rng.random() - 0.5)
                * np.fft.fftfreq(self.roi_shape[1])
            ).astype(config.get("dtype_complex"))
            probes[a0] = probes[a0] * shift_y[:, None] * shift_x[None]

        self.initial_probe = probes
        self.probe = probes

    def preprocess(
        self,
        object_padding_px: tuple[int, int] = (0, 0),
        com_fit_function: Literal[
            "none", "plane", "parabola", "bezier_two", "constant"
        ] = "plane",
        force_com_rotation: float | None = None,
        force_com_transpose: bool | None = None,
        padded_diffraction_intensities_shape: tuple[int, int] | None = None,
        plot_rotation: bool = True,
        plot_center_of_mass: str | bool = True,
        bilinear: bool = False,
        max_batch_size: int | None = None,
        vectorized: bool = True,
    ):
        """
        Rather than passing 100 flags here, I'm going to suggest that if users want to run very
        customized pre-processing, they just call the functions themselves directly.
        """

        self._check_dset()
        self._check_probe()

        # calculate CoM
        self._calculate_intensities_center_of_mass(
            self.raw_intensities,
            fit_function=com_fit_function,
            vectorized_calculation=vectorized,
        )
        self._solve_for_center_of_mass_relative_rotation(
            plot_rotation=plot_rotation,
            plot_center_of_mass=plot_center_of_mass,
            force_com_rotation=force_com_rotation,
            force_com_transpose=force_com_transpose,
        )

        # corner-center amplitudes
        self._normalize_diffraction_intensities(
            bilinear=bilinear,
        )

        if padded_diffraction_intensities_shape is not None:
            self._padded_diffraction_shape = np.array(
                padded_diffraction_intensities_shape
            )
            self.dset.pad(
                output_shape=padded_diffraction_intensities_shape, in_place=True
            )

            if self.vacuum_probe_intensity is not None:
                vppad = Dataset.from_array(np.fft.fftshift(self.vacuum_probe_intensity))
                vppad.pad(
                    output_shape=padded_diffraction_intensities_shape, in_place=True
                )
                self.vacuum_probe_intensity = np.fft.fftshift(vppad.array)
                self.set_initial_probe(  # set again because new roi shape
                    self.probe_params,
                    vacuum_probe_intensity=self.vacuum_probe_intensity,
                )
        else:
            self._padded_diffraction_shape = (
                0,
                0,
            )  # TODO should this just be roi shape?
            # depends on how/if it is used elsewhere

        ## Probe stuff
        self._normalize_initial_probe()

        # initialize probe positions
        self.object_padding_px = object_padding_px
        self._calculate_scan_positions_in_pixels(
            object_padding_px=self.object_padding_px,
        )
        self._set_patch_indices()

        # precompute propagators
        self._compute_propagator_arrays()

        # initialize object
        self._set_object_fov_mask(batch_size=max_batch_size)
        self.initial_object = np.ones(self.object_shape_full, dtype=self._object_dtype)
        self.object = self.initial_object.copy()

        self._preprocessed = True
        return

    def _calculate_intensities_center_of_mass(
        self,
        intensities: np.ndarray,
        dp_mask: np.ndarray | None = None,
        fit_function: Literal[
            "none", "plane", "parabola", "bezier_two", "constant"
        ] = "plane",
        com_shifts: np.ndarray | None = None,
        com_measured: np.ndarray | None = None,
        vectorized_calculation=True,
    ) -> None:
        """
        Common preprocessing function to compute and fit diffraction intensities CoM

        Parameters
        ----------
        intensities: (Rh,Rw,Qh,Qw) np.ndarray
            Raw intensities array stored on device, with dtype np.float32
        dp_mask: ndarray
            If not None, apply mask to datacube intensities
        fit_function: str, optional
            2D fitting function for CoM fitting. One of 'plane','parabola','bezier_two'
        com_shifts, tuple of ndarrays (CoMx measured, CoMy measured)
            If not None, com_shifts are fitted on the measured CoM values.
        com_measured: tuple of ndarrays (CoMx measured, CoMy measured)
            If not None, com_measured are passed as com_measured_x, com_measured_w
        vectorized_calculation: bool, optional
            If True (default), the calculation is vectorized

        Returns
        -------
        None
        """
        if com_measured is not None:
            com_measured_h = np.asarray(com_measured[0], dtype=config.get("dtype_real"))
            com_measured_w = np.asarray(com_measured[1], dtype=config.get("dtype_real"))
        else:
            if dp_mask is not None:
                if dp_mask.shape != intensities.shape[-2:]:
                    raise ValueError(
                        (
                            f"Mask shape should be (Qh,Qw):{intensities.shape[-2:]}, got {dp_mask.shape}"
                        )
                    )
                dp_mask = np.asarray(dp_mask, dtype=config.get("dtype_real"))

            # Coordinates
            kh = np.arange(intensities.shape[-2], dtype=config.get("dtype_real"))
            kw = np.arange(intensities.shape[-1], dtype=config.get("dtype_real"))
            kha, kwa = np.meshgrid(kh, kw, indexing="ij")

            if vectorized_calculation:
                # calculate CoM
                if dp_mask is not None:
                    intensities_mask = intensities * dp_mask
                else:
                    intensities_mask = intensities

                intensities_sum = np.sum(intensities_mask, axis=(-2, -1))

                com_measured_h = (
                    np.sum(intensities_mask * kha[None, None], axis=(-2, -1))
                    / intensities_sum
                )
                com_measured_w = (
                    np.sum(intensities_mask * kwa[None, None], axis=(-2, -1))
                    / intensities_sum
                )

            else:
                sh, sw = intensities.shape[:2]
                com_measured_h = np.zeros((sh, sw), dtype=config.get("dtype_real"))
                com_measured_w = np.zeros((sh, sw), dtype=config.get("dtype_real"))

                # loop of dps
                for rh, rw in tqdmnd(
                    range(sh),
                    range(sw),
                    desc="Calculating center of mass",
                    unit="probe position",
                    disable=not self._verbose,
                ):
                    masked_intensity = intensities[rh, rw]
                    if dp_mask is not None:
                        masked_intensity *= dp_mask
                    summed_intensity = masked_intensity.sum()
                    com_measured_h[rh, rw] = (
                        np.sum(masked_intensity * kwa) / summed_intensity
                    )
                    com_measured_w[rh, rw] = (
                        np.sum(masked_intensity * kha) / summed_intensity
                    )

        if com_shifts is None:
            if fit_function != "none":
                finite_mask = np.isfinite(com_measured_h)
                com_shifts_h, com_shifts_w, _com_res_h, _com_res_w = fit_origin(
                    data=(com_measured_h, com_measured_w),
                    fit_function=fit_function,
                    mask=finite_mask,
                )

                com_fitted_h = np.asarray(com_shifts_h, dtype=config.get("dtype_real"))
                com_fitted_w = np.asarray(com_shifts_w, dtype=config.get("dtype_real"))
            else:
                com_fitted_h = np.asarray(
                    com_measured_h, dtype=config.get("dtype_real")
                )
                com_fitted_w = np.asarray(
                    com_measured_w, dtype=config.get("dtype_real")
                )
        else:
            com_fitted_h = np.asarray(com_shifts[0], dtype=config.get("dtype_real"))
            com_fitted_w = np.asarray(com_shifts[1], dtype=config.get("dtype_real"))

        # fix CoM units
        com_normalized_h = (
            np.nan_to_num(com_measured_h - com_fitted_h) * self.reciprocal_sampling[0]
        )
        com_normalized_w = (
            np.nan_to_num(com_measured_w - com_fitted_w) * self.reciprocal_sampling[1]
        )

        self._com_measured: tuple[np.ndarray, np.ndarray] = (
            com_measured_h,
            com_measured_w,
        )  # raw measured pixels
        self._com_fitted: tuple[np.ndarray, np.ndarray] = (
            com_fitted_h,
            com_fitted_w,
        )  # fitted for descan, pixels
        # (measured - fitted) / reciprocal_sampling
        self._com_normalized = (com_normalized_h, com_normalized_w)
        return

    def _solve_for_center_of_mass_relative_rotation(
        self,
        rotation_angles_deg: np.ndarray | None = None,
        plot_rotation: bool = True,
        plot_center_of_mass: str | bool = "default",
        force_com_rotation: float | None = None,
        force_com_transpose: bool | None = None,
        **kwargs,
    ):
        """
        Common method to solve for the relative rotation between scan directions
        and the reciprocal coordinate system. We do this by minimizing the curl of the
        CoM gradient vector field or, alternatively, maximizing the divergence.

        Parameters
        ----------
        _com_measured_x: (Rx,Ry) xp.ndarray
            Measured horizontal center of mass gradient
        _com_measured_y: (Rx,Ry) xp.ndarray
            Measured vertical center of mass gradient
        _com_normalized_x: (Rx,Ry) xp.ndarray
            Normalized horizontal center of mass gradient
        _com_normalized_y: (Rx,Ry) xp.ndarray
            Normalized vertical center of mass gradient
        rotation_angles_deg: ndarray, optional
            Array of angles in degrees to perform curl minimization over
        plot_rotation: bool, optional
            If True, the CoM curl minimization search result will be displayed
        plot_center_of_mass: str, optional
            If 'default', the corrected CoM arrays will be displayed
            If 'all', the computed and fitted CoM arrays will be displayed
        force_com_rotation: float (degrees), optional
            Force relative rotation angle between real and reciprocal space
        force_com_transpose: bool, optional
            Force whether diffraction intensities need to be transposed.

        Returns
        --------
        _rotation_best_rad: float
            Rotation angle which minimizes CoM curl, in radians
        _rotation_best_transpose: bool
            Whether diffraction intensities need to be transposed to minimize CoM curl
        _com_x: xp.ndarray
            Corrected horizontal center of mass gradient, on calculation device
        _com_y: xp.ndarray
            Corrected vertical center of mass gradient, on calculation device

        Displays
        --------
        rotation_curl/div vs rotation_angles_deg, optional
            Vector calculus quantity being minimized/maximized
        com_measured_x/y, com_normalized_x/y and com_x/y, optional
            Measured and normalized CoM gradients
        rotation_best_deg, optional
            Summary statistics
        """

        # Helper functions
        def rotate_com_vectors(
            com: tuple[np.ndarray, np.ndarray],
            angle_rad: float,
            transpose: bool = False,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Rotate CoM vectors by angle_rad with optional transpose"""
            com_h, com_w = com
            if transpose:
                rotated_h = np.cos(angle_rad) * com_w - np.sin(angle_rad) * com_h
                rotated_w = np.sin(angle_rad) * com_w + np.cos(angle_rad) * com_h
            else:
                rotated_h = np.cos(angle_rad) * com_h - np.sin(angle_rad) * com_w
                rotated_w = np.sin(angle_rad) * com_h + np.cos(angle_rad) * com_w
            return rotated_h, rotated_w

        def calculate_curl(com_h: np.ndarray, com_w: np.ndarray) -> float:
            """Calculate curl of CoM gradient vector field"""
            grad_h_w = com_h[1:-1, 2:] - com_h[1:-1, :-2]  # dVh/dw
            grad_w_h = com_w[2:, 1:-1] - com_w[:-2, 1:-1]  # dVw/dh
            return float(np.mean(np.abs(grad_w_h - grad_h_w)))

        def calculate_curl_for_angles(
            angles_rad: np.ndarray,
            com_h: np.ndarray,
            com_w: np.ndarray,
            transpose: bool = False,
        ) -> np.ndarray:
            """Calculate curl for multiple angles"""
            angles_rad_expanded = angles_rad[:, None, None]

            if transpose:
                rotated_h = (
                    np.cos(angles_rad_expanded) * com_w[None]
                    - np.sin(angles_rad_expanded) * com_h[None]
                )
                rotated_w = (
                    np.sin(angles_rad_expanded) * com_w[None]
                    + np.cos(angles_rad_expanded) * com_h[None]
                )
            else:
                rotated_h = (
                    np.cos(angles_rad_expanded) * com_h[None]
                    - np.sin(angles_rad_expanded) * com_w[None]
                )
                rotated_w = (
                    np.sin(angles_rad_expanded) * com_h[None]
                    + np.cos(angles_rad_expanded) * com_w[None]
                )

            grad_h_w = rotated_h[:, 1:-1, 2:] - rotated_h[:, 1:-1, :-2]
            grad_w_h = rotated_w[:, 2:, 1:-1] - rotated_w[:, :-2, 1:-1]
            return np.mean(np.abs(grad_w_h - grad_h_w), axis=(-2, -1))

        def plot_curl_results(
            angles_deg: np.ndarray,
            curl_values: np.ndarray | tuple[np.ndarray, np.ndarray],
            best_angle: float,
            transpose: bool = False,
            **plot_kwargs,
        ) -> None:
            """Plot curl vs rotation angle"""
            figsize = plot_kwargs.get("figsize", (8, 2))
            fig, ax = plt.subplots(figsize=figsize)

            if isinstance(curl_values, tuple):
                ax.plot(angles_deg, curl_values[0], label="CoM")
                ax.plot(angles_deg, curl_values[1], label="CoM after transpose")
            else:
                label = "CoM after transpose" if transpose else "CoM"
                ax.plot(angles_deg, curl_values, label=label)

            y_range = ax.get_ylim()
            ax.plot(np.ones(2) * best_angle, y_range, color=(0, 0, 0, 1))

            ax.legend(loc="best")
            ax.set_xlabel("Rotation [degrees]")
            ax.set_ylabel("Mean Absolute Curl")

            if isinstance(curl_values, tuple):
                aspect_ratio = np.maximum(
                    np.ptp(curl_values[0]), np.ptp(curl_values[1])
                )
            else:
                aspect_ratio = np.ptp(curl_values)
            ax.set_aspect(np.ptp(angles_deg) / aspect_ratio / 4)

            fig.tight_layout()

        def plot_com_images(
            com_arrays: list[np.ndarray],
            titles: list[str],
            extent: list[float],
            **plot_kwargs,
        ) -> None:
            """Plot CoM vector fields"""
            if len(com_arrays) == 6:  # All CoM arrays
                figsize = plot_kwargs.pop("figsize", (8, 12))
                nrows, ncols = 3, 2
            else:  # Just corrected CoM
                figsize = plot_kwargs.pop("figsize", (8, 4))
                nrows, ncols = 1, 2

            cmap = plot_kwargs.pop("cmap", "RdBu_r")

            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=(0.25, 0.5))  # type:ignore

            for ax, arr, title in zip(grid, com_arrays, titles):  # type:ignore
                ax.imshow(arr, extent=extent, cmap=cmap, **plot_kwargs)
                ax.set_ylabel(f"x [{self.scan_units[0]}]")
                ax.set_xlabel(f"y [{self.scan_units[1]}]")
                ax.set_title(title)

        if rotation_angles_deg is None:
            rotation_angles_deg = np.arange(-89.0, 90.0, 1.0)

        rotation_angles_deg = np.asarray(
            rotation_angles_deg, dtype=config.get("dtype_real")
        )
        rotation_angles_rad = np.deg2rad(rotation_angles_deg)

        # Case 1: Known rotation
        if force_com_rotation is not None:
            _rotation_best_rad = np.deg2rad(force_com_rotation)
            self.vprint(
                f"Forcing best fit rotation to {force_com_rotation:.0f} degrees."
            )

            # Case 1.1: Known rotation and transpose
            if force_com_transpose is not None:
                _rotation_best_transpose = force_com_transpose

                self.vprint(
                    f"Forcing transpose of intensities to {force_com_transpose}."
                )

            # Case 1.2: Known rotation, unknown transpose
            else:
                # Calculate curl for both transpose options
                rotated_h, rotated_w = rotate_com_vectors(
                    self._com_normalized, _rotation_best_rad, transpose=False
                )
                rotation_curl = calculate_curl(rotated_h, rotated_w)

                rotated_h, rotated_w = rotate_com_vectors(
                    self._com_normalized, _rotation_best_rad, transpose=True
                )
                rotation_curl_transpose = calculate_curl(rotated_h, rotated_w)

                # Choose the option with minimum curl
                _rotation_best_transpose = rotation_curl_transpose < rotation_curl

                if _rotation_best_transpose:
                    self.vprint("Diffraction intensities should be transposed.")

        # Case 2: Unknown rotation
        else:
            # Case 2.1: Known transpose, unknown rotation
            if force_com_transpose is not None:
                _rotation_best_transpose = force_com_transpose

                self.vprint(
                    f"Forcing transpose of intensities to {force_com_transpose}."
                )

                # Calculate curl for all angles with known transpose
                curl_values = calculate_curl_for_angles(
                    rotation_angles_rad,
                    self._com_normalized[0],
                    self._com_normalized[1],
                    transpose=_rotation_best_transpose,
                )

                # Find angle with minimum curl
                min_index = np.argmin(curl_values).item()
                rotation_best_deg = rotation_angles_deg[min_index]
                _rotation_best_rad = rotation_angles_rad[min_index]

                self.vprint(
                    f"Calculated best fit rotation = {rotation_best_deg:.0f} degrees."
                )

                # Plot results if requested
                if plot_rotation:
                    plot_curl_results(
                        rotation_angles_deg,
                        curl_values,
                        rotation_best_deg,
                        transpose=_rotation_best_transpose,
                        **kwargs,
                    )

            else:
                # Case 2.2: Unknown rotation and transpose
                # Calculate curl for both transpose options
                rotation_curl = calculate_curl_for_angles(
                    rotation_angles_rad,
                    self._com_normalized[0],
                    self._com_normalized[1],
                    transpose=False,
                )

                rotation_curl_transpose = calculate_curl_for_angles(
                    rotation_angles_rad,
                    self._com_normalized[0],
                    self._com_normalized[1],
                    transpose=True,
                )

                # Minimize Curl
                ind_min = np.argmin(rotation_curl).item()
                ind_trans_min = np.argmin(rotation_curl_transpose).item()
                if rotation_curl[ind_min] <= rotation_curl_transpose[ind_trans_min]:
                    rotation_best_deg = rotation_angles_deg[ind_min]
                    _rotation_best_rad = rotation_angles_rad[ind_min]
                    _rotation_best_transpose = False
                else:
                    rotation_best_deg = rotation_angles_deg[ind_trans_min]
                    _rotation_best_rad = rotation_angles_rad[ind_trans_min]
                    _rotation_best_transpose = True

                self._rotation_angles_deg = rotation_angles_deg

                self.vprint(
                    f"Calculated best fit rotation = {rotation_best_deg:.0f} degrees."
                )
                if _rotation_best_transpose:
                    self.vprint("Diffraction intensities should be transposed.")

                if plot_rotation:
                    plot_curl_results(
                        rotation_angles_deg,
                        (rotation_curl, rotation_curl_transpose),
                        rotation_best_deg,
                        **kwargs,
                    )

        _com_h, _com_w = rotate_com_vectors(
            self._com_normalized,
            _rotation_best_rad,
            transpose=_rotation_best_transpose,
        )

        # Optionally, plot CoM
        if plot_center_of_mass == "all":
            extent = [  # TODO remove extent stuff
                0,
                self.scan_sampling[1] * self._com_measured[0].shape[1],
                self.scan_sampling[0] * self._com_measured[0].shape[0],
                0,
            ]
            plot_com_images(
                [
                    *self._com_measured,
                    *self._com_normalized,
                    _com_h,
                    _com_w,
                ],
                [
                    "CoM_h",
                    "CoM_w",
                    "Normalized CoM_h",
                    "Normalized CoM_w",
                    "Corrected CoM_h",
                    "Corrected CoM_w",
                ],
                extent,
                **kwargs,
            )
        elif plot_center_of_mass == "default" or plot_center_of_mass is True:
            extent = [
                0,
                self.scan_sampling[1] * _com_h.shape[1],
                self.scan_sampling[0] * _com_h.shape[0],
                0,
            ]
            plot_com_images(
                [_com_h, _com_w],
                ["Corrected CoM_h", "Corrected CoM_w"],
                extent,
                **kwargs,
            )

        self.com_rotation_rad = _rotation_best_rad
        self.com_transpose = _rotation_best_transpose
        self._com = _com_h, _com_w  # com_normalized rotated by com_rotation_rad
        return

    def _normalize_diffraction_intensities(
        self,
        positions_mask: np.ndarray | None = None,
        crop_patterns: bool = False,
        bilinear: bool = False,
        return_intensities_instead: bool = False,
    ):
        """
        Fix diffraction intensities CoM, shift to origin, and take square root

        Parameters
        ----------
        diffraction_intensities: (Rx,Ry,Sx,Sy) np.ndarray
            Zero-padded diffraction intensities
        com_fitted_h: (Rx,Ry) xp.ndarray
            Best fit horizontal center of mass gradient
        com_fitted_w: (Rx,Ry) xp.ndarray
            Best fit vertical center of mass gradient
        positions_mask: np.ndarray
            Boolean real space mask to select positions in datacube to skip for reconstruction
        crop_patterns: bool
            If True, patterns are cropped to avoid wrap around of patterns

        Returns
        -------
        diffraction_intensities: (Rx * Ry, Sx, Sy) np.ndarray
            Flat array of normalized diffraction amplitudes
        mean_intensity: float
            Mean intensity value
        crop_mask
            Mask to crop diffraction patterns with
        """

        mean_intensity = 0

        diff_intensities = self.raw_intensities.copy().astype(config.get("dtype_real"))

        com_fitted_h, com_fitted_w = self._com_fitted

        # Aggressive cropping for when off-centered high scattering angle data was recorded
        if crop_patterns:
            crop_h = int(
                np.minimum(
                    diff_intensities.shape[2] - com_fitted_h.max(),
                    com_fitted_h.min(),
                )
            )
            crop_w = int(
                np.minimum(
                    diff_intensities.shape[3] - com_fitted_w.max(),
                    com_fitted_w.min(),
                )
            )

            crop_m = np.minimum(crop_w, crop_h)

            crop_mask = np.zeros(self.roi_shape, dtype="bool")
            crop_mask[:crop_m, :crop_m] = True
            crop_mask[-crop_m:, :crop_m] = True
            crop_mask[:crop_m:, -crop_m:] = True
            crop_mask[-crop_m:, -crop_m:] = True

            crop_mask_shape = (crop_m * 2, crop_m * 2)

        else:
            crop_mask = None
            crop_mask_shape = self.roi_shape

        for rh, rw in tqdmnd(
            range(diff_intensities.shape[0]),
            range(diff_intensities.shape[1]),
            desc="Normalizing intensities",
            unit="probe position",
            disable=not self._verbose,
        ):
            if positions_mask is not None:
                if not positions_mask[rh, rw]:
                    continue

            intensities = get_shifted_array(
                diff_intensities[rh, rw],
                -(com_fitted_h[rh, rw] + 0.5),
                -(com_fitted_w[rh, rw] + 0.5),
                bilinear=bilinear,
            )

            mean_intensity += np.sum(intensities)
            if return_intensities_instead:
                diff_intensities[rh, rw] = np.maximum(intensities, 0)
            else:
                diff_intensities[rh, rw] = np.sqrt(np.maximum(intensities, 0))

        if positions_mask is not None:
            diff_intensities = diff_intensities[positions_mask]
        else:
            qh, qw = self.roi_shape
            diff_intensities = diff_intensities.reshape((-1, qh, qw))

        if crop_patterns:
            diff_intensities = diff_intensities[:, crop_mask].reshape(
                (-1, *crop_mask_shape)
            )

        mean_intensity /= diff_intensities.shape[0]

        self.shifted_amplitudes = diff_intensities
        self._mean_diffraction_intensity = mean_intensity
        self._crop_mask = crop_mask
        self._crop_mask_shape = crop_mask_shape
        return

    def _normalize_initial_probe(self):
        # Normalize probe to match mean diffraction intensity
        if hasattr(self, "_mean_diffraction_intensity"):
            probe = self.initial_probe.copy()
            probe_intensity = np.sum(np.abs(np.fft.fft2(probe)) ** 2)
            intensity_norm = np.sqrt(self._mean_diffraction_intensity / probe_intensity)
            probe *= intensity_norm
            self.initial_probe = probe.astype(config.get("dtype_complex"))
        else:
            raise AttributeError(
                "_mean_diffraction_intensity has not yet been set. Run preprocess."
            )

    def _calculate_scan_positions_in_pixels(
        self,
        positions: np.ndarray | None = None,
        positions_mask: np.ndarray | None = None,
        object_padding_px: np.ndarray | None = None,
        positions_offset_ang: tuple[float, float] | None = None,
    ):
        """
        Method to compute the initial guess of scan positions in pixels.

        Parameters
        ----------
        positions: (J,2) np.ndarray or None
            Input probe positions in Å.
            If None, a raster scan using experimental parameters is constructed.
        positions_mask: np.ndarray, optional
            Boolean real space mask to select positions in datacube to skip for reconstruction
        object_padding_px: Tuple[int,int], optional
            Pixel dimensions to pad object with
            If None, the padding is set to half the probe ROI dimensions
        positions_offset_ang, np.ndarray, optional
            Offset of positions in A

        Returns
        -------
        positions_in_px: (J,2) np.ndarray
            Initial guess of scan positions in pixels
        object_padding_px: Tupe[int,int]
            Updated object_padding_px
        """

        if object_padding_px is None:
            object_padding_px = np.array([0, 0])

        if positions is None:
            nx, ny = self.gpts
            sx, sy = self.scan_sampling
            x = np.arange(nx) * sx
            y = np.arange(ny) * sy

            x, y = np.meshgrid(x, y, indexing="ij")
            if positions_offset_ang is not None:
                x += positions_offset_ang[0]
                y += positions_offset_ang[1]

            if positions_mask is not None:
                x = x[positions_mask]
                y = y[positions_mask]

            positions = np.stack((x.ravel(), y.ravel()), axis=-1)
        else:
            positions = np.array(positions)

        if self.com_rotation_rad != 0:
            tf = AffineTransform(angle=self.com_rotation_rad)
            positions = tf(positions, positions.mean(0))

        sampling = self.sampling
        if self.com_transpose:
            positions = np.flip(positions, axis=1)
            sampling = sampling[::-1]

        # ensure positive
        m: np.ndarray = np.min(positions, axis=0).clip(-np.inf, 0)
        positions -= m

        # finally, switch to pixels
        positions[:, 0] /= sampling[0]
        positions[:, 1] /= sampling[1]

        # top-left padding
        positions[:, 0] += object_padding_px[0]
        positions[:, 1] += object_padding_px[1]

        self.positions_px = positions
        return

    def _compute_propagator_arrays(
        self,
        theta_x: float | None = None,
        theta_y: float | None = None,
    ):
        """
        Precomputes propagator arrays complex wave-function will be convolved by,
        for all slice thicknesses.

        Parameters
        ----------
        theta_x: float, optional
            x tilt of propagator in mrad
        theta_y: float, optional
            y tilt of propagator in mrad

        Returns
        -------
        propagator_arrays: np.ndarray
            (T,Sx,Sy) shape array storing propagator arrays
        """

        if self.num_slices == 1:
            self.propagators = np.array([])
            return

        kh, kw = tuple(
            np.fft.fftfreq(n, d) for n, d in zip(self.roi_shape, self.sampling)
        )

        wavelength = electron_wavelength_angstrom(self.probe_params["energy"])
        propagators = np.empty(
            (self.num_slices - 1, kh.shape[0], kw.shape[0]), dtype=np.complex64
        )

        for i, dz in enumerate(self.slice_thicknesses):
            propagators[i] = np.exp(
                1.0j * (-(kh**2)[:, None] * np.pi * wavelength * dz)
            )
            propagators[i] *= np.exp(1.0j * (-(kw**2)[None] * np.pi * wavelength * dz))

            if theta_x is not None:
                propagators[i] *= np.exp(
                    1.0j * (-2 * kh[:, None] * np.pi * dz * np.tan(theta_x / 1e3))
                )

            if theta_y is not None:
                propagators[i] *= np.exp(
                    1.0j * (-2 * kw[None] * np.pi * dz * np.tan(theta_y / 1e3))
                )
        self.propagators = propagators
        return

    def _set_patch_indices(self):
        x0 = np.round(self.positions_px[:, 0]).astype(np.int32)  # can fix here
        y0 = np.round(self.positions_px[:, 1]).astype(np.int32)

        x_ind = np.fft.fftfreq(self.roi_shape[0], d=1 / self.roi_shape[0]).astype(
            np.int32
        )
        y_ind = np.fft.fftfreq(self.roi_shape[1], d=1 / self.roi_shape[1]).astype(
            np.int32
        )
        row = (x0[:, None, None] + x_ind[None, :, None]) % self.object_shape_full[-2]
        col = (y0[:, None, None] + y_ind[None, None, :]) % self.object_shape_full[-1]

        self.patch_row = row
        self.patch_col = col

    def _set_object_fov_mask(self, gaussian_sigma: float = 5.0, batch_size=None):
        overlap = self._get_probe_overlap(batch_size)
        ov = overlap > overlap.max() * 0.3
        ov = ndi.binary_closing(ov, iterations=10)
        ov = ndi.binary_dilation(ov, iterations=min(32, np.min(self.object_padding_px)))
        # dont want mask too small
        # small mask is especially problematic with smaller batch sizes because applied more often
        ov = ndi.gaussian_filter(
            ov.astype(config.get("dtype_real")), sigma=gaussian_sigma
        )
        self.object_fov_mask = ov
        return

    def _get_probe_overlap(self, max_batch_size: int | None = None) -> np.ndarray:
        prb: np.ndarray = self.probe[0]
        num_dps = int(np.prod(self.gpts))
        shifted_probes = np.broadcast_to(prb, (num_dps, *self.roi_shape))

        batch_size = num_dps if max_batch_size is None else int(max_batch_size)
        probe_overlap = np.zeros(
            self.object_shape_full[-2:], dtype=config.get("dtype_real")
        )
        for start, end in generate_batches(num_dps, max_batch=batch_size):
            probe_overlap += sum_patches(
                np.abs(shifted_probes[start:end]) ** 2,
                self.patch_row[start:end],
                self.patch_col[start:end],
                tuple(self.object_shape_full[-2:]),
            )
        return probe_overlap

    # endregion --- preprocessing ---

    # region --- explicit class properties ---
    @property
    def dset(self) -> Dataset4dstem:
        self._check_dset()
        return self._dset

    @dset.setter
    def dset(self, new_dset: Dataset4dstem):
        if not isinstance(new_dset, Dataset4dstem):
            raise TypeError(f"dset should be a Dataset4dstem, got {type(new_dset)}")
        self._dset = new_dset.copy()

    @property
    def shifted_amplitudes(self) -> np.ndarray:
        """
        gives the amplitudes that have had descan corrected and which are corner centered
        shaped as (rx*ry, qx, qy)
        """
        return self._shifted_amplitudes

    @shifted_amplitudes.setter
    def shifted_amplitudes(self, arr: np.ndarray):
        arr = validate_array(
            arr,
            name="shifted_amplitudes",
            dtype=config.get("dtype_real"),
            shape=(np.prod(self.gpts), *self.roi_shape),
        )
        self._shifted_amplitudes = arr

    @property
    def object_type(self) -> str:
        return self._object_type

    @object_type.setter
    def object_type(self, t: Literal["potential", "pure_phase", "complex"]):
        t_str = str(t).lower()
        if t_str in ["potential", "pot", "potentials"]:
            self._object_type = "potential"
        elif t_str in ["pure_phase", "purephase", "pure phase", "pure"]:
            self._object_type = "pure_phase"
        elif t_str in ["complex", "c"]:
            self._object_type = "complex"
        else:
            raise ValueError(
                f"Object type should be 'potential', 'complex', or 'pure_phase', got {t}"
            )
        return

    @property
    def num_slices(self) -> int:
        """if num_slices > 1, then it is multislice reconstruction"""
        return self._num_slices

    @num_slices.setter
    def num_slices(self, val: int) -> None:
        self._num_slices = validate_int(validate_gt(val, 0, "num_slices"), "num_slices")

    @property
    def propagators(self) -> np.ndarray:
        if self.num_slices == 1:
            return np.array([])
        else:
            return self._propagators

    @propagators.setter
    def propagators(self, prop: np.ndarray | list[np.ndarray]) -> None:
        if self.num_slices == 1:
            self._propagators = np.array([])
        else:
            prop = validate_xplike(prop, "propagators")
            prop = validate_array(
                prop,
                name="propagators",
                dtype=config.get("dtype_complex"),
                ndim=3,
                shape=(self.num_slices - 1, *self.roi_shape),
                expand_dims=False,
            )
            self._propagators = self._to_xp(prop)

    @property
    def num_probes(self) -> int:
        """if num_probes > 1, then it is a mixed-state reconstruction"""
        return self._num_probes

    @num_probes.setter
    def num_probes(self, val: int) -> None:
        new_num = validate_int(validate_gt(val, 0, "num_probes"), "num_probes")
        self._check_rm_preprocessed(
            new_num, "_num_probes"
        )  # TODO make sure this actually requires rerunning preprocess
        self._num_probes = new_num

    @property
    def slice_thicknesses(self) -> np.ndarray:
        return self._slice_thicknesses

    @slice_thicknesses.setter
    def slice_thicknesses(self, val: float | np.ndarray | None) -> None:
        if val is None:
            if self.num_slices > 1:
                raise ValueError(
                    f"num slices = {self.num_slices}, so slice_thicknesses cannot be None"
                )
            else:
                self._slice_thicknesses = np.array([-1])
        elif isinstance(val, (float, int)):
            val = validate_gt(float(val), 0, "slice_thicknesses")
            self._slice_thicknesses = val * np.ones(self.num_slices - 1)
        else:
            if self.num_slices == 1:
                warn("Single slice reconstruction so not setting slice_thicknesses")
            arr = validate_array(
                self._as_numpy(val),
                name="slice_thicknesses",
                dtype=config.get("dtype_real"),
                ndim=1,
                shape=(self.num_slices - 1,),
            )
            arr = validate_arr_gt(arr, 0, "slice_thicknesses")
            arr = validate_np_len(arr, self.num_slices - 1, name="slice_thicknesses")
            self._slice_thicknesses = arr

        if hasattr(
            self, "_propagators"
        ):  # propagators already set, update with new slices
            self._compute_propagator_arrays()

    @property
    def verbose(self) -> int:
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool | int | float) -> None:
        self._verbose = validate_int(validate_gt(v, -1, "verbose"), "verbose")

    @property
    def com_transpose(self) -> bool:
        "whether or not the dset has been transposed"
        return self._transpose

    @com_transpose.setter
    def com_transpose(self, t: bool) -> None:
        self._transpose = bool(t)

    @property
    def com_rotation_rad(self) -> float:
        "Best fit rotation of the dc"
        return self._com_rotation_rad

    @com_rotation_rad.setter
    def com_rotation_rad(self, rot: float) -> None:
        self._com_rotation_rad = float(rot)

    @property
    def object(self) -> np.ndarray:
        return self._as_numpy(self._object)

    @object.setter
    def object(self, obj: "np.ndarray | cp.ndarray | torch.Tensor") -> None:
        """Shape [num_slices, height, width]"""
        obj = validate_xplike(obj, "object")
        obj = validate_array(
            obj,
            name="object",
            dtype=self._object_dtype,
            ndim=3,
            shape=self.object_shape_full,
            expand_dims=True,
        )
        xp = get_array_module(obj)
        masked_obj = np.abs(obj) * np.exp(
            1.0j * np.angle(obj) * xp.asarray(self._object_fov_mask)
        )
        self._object = masked_obj.astype(self._object_dtype)

    # @property
    # def _initial_object(self) -> np.ndarray:
    #     """Shape [num_slices, height, width]"""
    #     return self._initial_object

    @property
    def object_padding_px(self) -> np.ndarray:
        return self._object_padding_px

    @object_padding_px.setter
    def object_padding_px(self, pad: np.ndarray | tuple[int, int]):
        p2 = validate_xplike(pad, "object_padding_px")
        p2 = self._as_numpy(
            validate_array(
                validate_np_len(p2, 2, name="object_padding_px"),
                dtype="int16",
                ndim=1,
                name="object_padding_px",
            )
        )
        if self._object_padding_force_power2_level > 0:
            p2 = adjust_padding_power2(
                p2,
                self.object_shape_crop,
                self._object_padding_force_power2_level,
            )
        self._object_padding_px = p2

    @property
    def object_fov_mask(self) -> np.ndarray:
        return self._object_fov_mask

    @object_fov_mask.setter
    def object_fov_mask(self, mask: np.ndarray):
        mask = self._as_numpy(
            validate_array(
                mask,
                dtype=config.get("dtype_real"),
                ndim=3,
                name="object_fov_mask",
                expand_dims=True,
            )
        )
        self._object_fov_mask = mask.astype(config.get("dtype_real"))

    @property
    def vacuum_probe_intensity(self) -> np.ndarray | None:
        """corner centered vacuum probe"""
        return self._vacuum_probe_intensity

    @vacuum_probe_intensity.setter
    def vacuum_probe_intensity(self, vp: np.ndarray | Dataset4dstem | None):
        if vp is None:
            self._vacuum_probe_intensity = None
            return
        elif isinstance(vp, np.ndarray):
            vp2 = vp.astype(config.get("dtype_real"))
        elif isinstance(vp, (Dataset4dstem, Dataset)):
            vp2 = vp.array
        else:
            raise NotImplementedError(f"Unknown vacuum probe type: {type(vp)}")

        if vp2.ndim == 4:
            vp2 = np.mean(vp2, axis=(0, 1))
        elif vp2.ndim != 2:
            raise ValueError(
                f"Weird number of dimensions for vacuum probe, shape: {vp.shape}"
            )

        # vacuum probe should be corner centered
        corner_vals = vp2[:10, :10].mean()
        if corner_vals < 0.01 * vp2.max():
            warn("Looks like vacuum probe is not corner centered, fft shifting now)")
        else:
            vp2 = np.fft.fftshift(vp2)

        # fix centering
        com: list | tuple = ndi.center_of_mass(vp2)
        vp2 = get_shifted_array(
            vp2,
            -com[0],
            -com[1],
            bilinear=True,
        )

        self._vacuum_probe_intensity = vp2

    @property
    def positions_px(self) -> np.ndarray:
        return self._positions_px

    @positions_px.setter
    def positions_px(self, pos):
        self._positions_px = np.array(pos, dtype=config.get("dtype_real"))

    @property
    def positions_px_fractional(self) -> np.ndarray:
        return self.positions_px - np.round(self.positions_px)

    @property
    def losses(self) -> np.ndarray:  # TODO rename to loss_iters
        """
        Loss/MSE error for each epoch regardless of reconstruction method used
        """
        return np.array(self._losses)

    @property
    def num_epochs(self) -> int:
        """
        Number of epochs for which the recon has been run so far
        """
        return len(self.losses)

    @property
    def recon_types(self) -> list[str]:  # TODO rename to recon_type_iters
        """
        Keeping track of what reconstruction type was used
        """
        return self._recon_types

    @property
    def lrs(self) -> np.ndarray:  # TODO rename to lr_iters
        """
        List of step sizes/LRs depending on recon type
        """
        return np.array(self._lrs)

    @property
    def probe(self) -> np.ndarray:
        """Complex valued probe(s). Shape [num_probes, roi_height, roi_width]"""
        self._check_probe()
        return self._probe

    @probe.setter
    def probe(self, prb: "np.ndarray|cp.ndarray"):
        prb = validate_xplike(prb, "probe")
        prb = validate_array(
            prb,
            name="probe",
            dtype=config.get("dtype_complex"),
            ndim=3,
            shape=(self.num_probes, *self.roi_shape),
            expand_dims=True,
        )
        self._probe = prb

    @property
    def initial_probe(self) -> np.ndarray:
        self._check_initial_probe()
        return self._initial_probe

    @initial_probe.setter
    def initial_probe(self, prb: "np.ndarray|cp.ndarray"):
        prb = validate_xplike(prb, "probe")
        prb = validate_array(
            prb,
            name="probe",
            dtype=config.get("dtype_complex"),
            ndim=3,
            shape=(self.num_probes, *self.roi_shape),
            expand_dims=True,
        )
        self._initial_probe = prb

    @property
    def probe_params(self) -> dict[str, Any]:
        return self._probe_params

    @probe_params.setter
    def probe_params(self, params: dict[str, Any] = {}):
        validate_dict_keys(
            params,
            [*self.DEFAULT_PROBE_PARAMS.keys(), *POLAR_SYMBOLS, *POLAR_ALIASES.keys()],
        )
        polar_parameters: dict[str, float] = dict(
            zip(POLAR_SYMBOLS, [0.0] * len(POLAR_SYMBOLS))
        )

        def process_polar_params(p):
            bads = []
            for symbol, value in p.items():
                if isinstance(value, dict):
                    process_polar_params(
                        value
                    )  # Recursively process nested dictionaries
                elif value is None:
                    continue
                elif symbol in polar_parameters.keys():
                    polar_parameters[symbol] = float(value)
                    bads.append(symbol)
                elif symbol == "defocus":
                    polar_parameters[POLAR_ALIASES[symbol]] = -1 * float(value)
                elif symbol in POLAR_ALIASES:
                    polar_parameters[POLAR_ALIASES[symbol]] = float(value)
                    bads.append(symbol)
            [p.pop(bad) for bad in bads]
            # Ignore other parameters (energy, semiangle_cutoff, etc.)

        process_polar_params(params)
        params["polar_parameters"] = polar_parameters
        self._probe_params = (
            self.DEFAULT_PROBE_PARAMS | self._probe_params | params
        )  # prioritize new values

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator | int | None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, float)):
            rng = np.random.default_rng(rng)
        elif not isinstance(rng, np.random.Generator):
            raise TypeError(
                f"rng should be a np.random.Generator or a seed, got {type(rng)}"
            )
        self._rng = rng

    @property
    def constraints(self) -> dict[str, Any]:
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: dict[str, Any]):
        """
        Set constraints for the object reconstruction.
        """
        if not isinstance(constraints, dict):
            raise TypeError("Constraints should be a dictionary.")
        # TODO add validation for constraints
        # move this to a mixin
        self._constraints = constraints

    @property
    def store_iterations(self) -> bool:
        return self._store_iterations

    @store_iterations.setter
    def store_iterations(self, val: bool) -> None:
        self._store_iterations = bool(val)

    @property
    def store_iterations_every(self) -> int:
        return self._store_iterations_every

    @store_iterations_every.setter
    def store_iterations_every(self, val: int) -> None:
        self._store_iterations_every = int(val)

    @property
    def recon_iterations(self) -> list[dict[str, np.ndarray]]:
        return self._recon_iterations

    def get_recon_by_iter(self, iteration: int):
        iteration = int(iteration)
        for snapshot in self.recon_iterations:
            if snapshot["iteration"] == iteration:
                return snapshot
        raise ValueError(f"No snapshot found at iteration: {iteration}")

    # endregion --- explicit class properties ---

    # region --- implicit class properties ---

    @property
    def raw_intensities(self) -> np.ndarray:
        """
        gives the raw dc measured intensities
        """
        return self.dset.array

    @property
    def device(self) -> str:
        """This should be of form 'cuda:X' or 'cpu', as defined by quantem.config"""
        return config.get("device")

    @property
    def _object_dtype(self) -> str:
        if self.object_type == "potential":
            return config.get("dtype_real")
        else:
            return config.get("dtype_complex")

    @property
    def object_cropped(self) -> np.ndarray:
        cropped = self._crop_rotate_object_fov(self.object)
        if self.object_type == "pure_phase":
            cropped = np.exp(1j * np.angle(cropped))
        cropped = center_crop_arr(
            cropped, tuple(self.object_shape_crop)
        )  # sometimes 1 pixel off
        return cropped

    @property
    def roi_shape(self) -> np.ndarray:
        self._check_dset()
        return np.array(self.dset.shape[2:])

    @property
    def gpts(self) -> np.ndarray:
        self._check_dset()
        return np.array(self.dset.shape[:2])

    @property
    def reciprocal_sampling(self) -> np.ndarray:
        """
        Units A^-1 or raises error
        """
        self._check_dset()
        sampling = self.dset.sampling[2:]
        units = self.dset.units[2:]
        if units[0] == "A^-1":
            pass
        elif units[0] == "mrad":
            if self.probe_params["energy"] is not None:  # convert mrad -> A^-1
                sampling = (
                    sampling
                    / electron_wavelength_angstrom(self.probe_params["energy"])
                    / 1e3
                )
            else:
                raise ValueError(
                    "dc units given in mrad but no energy defined to convert to A^-1"
                )
        elif units[0] == "pixels":
            raise ValueError("dset Q units given in pixels, needs calibration")
        else:
            raise NotImplementedError(f"Unknown dset Q units: {units}")
        return sampling

    @property
    def reciprocal_units(self) -> list[str]:
        """Hardcoded to A^-1, self.reciprocal_sampling will raise an error if can't get A^-1"""
        self._check_dset()
        return ["A^-1", "A^-1"]

    @property
    def angular_sampling(self) -> np.ndarray:
        """
        Units mrad or raises error
        """
        self._check_dset()
        sampling = self.dset.sampling[2:]
        units = self.dset.units[2:]
        if units[0] == "mrad":
            pass
        elif units[0] == "A^-1":
            if self.probe_params["energy"] is not None:
                sampling = (
                    sampling
                    * electron_wavelength_angstrom(self.probe_params["energy"])
                    * 1e3
                )
            else:
                raise ValueError(
                    "dc units given in A^-1 but no energy defined to convert to mrad"
                )
        elif units[0] == "pixels":
            raise ValueError("dset Q units given in pixels, needs calibration")
        else:
            raise NotImplementedError(f"Unknown dset Q units: {units}")
        return sampling

    @property
    def angular_units(self) -> list[str]:
        """Hardcoded to mrad, self.angular_sampling will raise an error if can't get mrad"""
        self._check_dset()
        return ["mrad", "mrad"]

    @property
    def sampling(self) -> np.ndarray:
        return 1 / (self.roi_shape * self.reciprocal_sampling)

    @property
    def scan_sampling(self) -> np.ndarray:
        self._check_dset()
        return self.dset.sampling[:2]

    @property
    def scan_units(self) -> list[str]:
        self._check_dset()
        return self.dset.units[:2]

    @property
    def fov(self) -> np.ndarray:
        self._check_dset()
        return self.scan_sampling * (self.gpts - 1)

    @property
    def object_shape_crop(self) -> np.ndarray:
        """All object shapes are 3D"""
        shp = np.floor(self.fov / self.sampling)
        shp += shp % 2
        shp = np.concatenate([[self.num_slices], shp])
        return shp.astype("int")
        # return np.array(self.object_cropped.shape)

    @property
    def object_shape_full(self) -> np.ndarray:
        cshape = self.object_shape_crop.copy()
        rotshape = np.floor(
            [
                abs(cshape[-1] * np.sin(self.com_rotation_rad))
                + abs(cshape[-2] * np.cos(self.com_rotation_rad)),
                abs(cshape[-2] * np.sin(self.com_rotation_rad))
                + abs(cshape[-1] * np.cos(self.com_rotation_rad)),
            ]
        )
        rotshape += rotshape % 2
        rotshape += 2 * self.object_padding_px
        shape = np.concatenate([[self.num_slices], rotshape])
        return shape.astype("int")

    # endregion --- implicit class properties ---

    # region --- class methods ---
    def vprint(self, *args, **kwargs) -> None:
        """Print messages if verbose is enabled."""
        if self.verbose:
            print(*args, **kwargs)

    def _to_xp(
        self, arr: "np.ndarray|cp.ndarray|list|tuple"
    ) -> "np.ndarray|cp.ndarray":
        """returns a copy of arr as a np or cp array"""
        if "cuda" in self.device:  # this should only be possible if "has_cupy"
            return cp.asarray(arr)
        elif self.device == "cpu":
            return self._as_numpy(arr)
        else:
            raise NotImplementedError(f"Unknown config device {self.device}")

    def _check_dset(self):
        if not hasattr(self, "_dset"):
            raise AttributeError(
                "No Dataset4dstem attached. Run Ptycho.attach_dset(Dataset4dstem)"
            )

    def _check_probe(self):
        if not hasattr(self, "_probe"):
            raise AttributeError("No probe set. Run Ptycho.set_initial_probe()")

    def _check_initial_probe(self):
        if not hasattr(self, "_initial_probe"):
            raise AttributeError("No initial probe set. Run Ptycho.set_initial_probe()")

    def _check_preprocessed(self):
        if not self._preprocessed:
            raise AttributeError(
                "Preprocessing has not been completed. Please run Ptycho.preprocess()"
            )

    def _check_rm_preprocessed(self, new_val: Any, name: str) -> None:
        if hasattr(self, name):
            if getattr(self, name) != new_val:
                self._preprocessed = False

    def _as_numpy(self, array: "np.ndarray | cp.ndarray"):
        return as_numpy(array)

    def _crop_rotate_object_fov(
        self,
        array: "np.ndarray|cp.ndarray",
        positions_px: np.ndarray | None = None,
        com_rotation_rad: float | None = None,
        transpose: bool | None = None,
        padding: int = 0,
    ) -> np.ndarray:
        """
        Crops and rotated object to FOV bounded by current pixel positions.
        """
        array = self._as_numpy(array).copy()
        com_rotation_rad = (
            self.com_rotation_rad if com_rotation_rad is None else com_rotation_rad
        )
        transpose = self.com_transpose if transpose is None else transpose

        angle = (
            self.com_rotation_rad if self.com_transpose else -1 * self.com_rotation_rad
        )

        if positions_px is None:
            positions = self.positions_px
        else:
            positions = positions_px

        tf = AffineTransform(angle=angle)
        rotated_points = tf(positions, origin=positions.mean(0))

        min_x, min_y = np.floor(np.amin(rotated_points, axis=0) - padding).astype("int")
        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0
        max_x, max_y = np.ceil(np.amax(rotated_points, axis=0) + padding).astype("int")

        rotated_array = ndi.rotate(
            array, np.rad2deg(-angle), order=1, reshape=False, axes=(-2, -1)
        )[..., min_x:max_x, min_y:max_y]

        if transpose:
            rotated_array = rotated_array.swapaxes(-2, -1)

        return rotated_array

    def _repeat_arr(
        self, arr: "np.ndarray|cp.ndarray|torch.Tensor", repeats: int, axis: int
    ) -> "np.ndarray|cp.ndarray|torch.Tensor":
        """repeat the input array along the desired axis."""
        if config.get("has_torch"):
            if isinstance(arr, torch.Tensor):
                return torch.repeat_interleave(arr, repeats, dim=axis)
        return np.repeat(arr, repeats, axis=axis)

    def reset_recon(self) -> None:
        self._losses = []
        self._recon_types = []
        self._recon_iterations = []
        self._lrs = []

    def append_recon_iteration(
        self,
        object: "torch.Tensor | np.ndarray | None" = None,
        probe: "torch.Tensor | np.ndarray | None" = None,
    ) -> None:
        if probe is None:
            prb = self.probe
        else:
            prb = self._as_numpy(probe)
        if object is None:
            obj = self.object
        else:
            obj = self._as_numpy(object)
        self._recon_iterations.append(
            {
                "iteration": self.num_epochs,
                "object": obj,
                "probe": prb,
            }
        )
        return

    def _mse(
        self,
        pred: "np.ndarray|cp.ndarray|torch.Tensor",
        truth: "np.ndarray|cp.ndarray|torch.Tensor",
    ) -> "np.ndarray|cp.ndarray|torch.Tensor":
        """Calculate the mean squared error between two arrays."""
        if type(pred) is not type(truth):
            raise TypeError(
                f"pred and truth should be of the same type, got {type(pred)} and {type(truth)}"
            )
        if config.get("has_torch"):
            if isinstance(pred, torch.Tensor) and isinstance(truth, torch.Tensor):
                return torch.mean(torch.abs(truth - pred) ** 2)
        return np.mean(np.abs(truth - pred) ** 2)

    # endregion

    # region --- ptychography forward model ---

    def forward_operator(
        self, obj, probe, patch_row, patch_col, fract_positions, descan_shifts=None
    ):
        shifted_input_probes = fourier_shift(probe, fract_positions)
        # initial shape: (batch_size, nprobes, roi_shape[0], roi_shape[1])
        print("shifted input_probes shape1 : ", shifted_input_probes.shape)
        # shifted_input probe shape: (nslices, nprobes, batch_size, roi_shape[0], roi_shape[1])
        shifted_input_probes = self._repeat_arr(
            shifted_input_probes.swapaxes(0, 1)[None], self.num_slices, 0
        )
        # shifted_input_probes = np.repeat(
        #     np.swapaxes(shifted_input_probes, 0, 1)[None], self.num_slices, 0
        # )
        print("shifted input_probes shape2: ", shifted_input_probes.shape)
        obj_patches = self._get_object_patches(obj, patch_row, patch_col)
        # obj_patches shape: (num_slices, batch_size, roi_shape[0], roi_shape[1])
        print("obj patches shape: ", obj_patches.shape)
        propagated_probes, overlap = self.overlap_projection(
            obj_patches, shifted_input_probes, descan_shifts
        )
        # same propagated_probes shape as shifted_input_probes
        # ov shape: (nslices, nprobes, batch_size, roi_shape[0], roi_shape[1])
        # print("overlap shape0: ", overlap.shape)
        # overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        # overlap = overlap[0] # remove the num_slices channel
        print(
            "propagated_probes shape: ",
            propagated_probes.shape,
            " overlap: ",
            overlap.shape,
        )
        if descan_shifts is not None:
            # TODO move applying plane wave descan shift here
            ### applying plane wave shift here to overlap, reduce need for extra FFT
            #     shifts = fourier_translation_operator(
            #         descan_shifts, self.roi_shape, device=self.device
            #     )
            #     shifts = shifts[:, None]
            #     overlap *= shifts
            raise NotImplementedError("move descan shift stuff to here")

        return obj_patches, propagated_probes, overlap

    def error_estimate(
        self,
        obj,
        probe,
        patch_row,
        patch_col,
        fract_positions,
        true_amplitudes,
        descan_shifts=None,
    ) -> "float":
        obj_patches, propagated_probes, overlap = self.forward_operator(
            obj,
            probe,
            patch_row,
            patch_col,
            fract_positions,
        )
        if descan_shifts is not None:
            ### applying plane wave shift here to overlap, reduce need for extra FFT
            #     shifts = fourier_translation_operator(
            #         descan_shifts, self.roi_shape, device=self.device
            #     )
            #     shifts = shifts[:, None]
            #     overlap *= shifts
            raise NotImplementedError("move descan shift stuff to here")

        farfield_amplitudes = self.estimate_amplitudes(overlap)
        return self._mse(farfield_amplitudes, true_amplitudes)  # type:ignore ## FIXME

    def _get_object_patches(self, obj_array, patch_row, patch_col):
        """Extracts complex-valued roi-shaped patches from `obj_array`."""
        if (
            self.object_type == "potential"
        ):  # TODO - should potential be pure_phase + positivity?
            obj_array = np.exp(1j * obj_array)
        patches = obj_array[..., patch_row, patch_col]
        # reshape to (batch_size, num_slices, roi_shape[0], roi_shape[1])
        return patches

    def overlap_projection(self, obj_patches, input_probes, descan_shifts=None):
        """Multiplies `input_probes` with roi-shaped patches from `obj_array`.
        This version is for GD only -- AD does not require all the propagated probe
        slices and trying to store them causes in-place issues
        """
        # shifted_input probe shape: (nslices, nprobes, batch_size, roi_shape[0], roi_shape[1])
        # obj_patches shape: (nslices, batch_size, roi_shape[0], roi_shape[1])
        overlap = obj_patches[0] * input_probes[0]
        for s in range(1, self.num_slices):
            input_probes[s] = self._propagate_array(overlap, self._propagators[s - 1])
            overlap = obj_patches[s] * input_probes[s]
        propagated_probes = input_probes

        return propagated_probes, overlap

    def estimate_amplitudes(self, overlap_array: "np.ndarray | cp.ndarray"):
        """Returns the estimated fourier amplitudes from real-valued `overlap_array`."""
        # overlap shape: (batch_size, nprobes, roi_shape[0], roi_shape[1])
        # incoherent sum of all probe components
        eps = 1e-9  # this is to avoid diverging gradients at sqrt(0)
        if config.get("has_torch"):
            if isinstance(overlap_array, torch.Tensor):
                overlap_fft = torch.fft.fft2(overlap_array)
                return torch.sqrt(torch.sum(torch.abs(overlap_fft + eps) ** 2, dim=0))
        overlap_fft = np.fft.fft2(overlap_array)
        return np.sqrt(np.sum(np.abs(overlap_fft + eps) ** 2, axis=0))

    def _error_from_overlap(
        self,
        overlap_array: "np.ndarray | cp.ndarray",
        true_amplitudes: "np.ndarray | cp.ndarray",
    ):
        farfield_amplitudes = self.estimate_amplitudes(overlap_array)
        return np.mean((farfield_amplitudes - true_amplitudes) ** 2)

    def _propagate_array(
        self, array: "np.ndarray|cp.ndarray", propagator_array: "np.ndarray|cp.ndarray"
    ) -> "np.ndarray|cp.ndarray":
        """
        Propagates array by Fourier convolving array with propagator_array.

        Parameters
        ----------
        array: np.ndarray
            Wavefunction array to be convolved
        propagator_array: np.ndarray
            Propagator array to convolve array with

        Returns
        -------
        propagated_array: np.ndarray
            Fourier-convolved array
        """
        propagated = np.fft.ifft2(np.fft.fft2(array) * propagator_array)
        return propagated

    # endregion


# misc helpers to maybe move elsewhere


def adjust_padding_power2(pad, shape, power2_level):
    """
    Adjusts pad so that (shape + 2*pad) is divisible by 2**power2_level.
    """
    div = 2**power2_level
    rem0 = (shape[-2] + 2 * pad[-2]) % div
    rem1 = (shape[-1] + 2 * pad[-1]) % div
    if rem0 != 0:
        pad[-2] += (div - rem0) // 2
    if rem1 != 0:
        pad[-1] += (div - rem1) // 2

    if ((shape[-2] + 2 * pad[-2]) % div != 0) or ((shape[-1] + 2 * pad[-1]) % div != 0):
        raise ValueError(f"Adjustment failed to achieve divisibility by {div}")
    return pad
