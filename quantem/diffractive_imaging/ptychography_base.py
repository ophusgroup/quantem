from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

import numpy as np

from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import as_numpy, electron_wavelength_angstrom
from quantem.core.utils.validators import (
    validate_arr_gt,
    validate_array,
    validate_gt,
    validate_int,
    validate_np_len,
    validate_xplike,
)
from quantem.diffractive_imaging.ptycho_utils import (
    fourier_shift,
    fourier_translation_operator,
    generate_batches,
)

if TYPE_CHECKING:
    import cupy as cp
    import torch
else:
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
"""

# TODO 
# make a from_dataset4dstem class method
# look in pad to see what the steps are 
# set initial probe 
# preprocess 
# forward pass (be sure to have multislice and mixed state intrinsic)
# ptychoAD or ptychoGD 

class PtychographyBase(AutoSerialize):
    DEFAULT_PROBE_PARAMS = {
        "energy": None,
        "defocus": None,
        "semiangle_cutoff": None,
        "rolloff": 2,
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
        verbose: int | bool = True,
        object_type: Literal["complex", "pure_phase", "potential"] = "complex",
        num_probes: int = 1,
        num_slices: int = 1,
        slice_thicknesses: float | None = None,
        dtype_real: "np.dtype | torch.dtype | str" = "float32",
        dtype_complex: "np.dtype | torch.dtype | str" = "complex64",
    ):
        self.verbose = verbose
        self.object_type = object_type
        self.num_probes = num_probes
        self.num_slices = num_slices
        self.slice_thicknesses = slice_thicknesses

        # initializing attributes
        self.probe_params = self.DEFAULT_PROBE_PARAMS
        self._preprocessed: bool = False
        self._has_dset: bool = False
        self._has_initial_probe = False
        self._has_initial_complex_probe = False
        self._losses = []  # losses/errors across epochs
        self._recon_types = []  # tracking different recon types (e.g. pixelwise, DIP, etc.)
        self._lrs = []  # LRs/step_sizes across epochs

        self._object_padding_force_power2_level: int = 0  # forces padding to be divisble by 2^n
        self._store_iterations: bool = False
        self._store_iterations_every: int = 1
        self._recon_iterations: list = []

    # region --- preprocessing ---

    # endregion

    # region --- ptychography forward ---

    # endregion

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
                self.slice_thicknesses = np.array([-1])
        elif isinstance(val, (float, int)):
            val = validate_gt(float(val), 0, "slice_thicknesses")
            self._slice_thicknesses = val * np.ones(self.num_slices - 1)
        else:
            arr = validate_array(
                as_numpy(val),
                name="slice_thicknesses",
                dtype=config.get("dtype_real"),
                ndim=1,
                shape=(self.num_slices - 1,),
            )
            arr = validate_arr_gt(arr, 0, "slice_thicknesses")
            arr = validate_np_len(arr, self.num_slices - 1, name="slice_thicknesses")
            self._slice_thicknesses = arr

        if not hasattr(self, "_propagators"):
            raise NotImplementedError("propagators not precomputed")
            # self._precompute_propagator_arrays()

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
        return as_numpy(self._object)

    @object.setter
    def object(self, obj: "np.ndarray | cp.ndarray") -> None:
        obj = validate_xplike(obj, "object")
        obj = validate_array(
            obj,
            name="object",
            dtype=self._object_dtype,
            ndim=self.object_shape_full.ndim,
            shape=self.object_shape_full,
            expand_dims=True,
        )
        masked_obj = np.abs(obj) * np.exp(1.0j * np.angle(obj) * self._object_fov_mask)
        self._object = masked_obj.astype(self._object_dtype)

    @property
    def object_padding_px(self) -> np.ndarray:
        return self._object_padding_px

    @object_padding_px.setter
    def object_padding_px(self, pad):
        pad = as_numpy(
            validate_array(
                validate_np_len(pad, 2, name="object_padding_px"),
                dtype="int16",
                ndim=1,
                name="object_padding_px",
            )
        )
        if self._object_padding_force_power2_level > 0:
            pad = adjust_padding_power2(
                pad,
                self.object_shape_full,
                self._object_padding_force_power2_level,
            )
        self._object_padding_px = pad

    @property
    def object_fov_mask(self) -> np.ndarray:
        return self._object_fov_mask

    @object_fov_mask.setter
    def object_fov_mask(self, mask: np.ndarray):
        mask = as_numpy(
            validate_array(
                mask,
                dtype=config.get("dtype_real"),
                ndim=self.object_shape_full.ndim,
                name="object_fov_mask",
                expand_dims=True,
            )
        )
        self._object_fov_mask = mask.astype(config.get("dtype_real"))

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
    def recon_types(self) -> np.ndarray:  # TODO rename to recon_type_iters
        """
        Keeping track of what reconstruction type was used
        """
        return np.array(self._recon_types)

    @property
    def lrs(self) -> np.ndarray:  # TODO rename to lr_iters
        """
        List of step sizes/LRs depending on recon type
        """
        return np.array(self._lrs)

    # endregion

    # region --- implicit class properties ---

    @property
    def device(self) -> str:
        return config.get("device")

    @property
    def _object_dtype(self) -> str:
        if self.object_type == "potential":
            return config.get("dtype_real")
        else:
            return config.get("dtype_complex")

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
                    sampling / electron_wavelength_angstrom(self.probe_params["energy"]) / 1e3
                )
            else:
                raise ValueError("dc units given in mrad but no energy defined to convert to A^-1")
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
                    sampling * electron_wavelength_angstrom(self.probe_params["energy"]) * 1e3
                )
            else:
                raise ValueError("dc units given in A^-1 but no energy defined to convert to mrad")
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

    # endregion

    # region --- class methods ---
    def vprint(self, *args, **kwargs) -> None:
        """Print messages if verbose is enabled."""
        if self.verbose:
            print(*args, **kwargs)

    def _check_rm_preprocessed(self, new_val: Any, name: str) -> None:
        if hasattr(self, name):
            if getattr(self, name) != new_val:
                self._preprocessed = False

    def _check_dset(self):
        if not self._has_dset:
            raise AttributeError(
                "No Dataset4dstem attached. Run Ptycho.attach_dset(Dataset4dstem)"
            )

    def _check_preprocessed(self):
        if not self._preprocessed:
            raise AttributeError(
                "Preprocessing has not been completed. Please run Ptycho.preprocess()"
            )


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
