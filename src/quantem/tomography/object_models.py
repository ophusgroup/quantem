from abc import abstractmethod
from typing import Any, Self

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.loss_functions import get_loss_function


class ObjectBase(AutoSerialize):
    """
    Base class for object models in tomography.
    """

    def __init__(
        self,
        tilt_series: Any,
        recon_volume: Any,
        device: str = "cpu",
        _token: str = None,
    ):
        super().__init__(_token=_token)
        self.tilt_series = tilt_series
        self.recon_volume = recon_volume
        self.device = device

    @abstractmethod
    def reconstruct(self) -> Self:
        """
        Abstract method for reconstruction.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("Reconstruction method must be implemented in subclasses.")


class ObjectConstraints(ObjectBase):
    """
    Class for object models with constraints in tomography.
    """

    DEFAULT_CONSTRAINTS = {
        "enforce_positivity": True,
    }


class ObjectVoxelwise(ObjectBase):
    """
    Class for voxel-wise object models in tomography.
    """

    def __init__(
        self,
        tilt_series: Any,
        recon_volume: Any,
        device: str = "cpu",
        _token: str = None,
    ):
        super().__init__(tilt_series, recon_volume, device, _token)
        self._loss_function = get_loss_function(config.tomography.loss_function)
        self._optimizer = None
        self._scheduler = None

    def reconstruct(self) -> Self:
        """
        Abstract method for voxel-wise reconstruction.
        Must be implemented in subclasses.
        """
        raise NotImplementedError(
            "Voxel-wise reconstruction method must be implemented in subclasses."
        )
