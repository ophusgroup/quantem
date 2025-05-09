from typing import TYPE_CHECKING

from quantem.core import config
from quantem.diffractive_imaging.ptychography_base import PtychographyBase

if TYPE_CHECKING:
    pass
else:
    if config.get("has_torch"):
        pass


class PtychographyML(PtychographyBase):
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._object_padding_force_power2: bool = (
            True  # TODO might have to overload padding?
        )
        self._object_padding_force_power2_level: int = 3
