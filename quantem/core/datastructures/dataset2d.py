from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.utils.validators import ensure_valid_array


class Dataset2d(Dataset):
    """2D dataset class that inherits from Dataset.

    This class represents all 2D datasets

    Attributes
    ----------
    None beyond base Dataset.
    """

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        """Initialize a 2D dataset.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 2D array data
        name : str
            A descriptive name for the dataset
        origin : NDArray | tuple | list | float | int
            The origin coordinates for each dimension
        sampling : NDArray | tuple | list | float | int
            The sampling rate/spacing for each dimension
        units : list[str] | tuple | list
            Units for each dimension
        signal_units : str, optional
            Units for the array values, by default "arb. units"
        _token : object | None, optional
            Token to prevent direct instantiation, by default None
        """
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
        )

    @classmethod
    def from_file(cls, file_path: str, file_type: str | None = None) -> "Dataset2d":
        """
        Load a Dataset2d from a file using the appropriate file reader.

        Parameters
        ----------
        file_path : str
            Path to the data file.
        file_type : str | None, optional
            File type hint. If None, the format will be inferred automatically.

        Returns
        -------
        Dataset2d
            The loaded dataset.
        """
        # Import here to avoid circular imports
        from quantem.core.io.file_readers import read_2d

        return read_2d(file_path, file_type)

    @classmethod
    def from_array(
        cls,
        array: NDArray | Any,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """
        Create a new Dataset2d from an array.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 4D array data
        name : str | None, optional
            A descriptive name for the dataset. If None, defaults to "2D dataset"
        origin : NDArray | tuple | list | float | int | None, optional
            The origin coordinates for each dimension. If None, defaults to zeros
        sampling : NDArray | tuple | list | float | int | None, optional
            The sampling rate/spacing for each dimension. If None, defaults to ones
        units : list[str] | tuple | list | None, optional
            Units for each dimension. If None, defaults to ["pixels"] * 4
        signal_units : str, optional
            Units for the array values, by default "arb. units"

        Returns
        -------
        Dataset2d
            A new Dataset2d instance
        """
        array = ensure_valid_array(array, ndim=2)
        return cls(
            array=array,
            name=name if name is not None else "2D dataset",
            origin=origin if origin is not None else np.zeros(2),
            sampling=sampling if sampling is not None else np.ones(2),
            units=units if units is not None else ["pixels"] * 2,
            signal_units=signal_units,
            _token=cls._token,
        )

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, int],
        name: str = "empty 2D dataset",
        fill_value: float = 0.0,
        origin: NDArray = None,
        sampling: NDArray = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a new Dataset2d filled with a constant value."""
        array = np.full(shape, fill_value, dtype=np.float32)
        return cls.from_array(
            array=array,
            name=name,
            origin=origin if origin is not None else np.zeros(2),
            sampling=sampling if sampling is not None else np.ones(2),
            units=units if units is not None else ["pixels", "pixels"],
            signal_units=signal_units,
        )
