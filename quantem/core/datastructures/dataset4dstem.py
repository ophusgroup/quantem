import matplotlib.pyplot as plt
import numpy as np

from quantem.core import config
from quantem.core.datastructures.dataset import Dataset
from quantem.core.visualization.visualization_utils import ScalebarConfig

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


class Dataset4dstem(Dataset):
    """A 4D-STEM dataset class that inherits from Dataset.

    This class represents a 4D scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 4D array with dimensions (scan_y, scan_x, dp_y, dp_x).
    The first two dimensions represent real space scanning positions, while the latter
    two dimensions represent reciprocal space diffraction patterns.

    Attributes
    ----------
    virtual_images : dict[str, Dataset]
        Dictionary storing virtual images generated from the 4D-STEM dataset.
        Keys are image names and values are Dataset objects containing the images.
    """

    def __init__(
        self,
        array: np.ndarray | cp.ndarray,
        name: str,
        origin: np.ndarray,
        sampling: np.ndarray,
        units: list[str],
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        """Initialize a 4D-STEM dataset.

        Parameters
        ----------
        array : np.ndarray | cp.ndarray
            The underlying 4D array data
        name : str
            A descriptive name for the dataset
        origin : np.ndarray
            The origin coordinates for each dimension
        sampling : np.ndarray
            The sampling rate/spacing for each dimension
        units : list[str]
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
        self._virtual_images = {}

    @classmethod
    def from_file(cls, file_path: str, file_type: str) -> "Dataset4dstem":
        """
        Create a new Dataset4dstem from a file.

        Parameters
        ----------
        file_path : str
            Path to the data file
        file_type : str
            The type of file reader needed. See rosettasciio for supported formats
            https://hyperspy.org/rosettasciio/supported_formats/index.html

        Returns
        -------
        Dataset4dstem
            A new Dataset4dstem instance loaded from the file
        """
        # Import here to avoid circular imports
        from quantem.core.io.file_readers import read_4dstem

        return read_4dstem(file_path, file_type)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray | cp.ndarray,
        name: str | None = None,
        origin: np.ndarray | tuple | list | None = None,
        sampling: np.ndarray | tuple | list | None = None,
        units: list[str] | None = None,
        signal_units: str = "arb. units",
    ) -> "Dataset4dstem":
        """
        Create a new Dataset4dstem from an array.

        Parameters
        ----------
        array : np.ndarray | cp.ndarray
            The underlying 4D array data
        name : str | None, optional
            A descriptive name for the dataset. If None, defaults to "4D-STEM dataset"
        origin : np.ndarray | tuple | list | None, optional
            The origin coordinates for each dimension. If None, defaults to zeros
        sampling : np.ndarray | tuple | list | None, optional
            The sampling rate/spacing for each dimension. If None, defaults to ones
        units : list[str] | None, optional
            Units for each dimension. If None, defaults to ["pixels"] * 4
        signal_units : str, optional
            Units for the array values, by default "arb. units"

        Returns
        -------
        Dataset4dstem
            A new Dataset4dstem instance
        """
        return cls(
            array=array,
            name=name if name is not None else "4D-STEM dataset",
            origin=origin if origin is not None else np.zeros(4),
            sampling=sampling if sampling is not None else np.ones(4),
            units=units if units is not None else ["pixels"] * 4,
            signal_units=signal_units,
            _token=cls._token,
        )

    @property
    def virtual_images(self) -> dict[str, Dataset]:
        """Dictionary storing virtual images generated from the 4D-STEM dataset."""
        return self._virtual_images

    def __getitem__(self, index):
        """Simple indexing function to return Dataset view"""

        array_view = self.array[index]
        ndim = array_view.ndim
        calibrated_origin = self.origin.ndim == self.ndim

        return Dataset.from_array(
            array=array_view,
            name=self.name + str(index),
            origin=self.origin[index] if calibrated_origin else self.origin[-ndim:],
            sampling=self.sampling[-ndim:],
            units=self.units[-ndim:],
            signal_units=self.signal_units,
        )

    @property
    def dp_mean(self) -> Dataset:
        """
        Dataset containing the mean diffraction pattern
        """
        if hasattr(self, "_dp_mean"):
            return self._dp_mean
        else:
            print("Calculating dp_mean, attach with Dataset4dstem.get_dp_mean()")
            return self.get_dp_mean(attach=False)

    def get_dp_mean(self, attach: bool = True) -> Dataset:
        """
        Get mean diffraction pattern

        Parameters
        ----------
        attach: bool
            If True attachs mean diffraction pattern to self, callable with dataset.dp_mean

        Returns
        --------
        dp_mean: Dataset
            new Dataset with the mean diffraction pattern
        """
        dp_mean = self.mean((0, 1))

        dp_mean_dataset = Dataset.from_array(
            array=dp_mean,
            name=self.name + "_dp_mean",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self._dp_mean = dp_mean_dataset

        return dp_mean_dataset

    @property
    def dp_max(self) -> Dataset:
        """
        Dataset containing the max diffraction pattern
        """
        if hasattr(self, "_dp_max"):
            return self._dp_max
        else:
            print("Calculating dp_max, attach with Dataset4dstem.get_dp_max()")
            return self.get_dp_max(attach=False)

    def get_dp_max(self, attach: bool = True) -> Dataset:
        """
        Get max diffraction pattern

        Parameters
        ----------
        attach: bool
            If True attachs max diffraction pattern to dataset, callable with dataset.dp_max

        Returns
        --------
        dp_max: Dataset
            new Dataset with the max diffraction pattern
        """
        dp_max = self.max((0, 1))

        dp_max_dataset = Dataset.from_array(
            array=dp_max,
            name=self.name + "_dp_max",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self._dp_max = dp_max_dataset

        return dp_max_dataset

    @property
    def dp_median(self) -> Dataset:
        """
        Dataset containing the median diffraction pattern
        """
        if hasattr(self, "_dp_median"):
            return self._dp_median
        else:
            print("Calculating dp_median, attach with Dataset4dstem.get_dp_median()")
            return self.get_dp_median(attach=False)

    def get_dp_median(self, attach: bool = True) -> Dataset:
        """
        Get median diffraction pattern

        Parameters
        ----------
        attach: bool
            If True attachs median diffraction pattern to dataset

        Returns
        --------
        dp_median: Dataset
            new Dataset with the median diffraction pattern
        """
        dp_median = np.median(self.array, axis=(0, 1))

        dp_median_dataset = Dataset.from_array(
            array=dp_median,
            name=self.name + "_dp_median",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self._dp_median = dp_median_dataset

        return dp_median_dataset

    def get_virtual_image(
        self,
        mask: np.ndarray,
        name: str = "virtual_image",
        attach: bool = True,
    ) -> Dataset:
        """
        Get virtual image

        Parameters
        ----------
        mask: np.ndarray
            Mask for forming virtual images from 4D-STEM data. The mask should be the same
            shape as the datacube Kx and Ky
        attach: bool
            If True attachs median diffraction pattern to dataset
        name: string
            Name of virtual image. If None name is "virtual_image"

        Returns
        --------
        virtual image (if attach is True)
        """

        virtual_image = np.sum(self.array * mask, axis=(-1, -2))

        virtual_image_dataset = Dataset.from_array(
            array=virtual_image,
            name=name,
            origin=self.origin[0:2],
            sampling=self.sampling[0:2],
            units=self.units[0:2],
            signal_units=self.signal_units,
        )

        if attach is True:
            self._virtual_images[name] = virtual_image_dataset

        return virtual_image_dataset

    def show(
        self,
        scalebar: ScalebarConfig | bool = True,
        title: str | None = None,
        index=(0, 0),
        figax=None,
        axsize=(4, 4),
        **kwargs,
    ):
        """ """

        list_of_objs = [self[index]]
        if hasattr(self, "_dp_mean"):
            list_of_objs.append(self.dp_mean)
        if hasattr(self, "_dp_max"):
            list_of_objs.append(self.dp_max)
        if hasattr(self, "_dp_median"):
            list_of_objs.append(self.dp_median)

        ncols = len(list_of_objs)

        if figax is None:
            figsize = (axsize[0] * ncols, axsize[1])
            fig, axs = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
        else:
            fig, axs = figax
            if not isinstance(axs, np.ndarray):
                axs = np.array([[axs]])
            elif axs.ndim == 1:
                axs = axs.reshape(1, -1)
            if axs.shape != (1, ncols):
                raise ValueError()

        for obj, ax in zip(list_of_objs, axs[0]):
            obj.show(scalebar=scalebar, title=title, figax=(fig, ax), **kwargs)

        return fig, axs
