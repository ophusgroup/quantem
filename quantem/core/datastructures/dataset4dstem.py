import matplotlib.pyplot as plt
import numpy as np
from attrs import define

from quantem.core.datastructures.dataset import Dataset as Dataset
from quantem.core.visualization.visualization_utils import ScalebarConfig


@define
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

    virtual_images: dict[str, Dataset] = {}

    def __attrs_post_init__(self):
        if self.array.ndim != 4:
            raise ValueError(
                f"Dataset4dstem must have 4 dimensions, got {self.array.ndim}"
            )

    def __getitem__(self, index):
        """Simple indexing function to return Dataset view"""

        array_view = self.array[index]
        ndim = array_view.ndim
        calibrated_origin = self.origin.ndim == self.ndim

        return Dataset(
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

        dp_mean_dataset = Dataset(
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

        dp_max_dataset = Dataset(
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

        dp_median_dataset = Dataset(
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

        virtual_image_dataset = Dataset(
            array=virtual_image,
            name=name,
            origin=self.origin[0:2],
            sampling=self.sampling[0:2],
            units=self.units[0:2],
            signal_units=self.signal_units,
        )

        if attach is True:
            self.virtual_images[name] = virtual_image_dataset

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
