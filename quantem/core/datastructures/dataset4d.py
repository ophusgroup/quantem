import numpy as np

from quantem.core import config
from quantem.core.datastructures.dataset import Dataset as Dataset

if config.get("has_cupy"):
    import cupy as cp
else:
    import numpy as cp


# class for quantem 4d datasets
class Dataset4d(Dataset):
    def __init__(
        self,
        data: np.ndarray | cp.ndarray,
        name: str | None = None,
        origin: list | None = None,
        sampling: list | None = None,
        units: list[str] | None = None,
        signal_units: str | None = None,
    ):
        super().__init__(
            data=data,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
        )

        ## temp note: self.virtual_images is just returning a dictionary, which is probably fine.
        ## it means it can't be type protected, but it is implicitly by only setting values
        ## with self.get_virtual_image()
        self._virtual_images: dict[str, Dataset] = {}

    @property
    def dp_mean(self) -> Dataset:
        """
        Dataset containing the mean diffraction pattern
        """
        if hasattr(self, "_dp_mean"):
            return self._dp_mean
        else:
            print("Calculating dp_mean, attach with Dataset4d.get_dp_mean()")
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
            data=dp_mean,
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
            print("Calculating dp_max, attach with Dataset4d.get_dp_max()")
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
            data=dp_max,
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
            print("Calculating dp_median, attach with Dataset4d.get_dp_median()")
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
            data=dp_median,
            name=self.name + "_dp_median",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self._dp_median = dp_median_dataset

        return dp_median_dataset

    @property
    def virtual_images(self) -> dict[str, Dataset]:
        """
        This is just a dictionary for now
        """
        return self._virtual_images

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
            data=virtual_image,
            name=name,
            origin=self.origin[0:2],
            sampling=self.sampling[0:2],
            units=self.units[0:2],
            signal_units=self.signal_units,
        )

        if attach is True:
            self.virtual_images[name] = virtual_image_dataset

        return virtual_image_dataset
