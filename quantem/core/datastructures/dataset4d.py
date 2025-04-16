import numpy as np

from quantem.core.datastructures.dataset import Dataset as Dataset


# class for quantem 4d datasets
class Dataset4d(Dataset):
    def get_dp_mean(
        self,
        attach: bool = True,
    ):
        """
        Get mean diffraction pattern

        Parameters
        ----------
        attach: bool
            If True attachs mean diffraction pattern to dataset

        Returns
        --------
        mean diffraction pattern (if attach is True)
        """
        dp_mean = self.mean((0, 1))

        dp_mean_dataset = Dataset(
            data=dp_mean,
            name="dp_mean",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self.dp_mean = dp_mean

        return dp_mean_dataset

    def get_dp_max(
        self,
        attach: bool = True,
    ):
        """
        Get max diffraction pattern

        Parameters
        ----------
        attach: bool
            If True attachs max diffraction pattern to dataset

        Returns
        --------
        max diffraction pattern (if attach is True)
        """
        dp_max = self.max((0, 1))

        dp_max_dataset = Dataset(
            data=dp_max,
            name="dp_max",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self.dp_max = dp_max

        return dp_max_dataset

    def get_dp_median(
        self,
        attach: bool = True,
    ):
        """
        Get median diffraction pattern

        Parameters
        ----------
        attach: bool
            If True attachs median diffraction pattern to dataset

        Returns
        --------
        median diffraction pattern (if attach is True)
        """
        dp_median = np.median(self.array, axis=(0, 1))

        dp_median_dataset = Dataset(
            data=dp_median,
            name="dp_median",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self.dp_median = dp_median

        return dp_median_dataset

    def get_virtual_image(
        self,
        mask: np.ndarray,
        attach: bool = True,
        name: str = None,
    ):
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

        if name is None:
            name = "virual_image"

        virtual_image = np.sum(self.array * mask, axis=(-1, -2))

        virtual_image_dataset = Dataset(
            data=dp_median,
            name=name,
            origin=self.origin[0:2],
            sampling=self.sampling[0:2],
            units=self.units[0:2],
            signal_units=self.signal_units,
        )

        if attach is True:
            setattr(self, name, virtual_image)

        return virtual_image_dataset
