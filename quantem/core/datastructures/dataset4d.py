from quantem.core.datastructures.dataset import Dataset as Dataset


# class for quantem 4d datasets
class Dataset4d(Dataset):
    def get_dp_mean(
        self,
        attach=True,
    ):
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
