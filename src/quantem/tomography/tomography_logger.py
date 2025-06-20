import matplotlib.pyplot as plt

from quantem.core.ml.logger import LoggerBase
from quantem.tomography.tomography_dataset import TomographyDataset


class TomoLogger(LoggerBase):
    def __init__(self):
        super.__init__(self)

    # --- Tomography focused logging methods ---

    @staticmethod
    def tilt_angle_figure(dataset: TomographyDataset):
        figs = []
        for angle_array, title in zip(
            [dataset.z1_angles, dataset.tilt_angles, dataset.z3_angles],
            ["Z1 Angles", "Tilt/ X Angles", "Z3 Angles"],
        ):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(angle_array.detach().cpu().numpy())
            ax.set_title(title)
            ax.set_xlabel("Index")
            ax.set_ylabel("Angle")
            figs.append(fig)

        return figs
