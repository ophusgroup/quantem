import matplotlib.pyplot as plt

from quantem.core.ml.logger import LoggerBase
from quantem.tomography.object_models import ObjectModelType
from quantem.tomography.tomography_dataset import TomographyDataset


class TomoLogger(LoggerBase):
    def __init__(self):
        super.__init__(self)

    # --- Tomography focused logging methods ---
    @staticmethod
    def tilt_angles_figure(dataset: TomographyDataset):
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

    def projection_images(
        self, volume_obj: ObjectModelType, epoch: int, logger_cmap: str = "turbo"
    ):
        sum_0 = volume_obj.obj.sum(axis=0)
        sum_1 = volume_obj.obj.sum(axis=1)
        sum_2 = volume_obj.obj.sum(axis=2)

        self.log_image(
            tag="projections/Y-X Projection",
            image=sum_0,
            step=epoch,
            cmap=logger_cmap,
        )
        self.log_image(
            tag="projections/Z-X Projection",
            image=sum_1,
            step=epoch,
            cmap=logger_cmap,
        )
        self.log_image(
            tag="projections/Z-Y Projection",
            image=sum_2,
            step=epoch,
            cmap=logger_cmap,
        )
