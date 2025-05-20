from typing import TYPE_CHECKING, Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from quantem.core import config
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.ptychography_base import PtychographyBase

if TYPE_CHECKING:
    pass


# TODO make dataclass
class PtychographyVisualizations(PtychographyBase):
    def show_obj(
        self,
        obj: np.ndarray | None = None,
        cbar: bool = False,
        interval_type: Literal["quantile", "manual"] = "quantile",
    ):
        if obj is None:
            obj = self.obj_cropped
        else:
            obj = self._to_numpy(obj)
            if obj.ndim == 2:
                obj = obj[None, ...]

        if interval_type == "quantile":
            norm = {"interval_type": "quantile"}
        elif interval_type in ["manual", "minmax", "abs"]:
            norm = {"interval_type": "manual"}
        else:
            raise ValueError(f"Unknown interval type: {interval_type}")

        ph_cmap = config.get("visualize.phase_cmap")
        if obj.shape[0] > 1:
            t = "Summed "
        else:
            t = ""

        ims = []
        titles = []
        cmaps = []
        if self.obj_type == "potential":
            ims.append(np.abs(obj).sum(0))
            titles.append(t + "Potential")
            cmaps.append(ph_cmap)
        elif self.obj_type == "pure_phase":
            ims.append(np.angle(obj).sum(0))
            titles.append(t + "Pure Phase")
            cmaps.append(ph_cmap)
        else:
            ims.extend([np.angle(obj).sum(0), np.abs(obj).sum(0)])
            titles.extend([t + "Phase", t + "Amplitude"])
            cmaps.extend([ph_cmap, "gray"])

        scalebar = [{"sampling": self.sampling[0], "units": "Å"}] + [None] * (len(ims) - 1)

        show_2d(
            ims,
            title=titles,
            cmap=cmaps,
            norm=norm,
            cbar=cbar,
            scalebar=scalebar,
        )

    def show_probe(self, probe: np.ndarray | None = None):
        if probe is None:
            probe = self.probe
        else:
            probe = self._to_numpy(probe)
            if probe.ndim == 2:
                probe = probe[None, ...]

        probes = [np.fft.fftshift(probe[i]) for i in range(len(probe))]
        scalebar = {"sampling": self.reciprocal_sampling[0], "units": r"$\mathrm{A^{-1}}$"}
        if len(probes) > 1:
            titles = self.get_probe_intensities(probe)
            titles = [f"Probe {i + 1}/{len(titles)}: {t * 100:.1f}%" for i, t in enumerate(titles)]
        else:
            titles = "Probe"
        show_2d(probes, title=titles, scalebar=scalebar)

    def show_fourier_probe(self, probe: np.ndarray | None = None):
        if probe is None:
            probe = self.probe
        else:
            probe = self._to_numpy(probe)
            if probe.ndim == 2:
                probe = probe[None, ...]

        probes = [np.fft.fftshift(np.fft.fft2(probe[i])) for i in range(len(probe))]
        if len(probes) > 1:
            titles = self.get_probe_intensities(probe)
            titles = [
                f"Fourier Probe {i + 1}/{len(titles)}: {t * 100:.1f}%"
                for i, t in enumerate(titles)
            ]
        else:
            titles = "Fourier Probe"
        show_2d(probes, title=titles)

    def show_obj_and_probe(self, cbar: bool = False, figax=None):
        """shows the summed object and summed probe"""
        ims = []
        titles = []
        cmaps = []
        if self.obj_type == "potential":
            ims.append(np.abs(self.obj_cropped).sum(0))
            titles.append("Potential")
            cmaps.append(config.get("visualize.phase_cmap"))
        elif self.obj_type == "pure_phase":
            ims.append(np.angle(self.obj_cropped).sum(0))
            titles.append("Pure Phase")
            cmaps.append(config.get("visualize.phase_cmap"))
        else:
            ims.append(np.angle(self.obj_cropped).sum(0))
            ims.append(np.abs(self.obj_cropped).sum(0))
            titles.extend(["Phase", "Amplitude"])
            cmaps.extend([config.get("visualize.phase_cmap"), config.get("visualize.cmap")])

        ims.append(np.fft.fftshift(self.probe.sum(0)))
        titles.append("Probe")
        cmaps.append(None)
        scalebar = [{"sampling": self.sampling[0], "units": "Å"}] + [None] * (len(ims) - 1)
        cbars = [True] * (len(ims) - 1) + [False] if cbar else False
        show_2d(
            ims,
            title=titles,
            cmap=cmaps,
            scalebar=scalebar,
            cbar=cbars,
            figax=figax,
            tight_layout=True if figax is None else False,
        )

    def show_obj_slices(
        self,
        obj: np.ndarray | None = None,
        cbar: bool = False,
        interval_type: Literal["quantile", "manual"] = "quantile",
    ):
        if obj is None:
            obj = self.obj_cropped
        else:
            obj = self._to_numpy(obj)
            if obj.ndim == 2:
                obj = obj[None, ...]

        if interval_type == "quantile":
            norm = {"interval_type": "quantile"}
        elif interval_type in ["manual", "minmax", "abs"]:
            norm = {"interval_type": "manual"}
        else:
            raise ValueError(f"Unknown interval type: {interval_type}")

        if self.obj_type == "potential":
            objs = [np.abs(obj[i]) for i in range(len(obj))]
            titles = [f"Potential {i + 1}/{len(objs)}" for i in range(len(objs))]
        elif self.obj_type == "pure_phase":
            objs = [np.angle(obj[i]) for i in range(len(obj))]
            titles = [f"Pure Phase {i + 1}/{len(objs)}" for i in range(len(objs))]
        else:
            objs = [np.angle(obj[i]) for i in range(len(obj))]
            titles = [f"Phase {i + 1}/{len(objs)}" for i in range(len(objs))]

        scalebar = [{"sampling": self.sampling[0], "units": "Å"}] + [None] * (len(objs) - 1)
        show_2d(
            objs,
            title=titles,
            cmap=config.get("visualize.phase_cmap"),
            norm=norm,
            cbar=cbar,
            scalebar=scalebar,
        )

    def plot_losses(self, figax: tuple | None = None, plot_lrs: bool = True):
        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        lines = []
        epochs = np.arange(len(self.epoch_losses))
        colors = ["b", "orange", "r", "g"]
        lines.extend(ax.semilogy(epochs, self.epoch_losses, c=colors[0], label="loss"))
        ax.set_ylabel("Loss", color=colors[0])
        ax.tick_params(axis="y", which="both", colors=colors[0])
        ax.spines["left"].set_color(colors[0])
        # for label in ax.get_yticklabels():
        #     label.set_color(colors[0])
        ax.set_xlabel("Epochs")
        nx = ax.twinx()
        nx.spines["left"].set_visible(False)

        if plot_lrs:
            GD_inds = np.where(self.epoch_recon_types == "GD")[0]
            pixelwise_inds = np.where(self.epoch_recon_types == "pixelwise")[0]
            model_inds = np.where(self.epoch_recon_types == "model")[0]

            if len(GD_inds) > 0:
                lines.extend(
                    nx.plot(
                        GD_inds, self.epoch_lrs["GD"][GD_inds], c=colors[2], label="GD step size"
                    )
                )
                nx.set_ylabel("GD step size", c=colors[2])
                nx.spines["right"].set_color(colors[2])
                nx.tick_params(axis="y", which="both", colors=colors[2])
                # lines.append(
                #     ax.axvspan(
                #         min(epochs[GD_inds]) - 0.1,
                #         max(epochs[GD_inds]) + 0.1,
                #         color=colors[1],
                #         alpha=0.3,
                #         label="GD",
                #     )
                # )

            if len(pixelwise_inds) > 0:
                lines.extend(
                    nx.semilogy(
                        epochs[pixelwise_inds],
                        self.epoch_lrs["object"][pixelwise_inds],
                        c=colors[2],
                        label="object pixelwise LR",
                    )
                )
                nx.set_ylabel("LR", c=colors[2])
                nx.spines["right"].set_color(colors[2])
                nx.tick_params(axis="y", which="both", colors=colors[2])

            if len(model_inds) > 0:
                lines.extend(
                    nx.semilogy(
                        epochs[model_inds],
                        self.epoch_lrs["object"][model_inds],
                        c=colors[3],
                        label="object model LR",
                    )
                )
                nx.set_ylabel("LR", c=colors[3])
                nx.spines["right"].set_color(colors[3])
                nx.tick_params(axis="y", which="both", colors=colors[3])

        labs = [lin.get_label() for lin in lines]
        nx.legend(lines, labs)
        ax.set_xbound(-2, np.max(epochs if np.any(epochs) else [1]) + 2)
        if figax is None:
            plt.tight_layout()
            plt.show()

    def visualize(self, cbar: bool = True):
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax_top = fig.add_subplot(gs[0])
        self.plot_losses(figax=(fig, ax_top))

        n_bot = 3 if self.obj_type == "complex" else 2
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, n_bot, subplot_spec=gs[1])
        axs_bot = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(n_bot)])
        self.show_obj_and_probe(figax=(fig, axs_bot), cbar=cbar)
        plt.show()

    def show_object_epochs(self):
        # image grid
        pass
