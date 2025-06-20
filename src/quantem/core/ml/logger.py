import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch._tensor import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from quantem.core.io.serialize import AutoSerialize

"""
Tensorboard logger class for AD/ML reconstruction methods
"""


class LoggerBase(AutoSerialize):
    def __init__(
        self,
        base_log_dir: Path | str,
        run_prefix: str,
        run_suffix: str = None,
        log_images_every: int = 10,
    ):
        self._timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # This should never be reinstantiated.
        self.run_prefix = run_prefix
        self.run_suffix = run_suffix
        self.log_dir = base_log_dir
        self.log_images_every = log_images_every
        self.writer = SummaryWriter(str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag=tag, value=value, step=step)

    def log_image(self, tag: str, image: NDArray | Tensor, step: int, cmap: str = "turbo"):
        cmap_image = self.apply_colormap(image, cmap=cmap)
        self.writer.add_image(tag, cmap_image, step)

    def log_figure(self, tag: str, fig: Figure, step: int):
        self.writer.add_figure(tag, fig, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    # --- Properties ---

    @property
    def log_dir(self):
        return self._log_dir

    @log_dir.setter
    def log_dir(self, dir: str | Path):
        if not isinstance(dir, (str, Path)):
            raise TypeError("Log directory must be a str or Path.")

        dir = Path(dir)
        name = self.run_prefix + "_" + self._timestamp
        if self.run_suffix:
            name += f"_{self.run_suffix}"

        full_path = dir / name
        full_path.mkdir(parents=True, exist_ok=True)

        self._log_dir = full_path

    @property
    def run_prefix(self):
        return self._run_prefix

    @run_prefix.setter
    def run_prefix(self, prefix: str):
        if not isinstance(prefix, str):
            raise TypeError("Prefix must be a string")

        self._run_prefix = prefix

    @property
    def run_suffix(self):
        return self._run_suffix

    @run_suffix.setter
    def run_suffix(self, suffix: str):
        if not isinstance(suffix, str):
            raise TypeError("Suffix must be a string")

        self._run_suffix = suffix

    @property
    def log_images_every(self):
        return self._log_images_every

    # --- Helper Functions ---

    @staticmethod
    def apply_colormap(tensor_2d: Tensor | NDArray, cmap_name: str = "turbo") -> NDArray:
        """
        Apply colormap to a 2D tensor and return a [3, H, W] NumPy float32 array in [0, 1].
        """
        if isinstance(tensor_2d, Tensor):
            tensor_2d = tensor_2d.detach().cpu().numpy()

        tensor_2d = (tensor_2d - np.min(tensor_2d)) / (np.ptp(tensor_2d) + 1e-8)
        cmap = plt.get_cmap(cmap_name)
        colored = cmap(tensor_2d)[..., :3].transpose(2, 0, 1)  # [3, H, W]
        return colored.astype(np.float32)
