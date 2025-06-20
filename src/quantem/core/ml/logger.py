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
    ):
        self._timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # This should never be reinstantiated.
        self._run_prefix = run_prefix
        self._run_suffix = run_suffix
        self.log_dir = base_log_dir
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
        if not isinstance(dir, Path) or not isinstance(dir, str):
            raise TypeError("Log directory is not a str or Path")
        elif isinstance(dir, str):
            dir = Path(dir / f"{self._run_prefix}_{self._timestamp}")
        else:
            dir = dir / f"{self._run_prefix}_{self._timestamp}"

        return dir

    # --- Helper Functions ---

    @staticmethod
    def apply_colormap(tensor_2d: Tensor | NDArray, cmap_name: str = "turbo"):
        """
        Apply colormap to a 2D tensor and return a [3, H, W] NumPy float32 array in [0, 1].
        """
        if isinstance(tensor_2d, Tensor):
            tensor_2d = tensor_2d.detach().cpu().numpy()

        tensor_2d = (tensor_2d - np.min(tensor_2d)) / (np.ptp(tensor_2d) + 1e-8)
        cmap = plt.get_cmap(cmap_name)
        colored = cmap(tensor_2d)[..., :3].transpose(2, 0, 1)  # [3, H, W]
        return colored.astype(np.float32)
