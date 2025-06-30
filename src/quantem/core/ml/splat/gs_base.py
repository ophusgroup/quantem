import math

import numpy as np
import torch
from torch.optim import Adam, Optimizer
from torchmetrics.image import StructuralSimilarityIndexMeasure  # PeakSignalNoiseRatio

from quantem.core.utils.validators import validate_tensor

from .gs_config import Config

# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from .gs_strategy import DefaultStrategy

# todo
# random and torch rng


class SimpleDataset(torch.utils.data.Dataset):
    """A simple dataset class for fitting a single image or series of images."""

    def __init__(
        self,
        image: np.ndarray | torch.Tensor,
    ):
        self.images = validate_tensor(
            image,
            name="image",
            dtype=torch.float64,
            ndim=3,
            expand_dims=True,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index]


class GSBase:
    """Base Gaussian splatting class inherited by 2D and 3D"""

    def __init__(
        self, cfg: Config, trainset: SimpleDataset, rng: int | np.random.Generator | None = None
    ) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.rng = rng

        self.trainset = trainset

        # Model
        self.splats, self.optimizers = self._initialize_splats_and_optimizers()
        print("Model initialized. Number of GS:", len(self.splats["positions"]))
        self.model_type = cfg.model_type

        if self.model_type == "2dgs":
            key_for_gradient = "positions2d"
        else:
            raise NotImplementedError(f"bad model_type {self.model_type}")

        # Densification Strategy
        self.strategy = DefaultStrategy(
            cfg=cfg,
            verbose=True,
            mode="2d",
            key_for_gradient=key_for_gradient,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)  # TODO note data range
        # self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        # Precompute the 2D grid for the target image
        yy, xx = torch.meshgrid(
            torch.arange(cfg.raster_shape[0], device=self.device) * cfg.raster_sampling[0],
            torch.arange(cfg.raster_shape[1], device=self.device) * cfg.raster_sampling[1],
            indexing="ij",
        )
        self.grid_y = yy.type(torch.float64)
        self.grid_x = xx.type(torch.float64)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator | int | None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, float)):
            rng = np.random.default_rng(rng)
        elif not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng should be a np.random.Generator or a seed, got {type(rng)}")
        self._rng = rng
        seed = rng.bit_generator._seed_seq.entropy  # type:ignore ## get seed from the generator
        self._rng_torch = torch.Generator(device=self.device).manual_seed(seed % 2**32)

    def train(
        self,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor | np.ndarray, np.ndarray]:
        """Train to recreate an image/volume"""
        return torch.tensor(1), np.array(1)

    def rasterize_splats(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Produce output image(s)"""
        return torch.tensor(1)

    def _initialize_splats_and_optimizers(
        self,
    ) -> tuple[torch.nn.ParameterDict, dict[str, Optimizer]]:
        if self.cfg.init_type == "random":
            positions = torch.tensor(self.cfg.volume_size) * (
                torch.rand((self.cfg.init_num_pts, 3))
            )  # TODO use self.rng
        elif self.cfg.init_type == "grid":
            yy, xx = np.mgrid[
                0 : self.cfg.volume_size[-2] : self.cfg.init_grid_sampling,
                0 : self.cfg.volume_size[-1] : self.cfg.init_grid_sampling,
            ]
            yy += yy[1, 1] / 2
            xx += xx[1, 1] / 2
            positions = torch.tensor(
                np.stack([np.zeros_like(yy.ravel()), yy.ravel(), xx.ravel()]).T,
                dtype=torch.float64,
            )
        else:
            raise ValueError(f"Unknown init_type: {self.cfg.init_type}")

        N = positions.shape[0]
        sigmas = torch.ones((N, 3), dtype=torch.float64) * self.cfg.activation_sigma_inverse(
            self.cfg.init_sigma
        )
        # # --- SIGMA INIT ---
        # val = self.cfg.activation_sigma_inverse(self.cfg.init_sigma)
        # sigmas = torch.full((N, 3), fill_value=val, dtype=torch.float64)
        # # For 2D splatting, set sigma_z = 0 (no depth spread), only y and x are active
        # sigmas[:, 0] = 0.0  # sigma_z (depth)
        # sigmas[:, 1] = val  # sigma_y (rows)
        # sigmas[:, 2] = val  # sigma_x (columns)
        # if self.cfg.isotropic_splats:
        #     # Enforce sigma_y = sigma_x and sigma_z = 0 for all splats (already set above)
        #     pass
        # # else: allow optimization to change them independently

        intensities = torch.ones(N, dtype=torch.float64) * self.cfg.activation_intensity_inverse(
            self.cfg.init_intensity_scaled
        )
        params = [
            ("positions", torch.nn.Parameter(positions), self.cfg.lr_base),
            ("sigmas", torch.nn.Parameter(sigmas), self.cfg.lr_base / 10),
            ("intensities", torch.nn.Parameter(intensities), self.cfg.lr_base / 10),
        ]

        splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(self.cfg.device)
        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # Note that this would not make the training exactly equivalent, see
        # https://arxiv.org/pdf/2402.18824v1
        optimizers = {
            name: Adam(
                [{"params": splats[name], "lr": lr * math.sqrt(self.cfg.batch_size)}],
                eps=1e-15 / math.sqrt(self.cfg.batch_size),
                betas=(1 - self.cfg.batch_size * (1 - 0.9), 1 - self.cfg.batch_size * (1 - 0.999)),
            )
            for name, _, lr in params
        }
        return splats, optimizers  # type:ignore ## an instance of Adam is an Optimizer... idk.

    # def _add_new_gaussians_grid(
    #     self, grid_spacing: float
    # ) -> tuple[torch.nn.ParameterDict, dict[str, Optimizer]]:
    #     y, x = np.mgrid[0 : self.cfg.extent : grid_spacing, 0 : self.cfg.extent : grid_spacing]
    #     y += y[1, 1] / 2
    #     x += x[1, 1] / 2
    #     positions = torch.tensor(
    #         self.cfg.global_scale * np.stack([np.zeros_like(y.ravel()), y.ravel(), x.ravel()]).T,
    #         dtype=torch.float64,
    #     )
    #     positions[:, 0] = 0

    #     N = positions.shape[0]
    #     # # Initialize the GS size to be half the average dist of the 3 nearest neighbors
    #     sigmas = torch.ones(N, dtype=torch.float64) * self.cfg.activation_sigma_inverse(
    #         self.cfg.init_sigma
    #     )
    #     intensities = torch.ones(N, dtype=torch.float64) * self.cfg.activation_intensity_inverse(
    #         self.cfg.init_intensity_scaled
    #     )
    #     params = [
    #         # name, value, lr
    #         (
    #             "positions",
    #             torch.nn.Parameter(torch.cat([self.splats.positions.cpu().detach(), positions])),
    #             self.cfg.lr_base * self.cfg.global_scale,
    #         ),  # type:ignore #FIXME
    #         (
    #             "sigmas",
    #             torch.nn.Parameter(torch.cat([self.splats.sigmas.cpu().detach(), sigmas])),
    #             self.cfg.lr_base / 10,
    #         ),  # type:ignore #FIXME
    #         (
    #             "intensities",
    #             torch.nn.Parameter(
    #                 torch.cat([self.splats.intensities.cpu().detach(), intensities])
    #             ),
    #             self.cfg.lr_base / 10,
    #         ),  # type:ignore #FIXME
    #     ]

    #     splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(self.cfg.device)
    #     # Scale learning rate based on batch size, reference:
    #     # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    #     # Note that this would not make the training exactly equivalent, see
    #     # https://arxiv.org/pdf/2402.18824v1
    #     optimizers = {
    #         name: Adam(
    #             [{"params": splats[name], "lr": lr * math.sqrt(self.cfg.batch_size)}],
    #             eps=1e-15 / math.sqrt(self.cfg.batch_size),
    #             betas=(1 - self.cfg.batch_size * (1 - 0.9), 1 - self.cfg.batch_size * (1 - 0.999)),
    #         )
    #         for name, _, lr in params
    #     }
    #     return splats, optimizers  # type:ignore
