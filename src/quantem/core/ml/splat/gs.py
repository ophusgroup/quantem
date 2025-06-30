import math
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure  # PeakSignalNoiseRatio
from tqdm.auto import tqdm

from quantem.core.utils.validators import validate_tensor
from quantem.core.visualization import show_2d

from .gs_config import Config
from .gs_rendering import rasterization_2dgs  # , rasterization_2dgs_inria_wrapper

# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from .gs_strategy import DefaultStrategy


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


class GS:
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

    def rasterize_splats(
        self,
        rescale_sigmas: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # rasterization does normalization internally
        sigmas = self.cfg.activation_sigma(self.splats["sigmas"])  # [N, 3]
        if rescale_sigmas is not None:  # for visualization
            sigmas *= rescale_sigmas
        intensities = self.cfg.activation_intensity(self.splats["intensities"])  # [N,]
        positions = self.splats["positions"]  # don't normalize here, take care of in strategy

        if self.model_type == "2dgs":
            rendered_ims = rasterization_2dgs(
                positions=positions,
                sigmas=sigmas,
                intensities=intensities,
                grids=(self.grid_y, self.grid_x),
                isotropic_splats=self.cfg.isotropic_splats,
                **kwargs,
            )
        else:
            raise NotImplementedError

        return rendered_ims

    def train(self, max_steps: int | None = None) -> tuple[np.ndarray | torch.Tensor, np.ndarray]:
        cfg = self.cfg
        if max_steps is None:
            max_steps = cfg.max_steps

        init_step = 0

        # TODO reset optimizer LRs to initial LR
        # positions has a learning rate schedule, end at 0.1 of the initial value
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["positions"], gamma=0.1 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["sigmas"], gamma=1 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["intensities"], gamma=1 ** (1.0 / max_steps)
            ),
        ]
        # append other schedulers if wanted

        trainloader = DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        renders = torch.ones(1)  # for typing
        losses = []
        # Training loop.
        start_time = datetime.now()
        pbar = tqdm(range(init_step, max_steps))
        for step in pbar:
            try:
                image = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                image = next(trainloader_iter)
            image = image.to(self.device)

            # forward
            renders = self.rasterize_splats(
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )
            if (
                step == 1
                or (
                    ((step + 1) % cfg.refine_every) == 0
                    and cfg.refine_stop_iter > (step + 1) >= cfg.refine_start_iter
                )
                or (
                    (step % cfg.refine_every) == 0
                    and cfg.refine_stop_iter > step >= cfg.refine_start_iter
                )
                or ((step % cfg.reset_every) == 0 and ((step - 1) % cfg.reset_every) == 0)
            ):
                r = torch.squeeze(renders).cpu().detach().numpy()
                show_2d(r, title=f"iter {step} render", force_show=True, cbar=True, norm="minmax")
                # print(f"Intensities iter {step}: \n", self.cfg.activation_intensity(self.splats["intensities"]))
                # print(f"Scaled intensities iter {step}:\n", self.cfg.activation_intensity(self.splats["intensities"]) * np.sqrt(2*np.pi) * self.cfg.activation_sigma(self.splats["sigmas"]))
                # print(f"Sigmas iter {step}: \n", self.cfg.activation_sigma(self.splats["sigmas"]))

            # TODO rewrite this, the cfg info is already in the strategy
            # this should be "item to take gradient of" and just be positions
            info = {
                "height": self.cfg.raster_shape[0],
                "width": self.cfg.raster_shape[1],
                "positions2d": self.splats["positions"],
                "n_images": 1,
                "mask": torch.ones_like(self.splats["sigmas"], dtype=torch.float64),
            }
            # info["positions2d"] = self.splats["positions"] # works but moving the
            # info creation above doesnt

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step + 1,
                info=info,
            )

            ### loss
            l2loss = F.mse_loss(renders[None,], image[None,])
            # l2loss = F.mse_loss(renders[None,], image[None,])
            if cfg.ssim_lambda > 0:
                ssimloss = 1.0 - self.ssim(renders[None,], image[None,])
                loss = l2loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            else:
                loss = l2loss

            # loss += other loss terms

            loss.backward()
            desc = f"loss={loss.item():.3e} "
            pbar.set_description(desc)

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step + 1,
                info=info,
            )

            # print and/or checkpoint
            if cfg.print_every > 0:
                if (step + 1) % cfg.print_every == 0 or step == 0 or step == max_steps - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    d = timedelta(seconds=(datetime.now() - start_time).seconds)
                    print(
                        f"Step: {step} | num_GS: {len(self.splats['positions'])} | mem {mem:.2f} GB | ellapsed time (h:m:s) {d}"
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            losses.append(loss.item())

        return torch.squeeze(renders).cpu().detach().numpy(), np.array(losses)
