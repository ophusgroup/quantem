from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from quantem.core.visualization import show_2d

from .gs_base import GSBase, SimpleDataset
from .gs_config import Config
from .gs_rendering import rasterization_2dgs  # , rasterization_2dgs_inria_wrapper


class GS2D(GSBase):
    def __init__(self, cfg: Config, trainset: SimpleDataset) -> None:
        GSBase.__init__(self, cfg=cfg, trainset=trainset)

        return

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

            # TODO rewrite this, the cfg info is already in the strategy
            # this should be "item to take gradient of" and just be positions
            info = {
                "height": self.cfg.raster_shape,  # [0],
                "width": self.cfg.raster_shape,  # [1],
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

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            losses.append(loss.item())

            # print and/or checkpoint
            if cfg.print_every > 0:
                if (step + 1) % cfg.print_every == 0 or step == 0 or step == max_steps - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    d = timedelta(seconds=(datetime.now() - start_time).seconds)
                    print(
                        f"Step: {step} | num_GS: {len(self.splats['positions'])} | mem {mem:.2f} GB | ellapsed time (h:m:s) {d}"
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
                show_2d(r, title=f"iter {step} render", force_show=True)

        return torch.squeeze(renders).cpu().detach().numpy(), np.array(losses)

    def rasterize_splats(
        self,
        rescale_sigmas: float | None = None,
        **kwargs,
    ) -> Tensor:
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
                **kwargs,
            )
        else:
            raise NotImplementedError

        return rendered_ims
