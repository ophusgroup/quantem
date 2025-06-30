from typing import Any

import numpy as np
import torch
from scipy.spatial import KDTree
from torch.optim import Optimizer
from typing_extensions import Literal

from .gs_config import Config
from .gs_strategy_base import StrategyBase


class DefaultStrategy(StrategyBase):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small sigmas.
    - Periodically split GSs with high image plane gradients and large sigmas.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset intensities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "positions2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: dict[str, Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    def __init__(
        self,
        cfg: Config,
        refine_scale2d_stop_iter: int = 0,
        pause_refine_after_reset: int = 0,
        revised_intensity: bool = False,
        verbose: bool = False,
        mode: Literal["2d", "3d"] = "2d",
        key_for_gradient: Literal["positions2d", "positions3d"] = "positions2d",
    ):
        super().__init__(cfg, verbose)
        self.refine_scale2d_stop_iter = refine_scale2d_stop_iter
        self.pause_refine_after_reset = pause_refine_after_reset
        self.revised_intensity = revised_intensity
        self.mode = mode
        self.key_for_gradient = key_for_gradient

    def initialize_state(self) -> dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - sigmas: the sigmas of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": 1}
        if self.refine_scale2d_stop_iter > 0:
            state["sigmas"] = None
        return state

    def check_sanity(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"positions", "sigmas", "intensities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["positions", "sigmas", "intensities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert self.key_for_gradient in info, (
            "The 2D positions of the Gaussians is required but missing."
        )
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.cfg.refine_stop_iter:
            return

        self._update_state(params, state, info)

        if (
            step >= self.cfg.refine_start_iter
            and step % self.cfg.refine_every == 0
            and step % self.cfg.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            # TODO
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step} dupli/split: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now have {len(params['positions'])} GSs."
                )

            # TODO split prune, or pass in options of what to prune
            # prune big, boundaries
            # merge XY
            n_merge = self._merge_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step} merge: {n_merge} events | "
                    f"Now have {len(params['positions'])} GSs."
                )
            # prune small, intensities

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step} prune: {n_prune} GSs | Now have {len(params['positions'])} GSs."
                )

            n_add = self._add_gaussians(params, optimizers, state, step)
            if self.verbose and n_add > 0:
                print(
                    f"Step {step} adding new grid: {n_add} GSs | "
                    f"Now have {len(params['positions'])} GSs."
                )
            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["sigmas"].zero_()
            torch.cuda.empty_cache()

        # TODO - implement reset intensities?
        if step % self.cfg.reset_every == 0:
            # self.reset_intensities(
            #     params=params,
            #     optimizers=optimizers,
            #     state=state,
            #     # value=self.prune_intensities * 2.0,
            #     value=self.cfg.init_intensity,
            # )
            self.reset_sigmas(
                params=params,
                optimizers=optimizers,
                state=state,
                # value=self.prune_intensities * 2.0,
                value=self.cfg.init_sigma,
            )

    def _update_state(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        state: dict[str, Any],
        info: dict[str, Any],
    ):
        # print("update state state is: ", state)
        for key in [
            "width",
            "height",
            "n_images",
            "mask",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.cfg.absgrad:
            raise NotImplementedError
            # grads:torch.Tensor = info[self.key_for_gradient].absgrad.clone()
        else:
            # print("key for grad: ", self.key_for_gradient, " is none: ", info[self.key_for_gradient].grad is None)
            grads: torch.Tensor = info[self.key_for_gradient].grad.clone()

        grads[..., 0] *= info["width"] / 2.0 * info["n_images"]  # TODO figure out why this is?
        grads[..., 1] *= info["height"] / 2.0 * info["n_images"]  # Might be bad

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device, dtype=torch.float64)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device, dtype=torch.float64)
        if self.refine_scale2d_stop_iter > 0 and state["sigmas"] is None:
            assert "sigmas" in info, "sigmas is required but missing."
            state["sigmas"] = torch.zeros(n_gaussian, device=grads.device, dtype=torch.float64)

        ### masking gradients
        # # grads is [C, N, 2]
        gs_ids = torch.where(info["mask"])[0]  # [1]  # [nnz]
        grads = grads[gs_ids]  # [nnz, 2]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float64))

    # @torch.no_grad()
    def _grow_gs(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Any],
        step: int,
    ) -> tuple[int, int]:
        with torch.no_grad():
            count = state["count"]
            grads = state["grad2d"] / count.clamp_min(1)
            device = grads.device

            # print('grads : ', grads.max(), grads.mean(), grads.min(), grads.std(), grads)
            is_grad_high = grads > self.cfg.split_dup_grad2d
            is_small = (
                self.cfg.activation_sigma(params["sigmas"]).max(dim=-1).values
                <= self.cfg.grow_sigma3d * state["scene_scale"]
            )
            is_dupli = is_grad_high & is_small
            n_dupli = is_dupli.sum().item()

            is_large = ~is_small
            is_split = is_grad_high & is_large
            # if step < self.refine_scale2d_stop_iter:
            #     is_split |= state["sigmas"] > self.cfg.grow_sigma2d
            n_split = is_split.sum().item()

            # first duplicate
            if n_dupli > 0:
                print(f"skipping duplicate of {n_dupli} gs")
                # duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

            # new GSs added by duplication will not be split
            is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])

            # then split
            if n_split > 0:
                print(f"skipping splitting of {n_split} gs")
                # split(
                #     params=params,
                #     optimizers=optimizers,
                #     state=state,
                #     mask=is_split,
                #     revised_intensity=self.revised_intensity,
                # )
            return n_dupli, n_split

    # @torch.no_grad()
    def _prune_gs(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Any],
        step: int,
    ) -> int:
        ## TODO re-order
        # prune big
        # merge xy
        # prune small, intensity, xy

        # print("pre pruning state: ", state["count"].shape, state)
        with torch.no_grad():
            # TODO make strategy a inheritor of base GS so these functions are from cfg
            scaled_intensities = (
                self.cfg.activation_intensity(params["intensities"])
                * ((2 * torch.pi) ** 0.5)
                * self.cfg.activation_sigma(params["sigmas"])
            )
            prune_intensity = scaled_intensities.flatten() < (
                self.cfg.prune_intensity_fac * self.cfg.init_intensity
            )
            prune_big = (
                self.cfg.activation_sigma(params["sigmas"].flatten()) > self.cfg.prune_sigma_big_A
            )
            prune_small = (
                self.cfg.activation_sigma(params["sigmas"].flatten())
                < self.cfg.prune_sigma_small_A
            )
            prune_x = (
                params["positions"][:, 2] > self.cfg.volume_size[2] + self.cfg.prune_pad_A
            ) | (params["positions"][:, 2] < -self.cfg.prune_pad_A)
            prune_y = (
                params["positions"][:, 1] > self.cfg.volume_size[1] + self.cfg.prune_pad_A
            ) | (params["positions"][:, 1] < -self.cfg.prune_pad_A)
            if self.cfg.model_type == "3dgs":
                raise NotImplementedError
                # prune_z =
            # else:

            is_prune = prune_intensity | prune_big | prune_small | prune_x | prune_y
            n_prune = int(is_prune.sum().item())
            print(
                f"Step {step} pruned: total: {n_prune} | intensity: {prune_intensity.sum()} | big: {prune_big.sum()} | small: {prune_small.sum()} | xy: {(prune_x | prune_y).sum()}"
            )

            if n_prune > 0:
                self.remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

            # print("post pruning state: ", state["count"].shape, state)
            return n_prune

    def _merge_gs(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Any],
        step: int,
    ):
        # identify which indices need to be merged based on cutoff distance
        # not sure how to deal with strings of atoms.. could do a clustering?
        # start with most intense, gather all close, go to next (excepting ones already gathered)

        # for each group of indices, sum intensity, position becomes mean weighted by intensity
        # sigma is weighted sum. i.e. sigma of intensity max + sigma1 * intensity1/intensity_max + sigma2*...

        with torch.no_grad():
            pos = params["positions"].data.cpu().detach().numpy()
            sigmas = self.cfg.activation_sigma(params["sigmas"].flatten()).cpu().detach().numpy()
            sigmas_ranked = np.argsort(sigmas)[::-1]

            tree1 = KDTree(pos)
            tree2 = KDTree(pos)
            neighbor_indices = tree1.query_ball_tree(tree2, r=self.cfg.xy_merge_A)

            seen = []  # indices that have been accounted for, cannot be merged twice
            keeps = []  # indices updated with merged splats and kept, shape (N_merge_events)
            new_merges = []  # indices merged and deleted, ragged shape (N_merge_events, n_merge-1)
            # iterate through each splat in order of sigma, and gather those that should be merged
            for ind in sigmas_ranked:
                if ind in seen:
                    continue
                else:
                    seen.append(ind)
                closes = np.array(neighbor_indices[ind])
                check_seen = ~np.isin(closes, np.array(seen))
                sigma_keep = sigmas[ind]
                sigmas_merg = sigmas[neighbor_indices[ind]]
                check_sigmas = (sigma_keep * self.cfg.xy_merge_sigma_fac) >= sigmas_merg
                news = closes[check_seen & check_sigmas]
                if any(news):
                    seen.extend(news)
                    keeps.append(ind)
                    new_merges.append(list(news))

            if any(new_merges):
                assert len(keeps) == len(new_merges)
                assert len(seen) == len(params["positions"])
                assert len(np.unique(np.concatenate(new_merges))) == len(
                    np.concatenate(new_merges)
                )
                assert np.all(~np.isin(np.concatenate(new_merges), np.array(keeps)))
                n_merge = len(keeps)

                self._merge(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    keeps=keeps,
                    merges=new_merges,
                )
            else:
                n_merge = 0

        return n_merge

    def _add_gaussians(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Any],
        step: int,
    ):
        # mode grid, random, density (place in least dense areas)
        # grid check if 2d or not
        with torch.no_grad():
            if not (self.cfg.add_start_iter <= step <= self.cfg.add_stop_iter):
                return 0

            device = params["positions"].device
            if self.cfg.model_type == "3dgs":
                raise NotImplementedError("3DGS not implemented yet for adding new Gaussians")
            if self.cfg.add_gaussians_mode == "grid":
                y, x = np.mgrid[
                    0 : self.cfg.volume_size[1] : self.cfg.add_gaussian_sampling,
                    0 : self.cfg.volume_size[2] : self.cfg.add_gaussian_sampling,
                ]
                y += y[1, 1] / 2
                x += x[1, 1] / 2
                positions = torch.tensor(
                    np.stack([np.zeros_like(y.ravel()), y.ravel(), x.ravel()]).T,
                    dtype=torch.float64,
                    device=device,
                )
                positions[:, 0] = 0

                N = positions.shape[0]
                # # Initialize the GS size to be half the average dist of the 3 nearest neighbors
                sigmas = torch.ones(
                    N, dtype=torch.float64, device=device
                ) * self.cfg.activation_sigma_inverse(self.cfg.init_sigma)
                intensities = torch.ones(
                    N, dtype=torch.float64, device=device
                ) * self.cfg.activation_intensity_inverse(self.cfg.init_intensity_scaled)
                new_params = {
                    "positions": positions,
                    "sigmas": sigmas,
                    "intensities": intensities,
                }
            elif self.cfg.add_gaussians_mode == "random":
                raise NotImplementedError
            elif self.cfg.add_gaussians_mode == "density":
                raise NotImplementedError
            else:
                raise ValueError(f"Bad mode: {self.cfg.add_gaussians_mode}")

            self._add(params, new_params, optimizers, state)

            return N
