from abc import abstractmethod
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

from .gs_config import Config


class StrategyBase:
    """Base class for the GS densification strategy.

    This class is an base class that defines the interface for the GS
    densification strategy.
    """

    def __init__(self, cfg: Config, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self._refined_iters = []
        self._add_iters = []

    @property
    def refined_iters(self) -> np.ndarray:
        """
        Iterations where "refine" was run
        """
        return np.array(self._refined_iters)

    @property
    def add_iters(self) -> np.ndarray:
        """
        Iterations when "add_gaussians" was run
        """
        return np.array(self._add_iters)

    def check_sanity(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
    ):
        """Sanity check for the parameters and optimizers."""
        trainable_params = set([name for name, param in params.items() if param.requires_grad])
        assert trainable_params == set(optimizers.keys()), (
            "trainable parameters and optimizers must have the same keys, "
            f"but got {trainable_params} and {optimizers.keys()}"
        )

        for optimizer in optimizers.values():
            assert len(optimizer.param_groups) == 1, (
                "Each optimizer must have exactly one param_group, "
                "that cooresponds to each parameter, "
                f"but got {len(optimizer.param_groups)}"
            )

    @abstractmethod
    def step_pre_backward(self, *args, **kwargs):
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    @abstractmethod
    def step_post_backward(self, *args, **kwargs):
        """Callback function to be executed after the `loss.backward()` call."""
        pass

    # not using decorators cuz messes with autoreload
    # @torch.no_grad()
    def _update_param_with_optimizer(
        self,
        param_fn: Callable[[str, Tensor], Tensor],
        optimizer_fn: Callable[[str, Tensor], Tensor],
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        names: list[str] | None = None,
    ):
        """Update the parameters and the state in the optimizers with defined functions.

        Args:
            param_fn: A function that takes the name of the parameter and the parameter itself,
                and returns the new parameter.
            optimizer_fn: A function that takes the key of the optimizer state and the state value,
                and returns the new state value.
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            names: A list of key names to update. If None, update all. Default: None.
        """
        with torch.no_grad():
            if names is None:
                # If names is not provided, update all parameters
                names = list(params.keys())

            for name in names:
                param = params[name]
                new_param = param_fn(name, param)
                params[name] = new_param  # type:ignore
                if name not in optimizers:
                    assert not param.requires_grad, (
                        f"Optimizer for {name} is not found, but the parameter is trainable."
                        f"Got requires_grad={param.requires_grad}"
                    )
                    continue
                optimizer = optimizers[name]
                for i in range(len(optimizer.param_groups)):
                    param_state = optimizer.state[param]
                    del optimizer.state[param]
                    for key in param_state.keys():
                        if key != "step":
                            v = param_state[key]
                            param_state[key] = optimizer_fn(key, v)
                    optimizer.param_groups[i]["params"] = [new_param]
                    optimizer.state[new_param] = param_state

    def _duplicate(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Tensor],
        mask: Tensor,
    ):
        """Inplace duplicate the Gaussian with the given mask.

        Args:
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            mask: A boolean mask to duplicate the Gaussians.
            state: A dictionary keeping track of the gradient and number of steps for each gaussian
        """
        with torch.no_grad():
            device = mask.device
            sel = torch.where(mask)[0]

            def param_fn(name: str, p: Tensor) -> Tensor:
                return torch.nn.Parameter(torch.cat([p, p[sel]]), requires_grad=p.requires_grad)

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

            # update the parameters and the state in the optimizers
            self._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
            # update the extra running state
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = torch.cat((v, v[sel]))

    def _split(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Tensor],
        mask: Tensor,
        revised_intensity: bool = False,
    ):
        """Inplace split the Gaussian with the given mask.

        Args:
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            mask: A boolean mask to split the Gaussians.
            revised_intensity: Whether to use revised intensity formulation
            from arXiv:2404.06109. Default: False.
        """
        with torch.no_grad():
            device = mask.device
            sel = torch.where(mask)[0]
            rest = torch.where(~mask)[0]

            sigmas = self.cfg.activation_sigma(params["sigmas"][sel])
            # quats = F.normalize(params["quats"][sel], dim=-1)
            # rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]

            # print("splitting")
            # samples = torch.einsum(
            #     "nij,nj,bnj->bni",
            #     rotmats,
            #     sigmas,
            #     torch.randn(2, len(sigmas), 3, device=device),
            # )  # [2, N, 3]
            samples = None

            def param_fn(name: str, p: Tensor) -> Tensor:
                repeats = [2] + [1] * (p.dim() - 1)
                if name == "positions":
                    print("pos split: ", p[sel].shape)
                    # print(p[sel])
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3] # type:ignore
                elif name == "sigmas":
                    p_split = torch.log(sigmas / 1.6).repeat(2, 1)  # [2N, 3]
                elif name == "intensities" and revised_intensity:
                    new_intensities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
                    p_split = torch.logit(new_intensities).repeat(repeats)  # [2N]
                else:
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
                return p_new

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                return torch.cat([v[rest], v_split])

            # update the parameters and the state in the optimizers
            self._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
            # update the extra running state
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    repeats = [2] + [1] * (v.dim() - 1)
                    v_new = v[sel].repeat(repeats)
                    state[k] = torch.cat((v[rest], v_new))

    # @torch.no_grad()
    def _merge(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Tensor],
        keeps: list,
        merges: list,
    ):
        with torch.no_grad():
            sigmas = self.cfg.activation_sigma(params["sigmas"])
            intensities = self.cfg.activation_intensity(params["intensities"])
            positions = params["positions"]
            combined_inds = [[kp] + mg for kp, mg in zip(keeps, merges)]

            # make a param_fn, has cases for positions, sigmas, intensities like split()
            # masks are changes, merges (does not include changes), shouldn't need a rest
            # run _update_param_with_optimizer, which will only affect changes but uses merges as well
            # then run remove, with the mask being merges, this will also take care of updating the state dict
            def param_fn(name: str, p: Tensor) -> Tensor:
                if name == "positions":
                    # Weighted mean by intensity
                    weighted_pos = [
                        torch.sum(p[inds] * intensities[inds][:, None], dim=0)
                        / torch.sum(intensities[inds])
                        for inds in combined_inds
                    ]
                    p[keeps] = torch.stack(weighted_pos)
                elif name == "sigmas":
                    # Weighted mean for sigma_y and sigma_x, keep sigma_z=0
                    weighted_sigmas = []
                    for inds in combined_inds:
                        sigma = torch.sum(
                            sigmas[inds] * intensities[inds][:, None], dim=0
                        ) / torch.sum(intensities[inds])
                        weighted_sigmas.append(sigma)
                    weighted_sigmas = torch.stack(weighted_sigmas)
                    # Inverse activation for storage
                    p[keeps] = self.cfg.activation_sigma_inverse_torch(weighted_sigmas)
                elif name == "intensities":
                    sum_intensities = [torch.sum(intensities[inds]) for inds in combined_inds]
                    p[keeps] = self.cfg.activation_intensity_inverse_torch(
                        torch.tensor(sum_intensities, device=p.device)
                    )
                else:
                    raise ValueError(f"Unexpected parameter name in merge: {name}")
                return p

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                # effectively just keeping the original values for the kept GS, as rest deleted
                return v

            self._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

            # run remove
            rm_mask = torch.zeros(
                len(params["positions"]), device=positions.device, dtype=torch.bool
            )
            rm_mask[np.concatenate(merges)] = 1
            self.remove(params=params, optimizers=optimizers, state=state, mask=rm_mask)

        return

    # @torch.no_grad()
    def remove(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Tensor],
        mask: Tensor,
    ):
        """Inplace remove the Gaussian with the given mask.

        Args:
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            mask: A boolean mask to remove the Gaussians.
        """
        with torch.no_grad():
            sel = torch.where(~mask)[0]

            def param_fn(name: str, p: Tensor) -> Tensor:
                return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                return v[sel]

            # update the parameters and the state in the optimizers
            self._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
            # update the extra running state
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v[sel]

    def reset_intensities(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Tensor],
        value: float,
    ):
        """Inplace reset the intensities to the given post-sigmoid value.

        Args:
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            value: The value to reset the intensities
        """

        with torch.no_grad():

            def param_fn(name: str, p: Tensor) -> Tensor:
                if name == "intensities":
                    intensities = torch.clamp(self.cfg.activation_intensity(p), max=value)
                    intensities = self.cfg.activation_intensity_inverse_torch(intensities)
                    return torch.nn.Parameter(intensities, requires_grad=p.requires_grad)
                else:
                    raise ValueError(f"Unexpected parameter name in reset_intensities: {name}")

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                return torch.zeros_like(v)

            # update the parameters and the state in the optimizers
            self._update_param_with_optimizer(
                param_fn, optimizer_fn, params, optimizers, names=["intensities"]
            )

    def reset_sigmas(
        self,
        params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Tensor],
        value: float,
    ):
        """Inplace reset the sigmas to the given post-sigmoid value.

        Args:
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            value: The value to reset the sigmas
        """

        with torch.no_grad():

            def param_fn(name: str, p: Tensor) -> Tensor:
                if name == "sigmas":
                    sigmas = torch.clamp(self.cfg.activation_sigma(p), max=value)
                    sigmas = self.cfg.activation_sigma_inverse_torch(sigmas)
                    return torch.nn.Parameter(sigmas, requires_grad=p.requires_grad)
                else:
                    raise ValueError(f"Unexpected parameter name in reset_sigmas: {name}")

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                return torch.zeros_like(v)

            # update the parameters and the state in the optimizers
            self._update_param_with_optimizer(
                param_fn, optimizer_fn, params, optimizers, names=["sigmas"]
            )

    # @torch.no_grad()
    def _add(
        self,
        old_params: dict[str, torch.nn.Parameter] | torch.nn.ParameterDict,
        new_params: dict[str, torch.Tensor] | torch.nn.ParameterDict,
        optimizers: dict[str, Optimizer],
        state: dict[str, Tensor],
    ):
        """Inplace duplicate the Gaussian with the given mask.

        Args:
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            mask: A boolean mask to duplicate the Gaussians.
        """
        with torch.no_grad():
            device = old_params["positions"].device
            n_new = len(new_params["positions"])

            def param_fn(name: str, p: Tensor) -> Tensor:
                p_new = torch.cat([p, new_params[name]])
                return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=device)])

            self._update_param_with_optimizer(param_fn, optimizer_fn, old_params, optimizers)

            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    repeats = [n_new] + [1] * (v.dim() - 1)
                    v_new = torch.tensor(0.0, device=device).repeat(repeats)
                    state[k] = torch.cat((v, v_new))

            # initialize state?
