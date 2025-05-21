from typing import Callable, Optional

import torch
from torch.optim.optimizer import Optimizer


class PolakRibiereCG(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1.0,
        restart_interval: int = 10,
        line_search: bool = False,
        ls_max_iter: int = 10,
        ls_rho: float = 0.5,
        ls_c: float = 1e-4,
    ):
        """
        CG optimizer with optional backtracking line search.

        Args:
            params: iterable of torch.Tensor
            lr: initial step size
            restart_interval: steps before CG restart
            line_search: whether to use backtracking line search
            ls_max_iter: max line search iterations
            ls_rho: backtracking contraction factor
            ls_c: Armijo condition constant
        """
        defaults = {
            "lr": lr,
            "restart_interval": restart_interval,
            "step_count": 0,
            "line_search": line_search,
            "ls_max_iter": ls_max_iter,
            "ls_rho": ls_rho,
            "ls_c": ls_c,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is None:
            raise RuntimeError(
                "PolakRibiereCG requires a closure to evaluate loss and gradients."
            )

        loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            restart_interval = group["restart_interval"]
            group["step_count"] += 1
            do_restart = group["step_count"] % restart_interval == 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                state = self.state[p]

                if do_restart or "prev_grad" not in state:
                    direction = -grad.clone()
                    state["direction"] = direction
                    state["prev_grad"] = grad.clone()
                else:
                    prev_grad = state["prev_grad"]
                    direction = state["direction"]

                    y = grad - prev_grad
                    beta = (
                        torch.dot(grad.view(-1), y.view(-1)).real
                        / torch.dot(prev_grad.view(-1), prev_grad.view(-1)).real
                    )
                    beta = torch.clamp(beta, min=0.0)  # PR+

                    direction = -grad + beta * direction
                    state["direction"] = direction
                    state["prev_grad"] = grad.clone()

                if group["line_search"]:
                    # Save current parameter
                    orig = p.clone()
                    alpha = lr
                    loss_before = loss
                    grad_dot_dir = torch.dot(grad.view(-1), direction.view(-1)).real

                    for _ in range(group["ls_max_iter"]):
                        p.copy_(orig + alpha * direction)
                        loss_new = closure()
                        if (
                            loss_new
                            <= loss_before + group["ls_c"] * alpha * grad_dot_dir
                        ):
                            break
                        alpha *= group["ls_rho"]
                    else:
                        p.copy_(orig)  # Revert if line search failed

                else:
                    p.add_(direction, alpha=lr)

        return loss
