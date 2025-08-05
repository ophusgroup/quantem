import numpy as np
import torch

from quantem.core import config


class RNGMixin:
    """
    Mixin class providing consistent RNG functionality with both numpy and torch generators.
    If you do not provide a seed, the RNG will be initialized with a random seed, and subsequently
    resetting the RNG will use a _new_ random seed. Setting with a fixed seed or generator and then
    resetting the RNG will use the same seed.

    Provides:
    - self.rng: np.random.Generator property
    - self._rng_torch: torch.Generator for torch operations
    - self._reset_rng(): reset the RNG to the current seed
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = None

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator | int | None):
        if rng is None:
            self._rng_seed = None
            rng = np.random.default_rng()
        elif isinstance(rng, (int, float)):
            self._rng_seed = rng
            rng = np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator):
            self._rng_seed = rng.bit_generator._seed_seq.entropy  # type:ignore ## get the seed
        else:
            raise TypeError(f"rng should be a np.random.Generator or a seed, got {type(rng)}")

        self._rng = rng
        self._update_torch_rng()

    def _update_torch_rng(self):
        """Update the torch generator with current seed and device."""
        if self._rng_seed is not None:
            device = getattr(self, "device", "cpu")
            if self._rng_seed is None:
                self._rng_torch = torch.Generator(device=device)
            else:
                self._rng_torch = torch.Generator(device=device).manual_seed(
                    self._rng_seed % 2**32
                )

    def _reset_rng(self):
        """Reset RNG to current seed, useful for reproducible iterations."""
        if self._rng is not None:
            self.rng = self._rng_seed  # sets rng and _rng_torch

    def _rng_to_device(self, device: str | int | torch.device):
        """Update torch RNG when device changes."""
        if hasattr(self, "_rng_seed") and self._rng_seed is not None:
            dev, _id = config.validate_device(device)
            self._rng_torch = torch.Generator(device=dev).manual_seed(self._rng_seed % 2**32)
