from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from quantem.core import config
from quantem.core.config import (
    device_id_to_int as device_id_to_int,  # just adding to namespace
)

if TYPE_CHECKING:
    import cupy as cp  # type: ignore
    import torch  # type: ignore
else:
    if config.get("has_cupy"):
        import cupy as cp
    if config.get("has_torch"):
        import torch


def get_array_module(array: NDArray):
    """Returns np or cp depending on the array type."""
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp
    if isinstance(array, np.ndarray):
        return np
    raise ValueError(f"Input is not a numpy array or cupy array: {type(array)}")


def get_tensor_module(tensor: NDArray):
    """
    This is like get_array_module but includes torch. It is kept explicitly separate as in most
    cases get_array_module is used, and that fails if given a torch.Tensor.
    """
    if config.get("has_torch"):
        if isinstance(tensor, torch.Tensor):
            return torch
    if config.get("has_cupy"):
        if isinstance(tensor, cp.ndarray):
            return cp
    if isinstance(tensor, np.ndarray):
        return np
    raise ValueError(
        f"Input is not a numpy array, cupy array, or torch tensor: {type(tensor)}"
    )


def as_numpy(array: Any) -> np.ndarray:
    """Convert a torch.Tensor or cupy.ndarray to a numpy.ndarray."""
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    if config.get("has_torch"):
        if isinstance(array, torch.Tensor):
            return array.cpu().numpy()
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def subdivide_batches(
    num_items: int,
    num_batches: Optional[int] = None,
    max_batch: Optional[int] = None,
) -> List[int]:
    """
    Split `num_items` into a list of batch sizes.

    Parameters
    ----------
    num_items : int
        Total number of items to split into batches.
    num_batches : int, optional
        Number of desired batches. Cannot be used with `max_batch`.
    max_batch : int, optional
        Maximum batch size. Cannot be used with `num_batches`.

    Returns
    -------
    List[int]
        List of batch sizes that sum to `num_items`.
    """
    if num_batches is not None and max_batch is not None:
        raise RuntimeError("Specify only one of `num_batches` or `max_batch`.")

    if num_batches is None:
        if max_batch is None:
            raise RuntimeError("Must provide either `num_batches` or `max_batch`.")
        num_batches = (num_items + max_batch - 1) // max_batch

    if num_items < num_batches:
        raise ValueError("`num_batches` may not exceed `num_items`.")

    base_size = num_items // num_batches
    remainder = num_items % num_batches

    return [base_size + 1] * remainder + [base_size] * (num_batches - remainder)


def generate_batches(
    num_items: int,
    num_batches: Optional[int] = None,
    max_batch: Optional[int] = None,
    start_index: int = 0,
) -> Iterator[Tuple[int, int]]:
    """
    Yield (start, end) index tuples for each batch.

    Parameters
    ----------
    num_items : int
        Total number of items to batch.
    num_batches : int, optional
        Number of batches. Cannot be used with `max_batch`.
    max_batch : int, optional
        Maximum size of each batch. Cannot be used with `num_batches`.
    start_index : int, default = 0
        Optional offset to start indexing from.

    Yields
    ------
    (int, int)
        Tuple of (start, end) indices for each batch.
    """
    batch_sizes = subdivide_batches(num_items, num_batches, max_batch)
    idx = start_index
    for size in batch_sizes:
        yield idx, idx + size
        idx += size


class SimpleTorchBatcher:
    def __init__(
        self, indices: torch.Tensor, batch_size: int | None = None, shuffle: bool = True
    ):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.indices = self.indices[torch.randperm(len(self.indices))]

        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i : i + self.batch_size]
