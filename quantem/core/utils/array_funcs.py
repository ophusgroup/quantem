"""
Array module ambivalent functions, e.g. clip, exp, min, max, etc. that apply to numpy/cupy and torch
numpy/cupy are interchangeable and treated the same, but torch is not
"""

from typing import TYPE_CHECKING, Union, overload

import numpy as np

from quantem.core import config

if TYPE_CHECKING:
    import cupy as cp
    import torch
else:
    if config.get("has_cupy"):
        import cupy as cp
    if config.get("has_torch"):
        import torch

ArrayLike = Union[np.ndarray, "cp.ndarray", "torch.Tensor"]

# https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_utils.py#L1791
numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.uint16: torch.uint16,
    np.uint32: torch.uint32,
    np.uint64: torch.uint64,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}
torch_to_numpy_dtype_dict.update(
    {torch.bfloat16: np.float32, torch.complex32: np.complex64}
)


def numpy_to_torch_dtype(np_dtype) -> torch.dtype:
    try:
        return numpy_to_torch_dtype_dict[np_dtype]
    except KeyError:
        return numpy_to_torch_dtype_dict[np_dtype.type]


def get_array_module(array: "torch.Tensor | np.ndarray | cp.ndarray"):
    if config.get("has_torch"):
        if isinstance(array, torch.Tensor):
            return torch
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp
    if isinstance(array, np.ndarray):
        return np
    else:
        raise TypeError(f"Unsupported array type for get_array_module: {type(array)}")


def get_xp_module(array: "torch.Tensor | np.ndarray | cp.ndarray"):
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            return cp
    if isinstance(array, np.ndarray):
        return np
    else:
        raise TypeError(f"Unsupported array type for get_xp_module: {type(array)}")


def validate_arraylike(a: ArrayLike) -> None:
    """
    Validate that the input is a supported array-like type.
    """
    if not isinstance(a, (np.ndarray, cp.ndarray, torch.Tensor)):
        raise ValueError(f"Unsupported array type: {type(a)}")


def array_operation(a: ArrayLike, np_func, torch_func=None, *args, **kwargs):
    """
    Apply the appropriate operation based on array type.

    Args:
        a: Array-like input
        np_func: NumPy function to use
        torch_func: PyTorch function to use (if applicable)
        *args, **kwargs: Additional arguments to pass to the functions
    """
    validate_arraylike(a)
    if config.get("has_torch") and isinstance(a, torch.Tensor):
        if torch_func is None:
            raise NotImplementedError("Operation not implemented for torch tensors")
        return torch_func(a, *args, **kwargs)
    return np_func(a, *args, **kwargs)


@overload
def exp(a: np.ndarray) -> np.ndarray: ...
@overload
def exp(a: "torch.Tensor") -> "torch.Tensor": ...
def exp(a: ArrayLike) -> ArrayLike:
    """Compute the exponential of all elements in the input array."""
    return array_operation(a, np.exp, torch.exp)


@overload
def abs(a: np.ndarray) -> np.ndarray: ...
@overload
def abs(a: "torch.Tensor") -> "torch.Tensor": ...
def abs(a: ArrayLike) -> ArrayLike:
    """Compute the absolute value of all elements in the input array."""
    return array_operation(a, np.abs, torch.abs)


@overload
def angle(a: np.ndarray) -> np.ndarray: ...
@overload
def angle(a: "torch.Tensor") -> "torch.Tensor": ...
def angle(a: ArrayLike) -> ArrayLike:
    """Compute the phase angle value of all elements in the input array."""
    return array_operation(a, np.angle, torch.angle)


@overload
def sqrt(a: np.ndarray) -> np.ndarray: ...
@overload
def sqrt(a: "torch.Tensor") -> "torch.Tensor": ...
def sqrt(a: ArrayLike) -> ArrayLike:
    """Compute the square root of all elements in the input array."""
    return array_operation(a, np.sqrt, torch.sqrt)


@overload
def sum(a: np.ndarray, axis=None) -> np.ndarray: ...
@overload
def sum(a: "torch.Tensor", axis=None) -> "torch.Tensor": ...
def sum(a: ArrayLike, axis=None) -> ArrayLike:
    """Compute the sum of array elements over a given axis."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return torch.sum(a, dim=axis)
    return np.sum(a, axis=axis)


@overload
def clip(
    a: np.ndarray, a_min: float | None = None, a_max: float | None = None
) -> np.ndarray: ...
@overload
def clip(
    a: "torch.Tensor", a_min: float | None = None, a_max: float | None = None
) -> "torch.Tensor": ...
def clip(
    a: ArrayLike,
    a_min: float | None = None,
    a_max: float | None = None,
) -> ArrayLike:
    """Clip the values in an array to a given min and max value."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return torch.clamp(a, a_min, a_max)
    return np.clip(a, a_min, a_max)


@overload
def norm(
    a: np.ndarray,
    ord: int | None = None,
    axis: int | None = None,
    keepdim: bool = False,
) -> np.ndarray: ...
@overload
def norm(
    a: "torch.Tensor",
    ord: int | None = None,
    axis: int | None = None,
    keepdim: bool = False,
) -> "torch.Tensor": ...
def norm(
    a: ArrayLike,
    ord: int | None = None,
    axis: int | None = None,
    keepdim: bool = False,
) -> ArrayLike:
    """norm the values in an array to a given min and max value."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return torch.norm(a, p=ord, dim=axis, keepdim=keepdim)
    return np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdim)


@overload
def reshape(a: np.ndarray, newshape: tuple[int, ...]) -> np.ndarray: ...
@overload
def reshape(a: "torch.Tensor", newshape: tuple[int, ...]) -> "torch.Tensor": ...
def reshape(a: ArrayLike, newshape: tuple[int, ...]) -> ArrayLike:
    """Reshape an array to a new shape."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return a.view(*newshape)
    return np.reshape(a, newshape)


@overload
def min(
    a: np.ndarray, axis: int | None = None, keepdim: bool = False
) -> np.ndarray: ...
@overload
def min(
    a: "torch.Tensor", axis: int | None = None, keepdim: bool = False
) -> "torch.Tensor": ...
def min(a: ArrayLike, axis: int | None = None, keepdim: bool = False) -> ArrayLike:
    """Compute the minimum of array elements over a given axis."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return (
                torch.min(a, dim=axis, keepdim=keepdim)
                if axis is not None
                else torch.min(a)
            )
    return np.min(a, axis=axis, keepdims=keepdim)


@overload
def max(
    a: np.ndarray, axis: int | None = None, keepdim: bool = False
) -> np.ndarray: ...
@overload
def max(
    a: "torch.Tensor", axis: int | None = None, keepdim: bool = False
) -> "torch.Tensor": ...
def max(a: ArrayLike, axis: int | None = None, keepdim: bool = False) -> ArrayLike:
    """Compute the maximum of array elements over a given axis."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return (
                torch.max(a, dim=axis, keepdim=keepdim)
                if axis is not None
                else torch.max(a)
            )
    return np.max(a, axis=axis, keepdims=keepdim)


@overload
def mean(
    a: np.ndarray, axis: int | None = None, keepdim: bool = False
) -> np.ndarray: ...
@overload
def mean(
    a: "torch.Tensor", axis: int | None = None, keepdim: bool = False
) -> "torch.Tensor": ...
def mean(a: ArrayLike, axis: int | None = None, keepdim: bool = False) -> ArrayLike:
    """Compute the mean of array elements over a given axis."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return (
                torch.mean(a, dim=axis, keepdim=keepdim)
                if axis is not None
                else torch.mean(a)
            )
    return np.mean(a, axis=axis, keepdims=keepdim)


@overload
def fft2(a: np.ndarray) -> np.ndarray: ...
@overload
def fft2(a: "torch.Tensor") -> "torch.Tensor": ...
def fft2(a: ArrayLike) -> ArrayLike:
    """Compute the 2-dimensional discrete Fourier Transform."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return torch.fft.fft2(a)
    return np.fft.fft2(a)


@overload
def ifft2(a: np.ndarray) -> np.ndarray: ...
@overload
def ifft2(a: "torch.Tensor") -> "torch.Tensor": ...
def ifft2(a: ArrayLike) -> ArrayLike:
    """Compute the 2-dimensional inverse discrete Fourier Transform."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return torch.fft.ifft2(a)
    return np.fft.ifft2(a)


@overload
def fftshift(
    a: np.ndarray, axes: tuple[int, ...] | int | None = None
) -> np.ndarray: ...
@overload
def fftshift(
    a: "torch.Tensor", axes: tuple[int, ...] | int | None = None
) -> "torch.Tensor": ...
def fftshift(a: ArrayLike, axes: tuple[int, ...] | int | None = None) -> ArrayLike:
    """Shift the zero-frequency component to the center of the spectrum."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return torch.fft.fftshift(a, dim=axes)
    return np.fft.fftshift(a, axes=axes)


@overload
def as_type(a: np.ndarray, dtype: "type|str|torch.dtype") -> np.ndarray: ...
@overload
def as_type(a: "torch.Tensor", dtype: "type|str|torch.dtype") -> "torch.Tensor": ...
def as_type(a: ArrayLike, dtype: "type|str|torch.dtype") -> ArrayLike:
    """Cast the array to a specified type."""
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            if isinstance(dtype, torch.dtype):
                dt = dtype
            elif isinstance(dtype, str):
                dt = getattr(torch, dtype)
            elif isinstance(dtype, type):
                dt = numpy_to_torch_dtype(dtype)
            else:
                raise TypeError(f"Unsupported dtype for torch: {dtype}")
            return a.type(dt)
    if config.get("has_cupy"):
        if isinstance(a, cp.ndarray):
            if isinstance(dtype, (type, str)):
                dt = np.dtype(dtype)
            elif isinstance(dtype, torch.dtype):
                dt = torch_to_numpy_dtype_dict[dtype]
            else:
                raise TypeError(f"Unsupported dtype for cupy: {dtype}")
            return a.astype(dt)  # type:ignore ## cupy is fricken annoying sometimes
    if isinstance(a, np.ndarray):
        if isinstance(dtype, (type, str)):
            dt = np.dtype(dtype)
        elif isinstance(dtype, torch.dtype):
            dt = torch_to_numpy_dtype_dict[dtype]
        else:
            raise TypeError(f"Unsupported dtype for numpy: {dtype}")
        return a.astype(dt)
    else:
        raise ValueError(f"Unsupported array type: {type(a)}")


@overload
def match_device(inp_arr: ArrayLike, dev_arr: np.ndarray) -> np.ndarray: ...
@overload
def match_device(inp_arr: ArrayLike, dev_arr: "torch.Tensor") -> "torch.Tensor": ...
def match_device(inp_arr: ArrayLike, dev_arr: ArrayLike) -> ArrayLike:
    """
    Match the device of inp_arr to that of dev_arr.
    """
    if config.get("has_torch"):
        if isinstance(dev_arr, torch.Tensor):
            if isinstance(inp_arr, torch.Tensor):
                return inp_arr.to(device=dev_arr.device)
            elif isinstance(inp_arr, (tuple, list)):
                if isinstance(inp_arr[0], torch.Tensor):  # assume the rest are too...
                    return torch.stack(inp_arr).to(device=dev_arr.device)
                elif isinstance(inp_arr[0], np.ndarray):
                    return torch.tensor(np.array(inp_arr), device=dev_arr.device)
                elif isinstance(inp_arr[0], cp.ndarray):
                    return torch.tensor(cp.array(inp_arr), device=dev_arr.device)
                else:
                    return torch.tensor(inp_arr, device=dev_arr.device)
            else:
                # raise error?
                return torch.tensor(inp_arr, device=dev_arr.device)
    if config.get("has_cupy"):
        if isinstance(dev_arr, cp.ndarray):
            if isinstance(inp_arr, cp.ndarray):
                return inp_arr
            else:
                return cp.asarray(inp_arr)
    if isinstance(dev_arr, np.ndarray):
        if isinstance(inp_arr, np.ndarray):
            return inp_arr
        elif isinstance(inp_arr, cp.ndarray):
            return cp.asnumpy(inp_arr)
        elif isinstance(inp_arr, torch.Tensor):
            return inp_arr.cpu().detach().numpy()
        else:
            return np.array(inp_arr)
    else:
        raise TypeError(f"Unsupported array type: {type(dev_arr)}")


@overload
def argsort(a: np.ndarray, axis: int = -1) -> np.ndarray: ...
@overload
def argsort(a: "torch.Tensor", axis: int = -1) -> "torch.Tensor": ...
def argsort(a: ArrayLike, axis: int = -1) -> ArrayLike:
    """Return the indices that would sort an array along a specified axis."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            return torch.argsort(a, dim=axis)
    return np.argsort(a, axis=axis)


@overload
def flip(a: np.ndarray, axis: tuple | int | None = None) -> np.ndarray: ...
@overload
def flip(a: "torch.Tensor", axis: tuple | int | None = None) -> "torch.Tensor": ...
def flip(a: ArrayLike, axis: tuple | int | None = None) -> ArrayLike:
    """Reverse the order of elements in an array along the given axis."""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            dims = (axis,) if isinstance(axis, int) else axis
            return torch.flip(
                a, dims=dims if dims is not None else tuple(range(a.ndim))
            )
    return np.flip(a, axis=axis)


@overload
def stack(arrays: list[np.ndarray], axis: int = 0) -> np.ndarray: ...
@overload
def stack(arrays: list["torch.Tensor"], axis: int = 0) -> "torch.Tensor": ...
def stack(arrays: list[ArrayLike], axis: int = 0) -> ArrayLike:
    """Stack a sequence of arrays along a new axis."""
    if not arrays:
        raise ValueError("Need at least one array to stack.")
    for array in arrays:
        validate_arraylike(array)
    if config.get("has_torch"):
        if all(isinstance(array, torch.Tensor) for array in arrays):
            return torch.stack(arrays, dim=axis)  # type:ignore
    if config.get("has_cupy"):
        if all(isinstance(array, cp.ndarray) for array in arrays):
            return cp.stack(arrays, axis=axis)
    if all(isinstance(array, np.ndarray) for array in arrays):
        return np.stack(arrays, axis=axis)
    raise TypeError("All arrays must be of the same type to stack.")


@overload
def repeat(a: np.ndarray, repeats: int, axis: int | None = None) -> np.ndarray: ...
@overload
def repeat(
    a: "torch.Tensor", repeats: int, axis: int | None = None
) -> "torch.Tensor": ...
def repeat(a: ArrayLike, repeats: int, axis: int | None = None) -> ArrayLike:
    """Repeat elements of an array along a specified axis. Like np.repeat"""
    validate_arraylike(a)
    if config.get("has_torch"):
        if isinstance(a, torch.Tensor):
            if axis is None:
                return a.flatten().repeat(repeats)
            else:
                return a.repeat_interleave(repeats, dim=axis)
    return np.repeat(a, repeats, axis=axis)
