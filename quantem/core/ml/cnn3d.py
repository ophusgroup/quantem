import warnings
from typing import Callable

import torch
import torch.nn as nn

from .activation_functions import get_activation_function
from .blocks import Conv3dBlock, Upsample3dBlock, complex_pool, passfunc


class CNN3d(nn.Module):
    """Complex 3D CNN for processing volumetric data."""

    def __init__(
        self,
        shape: tuple[int, int, int, int],  # (C, D, H, W)
        start_filters: int = 16,
        num_layers: int = 3,
        num_per_layer: int = 2,
        use_skip_connections: bool = False,
        dtype: torch.dtype = torch.complex64,
        dropout: float = 0,
        activation: str | Callable = "relu",
        final_activation: (
            str | Callable | None
        ) = None,  # -> Identity or softplus depending on mode
        use_batchnorm: bool = True,
        mode: str = "complex",
    ):
        super().__init__()
        if len(shape) != 4:
            raise ValueError(f"Shape should be (C, D, H, W), got {shape}")

        self.inshape = self.outshape = shape
        for d in shape[1:]:
            if d % (2**num_layers) != 0:
                raise ValueError(
                    "Input/output shape must be divisible by 2^num_layers, got "
                    + f"shape={shape} and num_layers={num_layers}"
                )
        self.start_filters = start_filters
        self.num_layers = num_layers
        self._num_per_layer = num_per_layer
        self.use_skip_connections = use_skip_connections
        self.dtype = dtype
        self.dropout = dropout
        self._use_batchnorm = use_batchnorm
        self.mode = mode

        self.pool = complex_pool if dtype.is_complex else passfunc
        self._pooler = nn.MaxPool3d(kernel_size=2, stride=2)

        self.activation = activation
        if final_activation is None:
            if self.mode == "potential":
                self.final_activation = nn.Softplus()
            else:
                self.final_activation = nn.Identity()
        else:
            self.final_activation = final_activation

        self._build()

    def _build(self):
        self.down_conv_blocks = nn.ModuleList()
        self.up_conv_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        in_channels = self.inshape[0]
        out_channels = self.start_filters
        for a0 in range(self.num_layers):
            if a0 != 0:
                out_channels = in_channels * 2
            self.down_conv_blocks.append(
                Conv3dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=in_channels,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm,
                    dropout=self.dropout,
                    dtype=self.dtype,
                    activation=self.activation,
                )
            )
            in_channels = out_channels

        out_channels = in_channels * 2
        self.bottleneck = Conv3dBlock(
            nb_layers=self._num_per_layer,
            input_channels=in_channels,
            output_channels=out_channels,
            use_batchnorm=self._use_batchnorm,
            dropout=self.dropout,
            dtype=self.dtype,
            activation=self.activation,
        )
        in_channels = out_channels

        for a0 in range(self.num_layers):
            if a0 == self.num_layers - 1:
                out_channels = self.start_filters
            else:
                out_channels = in_channels // 2

            if self.use_skip_connections:
                in_channels2 = in_channels
            else:
                in_channels2 = out_channels

            self.upsample_blocks.append(
                Upsample3dBlock(
                    in_channels, out_channels, use_batchnorm=self._use_batchnorm, dtype=self.dtype
                )
            )

            self.up_conv_blocks.append(
                Conv3dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=in_channels2,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm,
                    dropout=self.dropout,
                    dtype=self.dtype,
                    activation=self.activation,
                )
            )

            in_channels = out_channels

        self.final_conv = Conv3dBlock(
            nb_layers=1,
            input_channels=self.start_filters,
            output_channels=self.outshape[0],
            use_batchnorm=self._use_batchnorm,
            dropout=self.dropout,
            dtype=self.dtype,
            activation=self.final_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down_block in self.down_conv_blocks:
            x = down_block(x)
            if self.use_skip_connections:
                skips.append(x)
            x = self.pool(x, self._pooler)

        x = self.bottleneck(x)
        for upsample_block, up_conv_block in zip(self.upsample_blocks, self.up_conv_blocks):
            x = upsample_block(x)
            if self.use_skip_connections:
                x = torch.cat((x, skips.pop()), dim=1)
            x = up_conv_block(x)

        return self.final_conv(x)

    @property
    def activation(self) -> Callable:
        return self._activation

    @activation.setter
    def activation(self, act: str | Callable):
        if isinstance(act, Callable):
            self._activation = act
        else:
            self._activation = get_activation_function(act, self.dtype)

    @property
    def final_activation(self) -> Callable:
        return self._final_activation

    @final_activation.setter
    def final_activation(self, act: str | Callable):
        if isinstance(act, Callable):
            self._final_activation = act
        else:
            self._final_activation = get_activation_function(act, self.dtype)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, m: str):
        m = m.lower()
        if m in ["object", "complex", "pure_phase", "purephase"]:
            if self.dtype.is_complex:
                if m in ["pure_phase", "purephase"]:
                    warnings.warn(
                        "Object type is 'pure_phase' but dtype is complex. "
                        + "Setting object type to 'complex'"
                    )
                mode = "complex"
            else:
                if m in ["object", "complex"]:
                    warnings.warn(
                        f"Object type is {m} but dtype is real. "
                        + "Setting object type to 'pure_phase'"
                    )
                mode = "pure_phase"
        elif m in ["probe"]:
            if not self.dtype.is_complex:
                raise TypeError(f"Mode is probe -> dtype must be complex, but got {self.dtype}")
            mode = "probe"
        elif m in ["potential"]:
            if self.dtype.is_complex:
                raise TypeError(f"Mode is potential -> dtype must be real, but got {self.dtype}")
            mode = "potential"
        else:
            raise ValueError(
                f"Unknown mode '{m}', should be 'object', 'pure_phase', 'potential', or 'probe'"
            )
        self._mode = mode
