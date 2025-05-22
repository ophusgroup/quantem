from typing import List, Self, Tuple

import torch
import torch.nn as nn

from quantem.core.datastructures import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.diffractive_imaging.ptychography.ptychography_utils import (
    compute_propagator_array,
    fourier_convolve_array,
    sum_overlapping_patches,
)

# region --- Pixelated Object ---


class PixelatedObjectModel(AutoSerialize):
    """ """

    _token = object()

    def __init__(
        self,
        obj_dataset: Dataset3d,
        energy: float,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use PixelatedObjectModel.from_array() or PixelatedObjectModel.from_positions() to instantiate this class."
            )

        self.dataset = obj_dataset
        self.energy = energy
        self.obj_shape = self.dataset.shape[-2:]
        self.sampling = self.dataset.sampling[-2:]
        self.num_slices = self.dataset.shape[0]
        self.slice_thickness = self.dataset.sampling[0]
        self.propagator_array = compute_propagator_array(
            energy, self.obj_shape, self.sampling, self.slice_thickness
        )

    @classmethod
    def from_array(
        cls,
        array: torch.Tensor,
        energy: float,
        sampling: Tuple[int, int],
        slice_thickness: float,  # TODO: support varying slice-thicknesses
    ) -> Self:
        obj_dataset = Dataset3d.from_array(
            torch.as_tensor(array).to(torch.cfloat),
            name="ptychographic object",
            sampling=(slice_thickness,) + tuple(sampling),
            units=("A", "A", "A"),
        )

        return cls(
            obj_dataset,
            energy,
            cls._token,
        )

    @classmethod
    def from_positions(
        cls,
        energy: float,
        positions_px: torch.Tensor,
        padding_px: Tuple[int, int, int, int],
        sampling: Tuple[int, int],
        slice_thickness: float,
        num_slices: int,
    ) -> Self:
        """ """

        bbox = cls._calculate_positions_bbox(positions_px, padding_px)
        obj = torch.ones(*bbox, dtype=torch.cfloat)
        obj = torch.tile(obj, (num_slices, 1, 1))

        return cls.from_array(
            obj,
            energy,
            sampling,
            slice_thickness,
        )

    @staticmethod
    def _calculate_positions_bbox(positions_px, padding_px):
        """ """
        bbox = positions_px.max(dim=0).values - positions_px.min(dim=0).values
        bbox = torch.round(bbox).to(torch.int)
        padding_dims = torch.tensor(padding_px).reshape(2, 2).sum(1)
        return bbox + padding_dims

    def forward(
        self,
        probe_array: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
    ):
        """ """
        num_slices = self.num_slices
        propagator = self.propagator_array
        obj_patches = torch.unsqueeze(self.tensor[..., row, col], 2)

        shape = (num_slices,) + probe_array.shape
        propagated_probes = torch.empty(shape, dtype=probe_array.dtype)
        propagated_probes[0] = probe_array

        for s in range(num_slices):
            exit_waves = obj_patches[s] * propagated_probes[s]
            if s + 1 < num_slices:
                propagated_probes[s + 1] = fourier_convolve_array(
                    exit_waves, propagator
                )

        return propagated_probes, obj_patches, exit_waves

    def backward(self, gradient_array, probe_array, obj_patches, positions_px):
        """ """
        if self.tensor.requires_grad:
            num_slices = self.num_slices
            propagator = self.propagator_array.conj()
            obj_gradient = torch.empty_like(self.tensor)

            for s in reversed(range(num_slices)):
                probe = probe_array[s]
                obj = obj_patches[s]

                probe_normalization = (
                    sum_overlapping_patches(
                        torch.square(torch.abs(probe)), positions_px, self.obj_shape
                    )
                    + 1e-10
                )

                obj_gradient[s] = (
                    sum_overlapping_patches(
                        gradient_array * torch.conj(probe),
                        positions_px,
                        self.obj_shape,
                    )
                    / probe_normalization
                )

                if s > 0:
                    gradient_array *= torch.conj(obj)  # back-transmit
                    gradient_array = fourier_convolve_array(
                        gradient_array, propagator
                    )  # back-propagate

            self.tensor.grad = obj_gradient.clone().detach()

            return gradient_array

    @property
    def tensor(self) -> torch.Tensor:
        return self.dataset.array

    def parameters(self) -> List[torch.Tensor]:
        return [self.tensor]


# endregion --- Pixelated Object ---

# region --- DIP ---


class ConvNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 32,
        hidden_channels: int = 64,
        output_channels: int = 1,
        kernel_size: int = 3,
        depth: int = 4,
        num_slices: int = 1,
    ):
        """
        Simple convolutional network for DIP-style object model.

        Args:
            input_channels: number of input noise channels (e.g. 32)
            hidden_channels: number of channels in hidden layers (e.g. 64)
            output_channels: number of channels per slice (e.g. 1 for phase-only or complex)
            kernel_size: kernel size for intermediate convolutions (e.g. 3 for 3x3)
            depth: number of conv + ReLU blocks
            num_slices: number of object slices in z (for 3D object)
        """
        super().__init__()

        layers = []
        in_channels = input_channels
        for _ in range(depth):
            layers.append(
                nn.Conv2d(
                    in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_channels

        # Final layer: complex output = real + imag = 2 channels per slice
        final_channels = 2 * num_slices * output_channels
        layers.append(nn.Conv2d(hidden_channels, final_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)
        self.num_slices = num_slices
        self.output_channels = output_channels
        self.input_channels = input_channels

    def forward(self, x):
        out = self.net(x)
        real, imag = out.chunk(2, dim=1)  # split channel dimension into real + imag
        complex_out = torch.complex(real, imag)

        # Reshape to [num_slices, output_channels, H, W] if needed
        B, C, H, W = complex_out.shape
        complex_out = complex_out.view(self.num_slices, self.output_channels, H, W)
        return complex_out.squeeze(1)  # remove channel dim if output_channels == 1


class DeepImagePriorObjectModel(PixelatedObjectModel):
    """ """

    _token = object()

    def __init__(
        self,
        conv_net: ConvNet,
        obj_shape: Tuple[int, int],
        sampling: Tuple[float, float],
        slice_thickness: float,
        energy: float,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DeepImagePriorObjectModel.from_positions() to instantiate this class."
            )

        self.net = conv_net
        self.energy = energy
        self.obj_shape = obj_shape
        self.sampling = sampling
        self.slice_thickness = slice_thickness
        self.propagator_array = compute_propagator_array(
            energy, self.obj_shape, self.sampling, self.slice_thickness
        )

        self.num_slices = self.net.num_slices

        self.input_noise = torch.randn(
            self.net.output_channels, self.net.input_channels, *self.obj_shape
        )
        self._scale_output_layer_weights()

    def _scale_output_layer_weights(self):
        """scale output layer weights to give near unit amplitude"""
        with torch.no_grad():
            out = self.net(self.input_noise)
            real, imag = out.chunk(2, dim=1)
            amp = torch.sqrt(torch.square(real) + torch.square(imag))
            mean_amp = torch.abs(amp.mean())

            scale = 1.0 / (mean_amp + 1e-8)
            final_layer = self.net.net[-1]
            final_layer.weight.data *= scale
            if final_layer.bias is not None:
                final_layer.bias.data *= scale

        return None

    @classmethod
    def from_positions(
        cls,
        energy: float,
        positions_px: torch.Tensor,
        padding_px: Tuple[int, int, int, int],
        sampling: Tuple[int, int],
        slice_thickness: float,
        num_slices: int,
        net_input_channels: int = 32,
        net_hidden_channels: int = 64,
        net_output_channels: int = 1,
        net_kernel_size: int = 3,
        net_image_depth: int = 4,
    ) -> Self:
        """ """

        obj_shape = cls._calculate_positions_bbox(positions_px, padding_px)

        conv_net = ConvNet(
            net_input_channels,
            net_hidden_channels,
            net_output_channels,
            net_kernel_size,
            net_image_depth,
            num_slices,
        )

        return cls(
            conv_net,
            obj_shape,
            sampling,
            slice_thickness,
            energy,
            cls._token,
        )

    def backward(self, *args, **kwargs):
        raise NotImplementedError("Use autograd for DeepImagePriorObjectModel.")

    @property
    def tensor(self) -> torch.Tensor:
        out = self.net(self.input_noise)
        return out.view(self.num_slices, *self.obj_shape)

    def parameters(self):  # -> Generator[nn.Parameter]:
        return self.net.parameters()


# region --- DIP ---
