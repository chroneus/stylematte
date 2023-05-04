from typing import Callable, List, Optional
import torch

import torch.nn.functional as F
from torch import nn
import numpy as np


def generate_composition_img(LB, img, alpha_channel, bg_img):
    """
    img: image from StyleGAN [-1, 1]
    alpha_channel: [0, 1]
    bg_img: [0, 1]
    output: [-1, 1]
    """
    alpha_channel = alpha_channel.unsqueeze(1)
    img = (img + 1) / 2
    alpha_channel = (alpha_channel + 1) / 2
    bg_img = (bg_img + 1) / 2
    out = LB(img, bg_img, alpha_channel)
    return 2 * out - 1


class SimpleBlending(nn.Module):
    def __init__(self, iters=4):
        super().__init__()
        self.iters = iters

    @staticmethod
    def scale(x, s):
        x = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=True)
        return x

    @staticmethod
    def run(source, target, mask):
        return source * mask + target * (1 - mask)

    def get_blend_mask(self, mask):
        for i in range(self.iters):
            mask = self.scale(mask, 0.5)
        for i in range(self.iters):
            mask = self.scale(mask, 2)
        return mask

    def forward(self, x, y, mask):
        mask = self.get_blend_mask(mask)
        xy_blend = self.run(x, y, mask)
        xy_blend = torch.clamp(xy_blend, 0.0, 1.0)
        return xy_blend


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size, sigma=0, channels=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, 1, padding, groups=channels, bias=False,
                              padding_mode='reflect')
        self.init_weight(kernel_size, sigma, channels)

    @staticmethod
    @torch.no_grad()
    def get_kernel(kernel_size, sigma=0):
        if sigma <= 0:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        center = kernel_size // 2
        xs = (np.arange(kernel_size, dtype=np.float32) - center)
        kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
        kernel = kernel1d[..., None] @ kernel1d[None, ...]
        kernel = torch.from_numpy(kernel)
        kernel = kernel / kernel.sum()
        return kernel.type(torch.float32)

    def init_weight(self, kernel_size, sigma, channels):
        kernel = self.get_kernel(kernel_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.conv.weight.data = kernel

    def forward(self, img):
        return self.conv(img)


class LaplacianBlending(nn.Module):
    def __init__(self, kernel_size=3, sigma=0, channels=3, iters=4):
        super().__init__()
        self.gaussian = GaussianBlur(kernel_size, sigma, channels)
        self.iters = iters

    @staticmethod
    def scale(x, s):
        x = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=True)
        return x

    def down(self, x, y, mask):
        input = torch.cat((x, y))
        input_blur = self.gaussian(input)
        input_blur_half = self.scale(input_blur, 0.5)
        input_lap = input - self.scale(input_blur_half, 2)
        mask_half = self.scale(mask, 0.5)
        x_blur_half, y_blur_half = torch.chunk(input_blur_half, 2)
        x_lap, y_lap = torch.chunk(input_lap, 2)

        return x_blur_half, y_blur_half, x_lap, y_lap, mask_half

    @staticmethod
    def run(x, y, mask):
        return x * mask + y * (1 - mask)

    def up(self, xy_blend, x_lap, y_lap, mask):
        out = self.scale(xy_blend, 2)
        diff = self.run(x_lap, y_lap, mask)
        out = out + diff
        return out

    def forward(self, x, y, mask):
        x_laps = []
        y_laps = []
        masks = [mask]
        for it in range(self.iters):
            x, y, x_lap, y_lap, mask = self.down(x, y, mask)
            x_laps.append(x_lap)
            y_laps.append(y_lap)
            masks.append(mask)

        xy_blend = self.run(x, y, masks[-1])
        for it in range(self.iters):
            idx = self.iters - 1 - it
            x_lap = x_laps[idx]
            y_lap = y_laps[idx]
            msk = masks[idx]
            xy_blend = self.up(xy_blend, x_lap, y_lap, msk)

        xy_blend = torch.clamp(xy_blend, 0.0, 1.0)
        return xy_blend


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.
    from torchvision
    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: List[int],
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            inplace: Optional[bool] = True,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
