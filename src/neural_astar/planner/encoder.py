"""Encoder module
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class EncoderBase(nn.Module):
    def __init__(self, input_dim: int, encoder_depth: int = 4, const: float | None = None):
        super().__init__()
        self.model = self.construct_encoder(input_dim, encoder_depth)
        if const is not None:
            self.const = nn.Parameter(torch.ones(1) * const)
        else:
            self.const = 1.0

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sigmoid(self.model(x))
        return y * self.const


class Unet(EncoderBase):
    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        return smp.Unet(
            encoder_name="vgg16_bn",
            encoder_weights=None,
            classes=1,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )


class CNN(EncoderBase):
    CHANNELS = [32, 64, 128, 256]

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        channels = [input_dim] + self.CHANNELS[:encoder_depth] + [1]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
        return nn.Sequential(*blocks[:-1])


class CNNDownSize(CNN):
    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        channels = [input_dim] + self.CHANNELS[:encoder_depth] + [1]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d((2, 2)))
        return nn.Sequential(*blocks[:-2])
