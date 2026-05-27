"""
Custom ResNet for CIFAR-10.

Components used:
  - Conv2d, BatchNorm2d, Residual connections, Dropout
  - Multiple activation options (ReLU, GELU, SiLU)
  - Global Average Pooling + Fully Connected layer
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="relu", dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = _get_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class CifarResNet(nn.Module):
    """
    A compact ResNet tailored for CIFAR-10 (32x32 input).

    Architecture:
        conv(3->64) -> [ResBlock x n] * 3 stages -> GAP -> FC(10)
        Stage channels: 64 -> 128 -> 256
        Downsampling via stride=2 at stage transitions.
    """

    def __init__(
        self,
        num_classes=10,
        blocks_per_stage=(3, 3, 3),
        channels=(64, 128, 256),
        activation="relu",
        dropout=0.1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.act = _get_activation(activation)

        layers = []
        in_ch = channels[0]
        for stage_idx, (num_blocks, out_ch) in enumerate(zip(blocks_per_stage, channels)):
            for block_idx in range(num_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                layers.append(ResidualBlock(in_ch, out_ch, stride, activation, dropout))
                in_ch = out_ch
        self.stages = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


def _get_activation(name):
    activations = {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
    }
    return activations[name]


def build_model(config=None):
    """Build model from config dict. Returns model and a description string."""
    defaults = {
        "num_classes": 10,
        "blocks_per_stage": (3, 3, 3),
        "channels": (64, 128, 256),
        "activation": "relu",
        "dropout": 0.1,
    }
    if config:
        defaults.update(config)

    model = CifarResNet(**defaults)
    n_params = sum(p.numel() for p in model.parameters())
    desc = (
        f"CifarResNet | blocks={defaults['blocks_per_stage']} "
        f"channels={defaults['channels']} act={defaults['activation']} "
        f"drop={defaults['dropout']} | params={n_params:,}"
    )
    return model, desc
