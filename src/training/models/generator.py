import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, features, n_residual_blocks: int = 9, hidden_blocks: int = 64):
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, hidden_blocks, 7),
            nn.InstanceNorm2d(hidden_blocks),
            nn.ReLU(inplace=True),
        ]

        hidden_blocks = 64
        out_features = hidden_blocks * 2
        for _ in range(2):
            model += [
                nn.Conv2d(hidden_blocks, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            hidden_blocks = out_features
            out_features = hidden_blocks * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(hidden_blocks)]

        out_features = hidden_blocks // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    hidden_blocks,
                    out_features,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            hidden_blocks = out_features
            out_features = hidden_blocks // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(hidden_blocks, features, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
