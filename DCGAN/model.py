"""
    Discriminator and Generator class
"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, num_features):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False),
            # nn.Flatten(),
            # nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, num_features):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose2d(channels_noise, num_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(),
            # Second upsampling block
            nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(),
            # Third upsampling block
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(),
            # Fourth upsampling block
            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            # Fifth upsampling block
            nn.ConvTranspose2d(num_features, channels_img, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
