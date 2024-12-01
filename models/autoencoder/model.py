import torch
import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.fc_encoder = nn.Linear(128 * 2 * 2, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 128 * 2 * 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(
                16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_encoder(x)

    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(-1, 128, 2, 2)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
