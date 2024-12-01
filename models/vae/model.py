import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # 32x32x3 -> 16x16x64
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # Flatten -> latent parameters
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)

        # Latent -> initial decoder shape
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)

        # Decoder
        self.decoder = nn.Sequential(
            # 4x4x256 -> 8x8x128
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 8x8x128 -> 16x16x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 16x16x64 -> 32x32x3
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            # Final activation - bounded output for stable training
            nn.Tanh(),
        )

    def encode(self, x):
        # Encode
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        # Get latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    def decode(self, z):
        # Reconstruct initial decoder shape
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)

        # Decode
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
