import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, h, w = x.size()
        q = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h * w)
        v = self.value(x).view(batch, -1, h * w)

        attention = F.softmax(torch.bmm(q, k), dim=2)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, c, h, w)
        return self.gamma * out + x


class Generator(nn.Module):
    def __init__(self, latent_dim=100, feature_maps=128):
        super().__init__()

        self.latent_dim = latent_dim
        self.feature_maps = feature_maps

        # Initial dense layer
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * feature_maps * 8), nn.GELU()
        )

        self.layers = nn.ModuleList(
            [
                # 4x4 -> 8x8
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(feature_maps * 8, feature_maps * 4, 3, 1, 1),
                    nn.BatchNorm2d(feature_maps * 4),
                    nn.GELU(),
                    ResBlock(feature_maps * 4),
                ),
                # 8x8 -> 16x16
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(feature_maps * 4, feature_maps * 2, 3, 1, 1),
                    nn.BatchNorm2d(feature_maps * 2),
                    nn.GELU(),
                    ResBlock(feature_maps * 2),
                    SelfAttention(feature_maps * 2),
                ),
                # 16x16 -> 32x32
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(feature_maps * 2, feature_maps, 3, 1, 1),
                    nn.BatchNorm2d(feature_maps),
                    nn.GELU(),
                    ResBlock(feature_maps),
                    nn.Conv2d(feature_maps, 3, 3, 1, 1),
                    nn.Tanh(),
                ),
            ]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # Initial projection and reshape
        x = self.initial(z)
        x = x.view(-1, self.feature_maps * 8, 4, 4)

        # Progressive generation
        for layer in self.layers:
            x = layer(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, feature_maps: int = 64):
        super().__init__()

        self.main = nn.Sequential(
            # 32x32x3 -> 16x16xfeature_maps
            nn.Conv2d(3, feature_maps, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 16x16xfeature_maps -> 8x8x(feature_maps*2)
            nn.Conv2d(feature_maps, feature_maps * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2),
            # 8x8x(feature_maps*2) -> 4x4x(feature_maps*4)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2),
        )

        # Output layer - removed sigmoid activation since using BCE with logits
        self.fc = nn.Linear(4 * 4 * feature_maps * 4, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using normal distribution"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        features = self.main(x)
        features = torch.flatten(features, start_dim=1)

        # Classification (without sigmoid)
        return self.fc(features)


class GAN(nn.Module):
    def __init__(
        self, latent_dim: int = 100, feature_maps: int = 64, device: str = "cuda"
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.device = device

        # Initialize Generator and Discriminator
        self.generator = Generator(latent_dim, feature_maps)
        self.discriminator = Discriminator(feature_maps)

        # Move models to device
        self.to(device)

    def generate_latent(self, batch_size: int) -> torch.Tensor:
        """Generate random latent vectors"""
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def generate_images(self, batch_size: int) -> torch.Tensor:
        """Generate images from random latent vectors"""
        z = self.generate_latent(batch_size)
        return self.generator(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator"""
        return self.generator(z)
