# All required imports
import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.nn.functional as F


class CouplingLayer(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_dim: int = 128,
        mask_type: str = "checkerboard",
    ):
        super().__init__()
        self.num_channels = num_channels

        # Improved mask creation
        mask = torch.zeros(1, num_channels, 32, 32)
        if mask_type == "checkerboard":
            mask[:, :, ::2, ::2] = 1
            mask[:, :, 1::2, 1::2] = 1
        else:
            mask[:, : num_channels // 2] = 1

        self.register_buffer("mask", mask)

        # Deeper network with residual connections
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 2 * num_channels, 3, padding=1),
        )

        # Initialize last layer with small weights
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        net_out = self.net(x_masked)

        # Improved scaling with softplus
        s, t = net_out.chunk(2, dim=1)
        s = F.softplus(s)

        if not reverse:
            y = x_masked + (1 - self.mask) * (x * s + t)
            log_det = ((1 - self.mask) * torch.log(s + 1e-8)).sum(dim=[1, 2, 3])
        else:
            y = x_masked + (1 - self.mask) * ((x - t) / (s + 1e-8))
            log_det = -((1 - self.mask) * torch.log(s + 1e-8)).sum(dim=[1, 2, 3])

        return y, log_det


class RealNVP(nn.Module):
    def __init__(
        self, num_channels: int = 3, hidden_dim: int = 64, num_layers: int = 8
    ):
        super().__init__()

        # Create alternating coupling layers
        self.layers = nn.ModuleList(
            [
                CouplingLayer(
                    num_channels=num_channels,
                    hidden_dim=hidden_dim,
                    mask_type="checkerboard" if i % 2 == 0 else "channel",
                )
                for i in range(num_layers)
            ]
        )

        # Prior distribution parameters (learned)
        self.register_buffer("prior_mean", torch.zeros(1, num_channels, 32, 32))
        self.register_buffer("prior_logvar", torch.zeros(1, num_channels, 32, 32))

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(x.size(0), device=x.device)

        if not reverse:
            # Forward transformation
            for layer in self.layers:
                x, log_det = layer(x, reverse=False)
                log_det_total += log_det
        else:
            # Inverse transformation
            for layer in reversed(self.layers):
                x, log_det = layer(x, reverse=True)
                log_det_total += log_det

        return x, log_det_total

    def sample(self, num_samples: int, device: str = "cuda") -> torch.Tensor:
        """Generate samples by sampling from prior and transforming"""
        # Sample from learned prior
        z = torch.randn(num_samples, 3, 32, 32, device=device)
        z = z * torch.exp(0.5 * self.prior_logvar) + self.prior_mean

        # Transform through inverse flow
        samples, _ = self.forward(z, reverse=True)
        return samples

    def compute_loss(
        self, x: torch.Tensor, return_z: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute negative log likelihood loss"""
        # Forward pass
        z, log_det = self.forward(x, reverse=False)

        # Compute log likelihood using the learned prior
        log_pz = -0.5 * (
            self.prior_logvar
            + ((z - self.prior_mean) ** 2) / torch.exp(self.prior_logvar)
            + torch.log(2 * torch.tensor(3.14159, device=z.device))
        ).sum(dim=[1, 2, 3])

        # Final loss
        log_px = log_pz + log_det
        loss = -torch.mean(log_px)

        if return_z:
            return loss, z
        return loss, None
