# All required imports
import torch
import torchvision.utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES
from torch.utils.tensorboard import SummaryWriter
import wandb
from pathlib import Path
from typing import Dict

# class EarlyStopping:
#     def __init__(self, patience=6, min_delta=0.0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0
#         return self.early_stop


class NFEarlyStopping:
    """Early stopping implementation"""

    def __init__(self, patience: int = 25, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.mode == "min" and val_loss > self.best_loss - self.min_delta) or (
            self.mode == "max" and val_loss < self.best_loss + self.min_delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class GANEarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience=25, min_delta=0.0, monitor="fid"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, metrics):
        score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


def log_images(writer, real_images, fake_images, epoch):
    """Log real and generated images to tensorboard and wandb"""
    # Create image grid
    n_images = min(real_images.size(0), 8)
    comparison = torch.cat([real_images[:n_images], fake_images[:n_images]])
    grid = vutils.make_grid(comparison, nrow=n_images, normalize=True)

    # Log to tensorboard
    writer.add_image("real_vs_fake", grid, epoch)

    # Log to wandb
    wandb.log({"real_vs_fake": wandb.Image(grid)}, step=epoch)


def init_logging(save_dir="checkpoints", log_dir="runs"):
    """Initialize logging directories and wandb with unique name"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Generate unique run name
    run_id = get_random_name(combo=[ADJECTIVES, NAMES], separator="")

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=f"{log_dir}/{run_id}")

    # Initialize wandb
    wandb.init(
        project="gan-cifar10",
        name=run_id,
        id=run_id,
        config={"run_id": run_id, "save_dir": str(save_dir), "log_dir": log_dir},
    )

    return run_id, writer


def log_metrics(writer, metrics, epoch):
    """Log metrics to both tensorboard and wandb without 'train/' prefix"""
    # Unpack inception score mean and std
    inception_mean, inception_std = metrics["inception_score"]

    # Create metrics dict with modified names
    wandb_metrics = {
        "d_loss": metrics["d_loss"],
        "g_loss": metrics["g_loss"],
        "fid": metrics["fid"],
        "inception_score_mean": inception_mean,
        "inception_score_std": inception_std,
        "kid_mean": metrics["kid_mean"],
        "kid_std": metrics["kid_std"],
        "ssim": metrics["ssim"],
        "psnr": metrics["psnr"],
    }

    # Log to wandb without prefix
    wandb.log(wandb_metrics, step=epoch)
    # Log to tensorboard without prefix
    for name, value in wandb_metrics.items():
        writer.add_scalar(name, value, epoch)


class FlowTrainingUtils:
    def __init__(self, device="cuda"):
        self.device = device
        self.setup_metrics()
        self.setup_ranges()

    def setup_metrics(self):
        """Initialize all metrics"""
        # Initialize all metrics on the correct device
        self.metrics = {
            "fid": FrechetInceptionDistance(feature=64, normalize=True).to(self.device),
            "inception_score": InceptionScore(normalize=True).to(self.device),
            "kid": KernelInceptionDistance(subset_size=50, normalize=True).to(
                self.device
            ),
            "ssim": StructuralSimilarityIndexMeasure().to(self.device),
            "psnr": PeakSignalNoiseRatio().to(self.device),
        }

    def setup_ranges(self):
        """Setup data normalization parameters"""
        self.norm_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)

    def reset_metrics(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from normalized space to image space"""
        return (x * self.norm_std + self.norm_mean).clamp(0, 1)

    def prepare_for_metrics(self, images: torch.Tensor) -> torch.Tensor:
        """Prepare images for metric computation"""
        images = self.denormalize(images)
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        return images

    @torch.no_grad()
    def update_metrics(self, real_images: torch.Tensor, generated_images: torch.Tensor):
        """Update all metrics with a batch of images"""
        real_uint8 = self.prepare_for_metrics(real_images)
        fake_uint8 = self.prepare_for_metrics(generated_images)

        real_float = self.denormalize(real_images)
        fake_float = self.denormalize(generated_images)

        # Update metrics
        self.metrics["fid"].update(real_uint8, real=True)
        self.metrics["fid"].update(fake_uint8, real=False)

        self.metrics["inception_score"].update(fake_uint8)

        self.metrics["kid"].update(real_uint8, real=True)
        self.metrics["kid"].update(fake_uint8, real=False)

        self.metrics["ssim"].update(fake_float, real_float)
        self.metrics["psnr"].update(fake_float, real_float)

    @torch.no_grad()
    def compute_metrics(self, loss: float) -> Dict[str, float]:
        """Compute and return all metrics"""
        try:
            inception_mean, inception_std = self.metrics["inception_score"].compute()
            kid_mean, kid_std = self.metrics["kid"].compute()

            metrics = {
                "loss": loss,
                "fid": self.metrics["fid"].compute().item(),
                "inception_score": inception_mean.item(),
                "inception_std": inception_std.item(),
                "kid": kid_mean.item(),
                "kid_std": kid_std.item(),
                "ssim": self.metrics["ssim"].compute().item(),
                "psnr": self.metrics["psnr"].compute().item(),
            }
        except Exception as e:
            print(f"Error computing metrics: {str(e)}")
            metrics = {"loss": loss}

        # Reset metrics after computation
        self.reset_metrics()
        return metrics

    def log_metrics(self, writer: SummaryWriter, metrics: Dict[str, float], step: int):
        """Log metrics to both tensorboard and wandb"""
        for name, value in metrics.items():
            # Log to tensorboard
            writer.add_scalar(name, value, step)
            # Log to wandb
            wandb.log({name: value}, step=step)

    def log_images(
        self,
        writer: SummaryWriter,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        step: int,
        max_images: int = 16,
    ):
        """Log images to both tensorboard and wandb"""
        with torch.no_grad():
            # Select subset of images
            real_images = real_images[:max_images]
            generated_images = generated_images[:max_images]

            # Denormalize images
            real_images = self.denormalize(real_images)
            generated_images = self.denormalize(generated_images)

            # Create comparison grid
            comparison = torch.cat([real_images, generated_images])
            grid = vutils.make_grid(comparison, nrow=max_images, normalize=True)

            # Log to tensorboard
            writer.add_image("real_vs_generated", grid, step)

            # Log to wandb
            wandb.log({"real_vs_generated": wandb.Image(grid)}, step=step)


class EarlyStopping:# AE & NormFlow
    def __init__(self, patience: int = 25, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.mode == "min" and val_loss > self.best_loss - self.min_delta) or (
            self.mode == "max" and val_loss < self.best_loss + self.min_delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop

