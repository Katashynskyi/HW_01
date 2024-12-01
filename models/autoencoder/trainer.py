import torch
import torch.optim as optim
import numpy as np
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    InceptionScore,
)
from pathlib import Path
from tqdm.auto import tqdm
import wandb
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import wandb
import torch.nn.functional as F

from common.utils import EarlyStopping
from common.data import VAEDataModule
from models.autoencoder.model import Autoencoder

class AutoencoderTrainingUtils:
    def __init__(self, device="cuda"):
        self.device = device

        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.inception = InceptionScore(normalize=True).to(device)

    def denormalize(self, x):
        return x * self.norm_std + self.norm_mean

    def normalize_for_display(self, x):
        return (x + 1) / 2

    def prepare_for_metrics(self, x_recon, x_target):
        x_recon = self.denormalize(x_recon)
        x_target = self.denormalize(x_target)

        x_recon = self.normalize_for_display(x_recon)
        x_target = self.normalize_for_display(x_target)

        x_recon = torch.clamp(x_recon, 0, 1)
        x_target = torch.clamp(x_target, 0, 1)

        return x_recon, x_target

    def compute_loss(self, recon_x, x):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        return recon_loss, {"loss": recon_loss.item()}

    @torch.no_grad()
    def compute_metrics(self, recon_x, x):
        recon_x, x = self.prepare_for_metrics(recon_x, x)

        ssim_val = self.ssim(recon_x, x)
        psnr_val = self.psnr(recon_x, x)

        self.inception.update(recon_x)
        is_mean, is_std = self.inception.compute()
        self.inception.reset()

        return {
            "ssim": ssim_val.item(),
            "psnr": psnr_val.item(),
            "inception_score_mean": is_mean.item(),
            "inception_score_std": is_std.item(),
        }

    def reset_metrics(self):
        self.ssim.reset()
        self.psnr.reset()
        self.inception.reset()


class AutoencoderTrainer:
    def __init__(self, model, train_loader, val_loader, utils, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.utils = utils

        self.config = {
            "epochs": 100,
            "initial_lr": 1e-3,
            "min_lr": 1e-6,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "patience": 6,
            "save_dir": "checkpoints",
            "log_dir": "runs",
            "project_name": "autoencoder-cifar",
        }
        if config:
            self.config.update(config)

        self.device = next(model.parameters()).device
        self.save_dir = Path(self.config["save_dir"])
        self.save_dir.mkdir(exist_ok=True)
        self.run_id = get_random_name(combo=[ADJECTIVES, NAMES], separator="")
        self.setup_logging()
        self.setup_training()

    def setup_logging(self):
        self.writer = SummaryWriter(log_dir=f"{self.config['log_dir']}/{self.run_id}")
        wandb.init(
            project=self.config["project_name"],
            name=self.run_id,
            id=self.run_id,
            config=self.config,
        )

    def setup_training(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["initial_lr"],
            weight_decay=self.config["weight_decay"],
        )

        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=5
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=self.config["min_lr"],
            verbose=True,
        )

        self.early_stopping = EarlyStopping(patience=self.config["patience"])
        self.scaler = GradScaler()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "run_id": self.run_id,
        }

        torch.save(checkpoint, self.save_dir / f"latest_{self.run_id}.pt")

        if is_best:
            torch.save(checkpoint, self.save_dir / f"best_{self.run_id}.pt")

    def log_reconstructions(self, epoch, images, recons):
        with torch.no_grad():
            images, recons = self.utils.prepare_for_metrics(images, recons)
            n = min(8, images.size(0))
            comparison = torch.cat([images[:n], recons[:n]])
            grid = vutils.make_grid(comparison, nrow=n)

            self.writer.add_image("reconstructions", grid, epoch)
            wandb.log({"reconstructions": wandb.Image(grid)}, step=epoch)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        val_metrics = []
        self.utils.reset_metrics()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        for images, _ in pbar:
            images = images.to(self.device)
            recons = self.model(images)
            loss, loss_dict = self.utils.compute_loss(recons, images)
            metrics = self.utils.compute_metrics(recons, images)
            metrics.update(loss_dict)

            pbar.set_postfix(
                {"loss": f"{metrics['loss']:.4f}", "ssim": f"{metrics['ssim']:.4f}"}
            )

            val_metrics.append(metrics)

        avg_metrics = {
            k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()
        }

        for name, value in avg_metrics.items():
            self.writer.add_scalar(f"val/{name}", value, epoch)
            wandb.log({f"val_{name}": value}, step=epoch)

        return avg_metrics

    def train_epoch(self, epoch):
        self.model.train()
        train_metrics = []
        self.utils.reset_metrics()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for images, _ in pbar:
            images = images.to(self.device)

            with autocast():
                recons = self.model(images)
                loss, loss_dict = self.utils.compute_loss(recons, images)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.config["grad_clip"] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            metrics = self.utils.compute_metrics(recons, images)
            metrics.update(loss_dict)

            pbar.set_postfix(
                {"loss": f"{metrics['loss']:.4f}", "ssim": f"{metrics['ssim']:.4f}"}
            )

            train_metrics.append(metrics)

        avg_metrics = {
            k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0].keys()
        }

        for name, value in avg_metrics.items():
            self.writer.add_scalar(f"train/{name}", value, epoch)
            wandb.log({f"train_{name}": value}, step=epoch)

        return avg_metrics

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.config["epochs"]):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            if epoch % 5 == 0:
                images, _ = next(iter(self.val_loader))
                images = images.to(self.device)
                with torch.no_grad():
                    recons = self.model(images)
                self.log_reconstructions(epoch, images, recons)

            if epoch < 5:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint(epoch, val_metrics, is_best=True)

            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics)

            if self.early_stopping(val_metrics["loss"]):
                print("Early stopping triggered!")
                break

            print(f"\nEpoch {epoch+1} Summary:")
            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"SSIM: {train_metrics['ssim']:.4f}, "
                f"PSNR: {train_metrics['psnr']:.4f}"
            )
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"SSIM: {val_metrics['ssim']:.4f}, "
                f"PSNR: {val_metrics['psnr']:.4f}"
            )
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        self.writer.close()
        wandb.finish()

        return {
            "run_id": self.run_id,
            "best_val_loss": best_val_loss,
            "early_stopped": self.early_stopping.early_stop,
        }




class AutoencoderEvaluator:
    def __init__(
        self,
        checkpoint_path: str,
        data_module: VAEDataModule,
        utils: AutoencoderTrainingUtils,
        device: str = "cuda",
        project_name: str = "autoencoder-cifar",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_module = data_module
        self.utils = utils
        self.device = device
        self.project_name = project_name

        self.run_id = get_random_name(combo=[ADJECTIVES, NAMES], separator="")
        self.load_model()
        self.setup_logging()

    def load_model(self):
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.model = Autoencoder()

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            self.original_config = checkpoint.get("config", {})
            self.original_run_id = checkpoint.get("run_id", "unknown")
        else:
            self.model.load_state_dict(checkpoint)
            self.original_config = {}
            self.original_run_id = "unknown"

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def setup_logging(self):
        self.writer = SummaryWriter(log_dir=f"eval_runs/{self.run_id}")
        wandb.init(
            project=self.project_name,
            name=f"eval_{self.run_id}",
            id=self.run_id,
            config={
                "original_run_id": self.original_run_id,
                "original_config": self.original_config,
                "checkpoint_path": str(self.checkpoint_path),
            },
        )

    @torch.no_grad()
    def evaluate_and_visualize(self, num_vis_samples: int = 10):
        test_metrics = []
        first_batch = None
        self.utils.reset_metrics()
        test_loader = self.data_module.test_dataloader()

        print("Running evaluation on test set...")
        pbar = tqdm(test_loader, desc="Evaluating")
        for batch_idx, (images, _) in enumerate(pbar):
            if batch_idx == 0:
                first_batch = images[:num_vis_samples].to(self.device)

            images = images.to(self.device)
            recons = self.model(images)
            loss, loss_dict = self.utils.compute_loss(recons, images)
            metrics = self.utils.compute_metrics(recons, images)
            metrics.update(loss_dict)

            pbar.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "ssim": f"{metrics['ssim']:.4f}",
                    "is_mean": f"{metrics['inception_score_mean']:.2f}",
                }
            )

            test_metrics.append(metrics)

        avg_metrics = {
            k: np.mean([m[k] for m in test_metrics]) for k in test_metrics[0].keys()
        }

        print("\nTest Set Metrics:")
        for name, value in avg_metrics.items():
            print(f"{name}: {value:.4f}")
            self.writer.add_scalar(f"test/{name}", value, 0)
            wandb.log({f"test_{name}": value})

        if first_batch is not None:
            self.visualize_reconstructions(first_batch)

        self.writer.close()
        wandb.finish()

        return avg_metrics

    def visualize_reconstructions(self, images):
        self.model.eval()
        with torch.no_grad():
            recons = self.model(images)

            images_denorm = self.utils.denormalize(images)
            recons_denorm = self.utils.denormalize(recons)

            images_display = self.utils.normalize_for_display(images_denorm)
            recons_display = self.utils.normalize_for_display(recons_denorm)

            images_display = torch.clamp(images_display, 0, 1)
            recons_display = torch.clamp(recons_display, 0, 1)

            comparison = torch.cat([images_display, recons_display])
            grid = vutils.make_grid(comparison, nrow=len(images))

            save_dir = Path("eval_results") / self.run_id
            save_dir.mkdir(parents=True, exist_ok=True)
            vutils.save_image(grid, save_dir / "reconstructions.png")

            self.writer.add_image("test_reconstructions", grid, 0)
            wandb.log(
                {
                    "test_reconstructions": wandb.Image(grid),
                    "test_samples_original": [
                        wandb.Image(img) for img in images_display
                    ],
                    "test_samples_reconstructed": [
                        wandb.Image(img) for img in recons_display
                    ],
                }
            )
