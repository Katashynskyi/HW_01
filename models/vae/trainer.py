import torch
import torch.optim as optim
from pathlib import Path
from tqdm.auto import tqdm
import wandb
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torch.nn.functional as F
from piq import ssim as piq_ssim, psnr as piq_psnr
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from common.utils import EarlyStopping
from common.logging import ReconstructionLogger
from models.vae.model import VAE
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES


class VAETrainingUtils:
    """Training utilities for VAE"""

    def __init__(self, device="cuda"):
        self.device = device
        # ImageNet normalization constants
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def denormalize(self, x):
        """Convert from normalized space back to [0, 1] range"""
        return x * self.norm_std + self.norm_mean

    def normalize_for_display(self, x):
        """Convert tanh output [-1, 1] to [0, 1] range"""
        return (x + 1) / 2

    def prepare_for_metrics(self, x_recon, x_target):
        """Prepare images for metric calculation"""
        # First denormalize ImageNet normalization
        x_recon = self.denormalize(x_recon)
        x_target = self.denormalize(x_target)

        # Then convert from tanh range to [0, 1]
        x_recon = self.normalize_for_display(x_recon)
        x_target = self.normalize_for_display(x_target)

        # Clamp to ensure we're in [0, 1]
        x_recon = torch.clamp(x_recon, 0, 1)
        x_target = torch.clamp(x_target, 0, 1)

        return x_recon, x_target

    def compute_vae_loss(self, recon_x, x, mu, logvar, kl_weight=1.0):
        """
        Compute VAE loss with proper KL weighting
        Returns:
            total_loss: weighted sum of reconstruction and KL loss
            loss_dict: dictionary containing individual loss components
        """
        # Reconstruction loss (in normalized space)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def compute_metrics(self, recon_x, x):
        """
        Compute SSIM and PSNR metrics
        Returns:
            dict: containing SSIM and PSNR values
        """
        # Prepare images for metric calculation
        recon_x, x = self.prepare_for_metrics(recon_x, x)

        with torch.no_grad():
            ssim_val = piq_ssim(recon_x, x, data_range=1.0)
            psnr_val = piq_psnr(recon_x, x, data_range=1.0)

        return {"ssim": ssim_val.item(), "psnr": psnr_val.item()}

    def save_sample_images(self, images, recons, run_id, epoch, save_dir="samples"):
        """Save sample images with unique name"""
        save_dir = Path(save_dir) / run_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare images for saving
        images, recons = self.prepare_for_metrics(images, recons)

        # Create comparison grid
        comparison = torch.cat([images, recons])
        save_image(
            comparison, save_dir / f"{run_id}_epoch_{epoch}.png", nrow=len(images)
        )

    def init_run_id(self):
        """Generate a unique run ID for training or evaluation"""
        return get_random_name(combo=[ADJECTIVES, NAMES], separator="")


class VAETrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        utils,  # VAETrainingUtils instance
        config=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.utils = utils

        # Default configuration
        self.config = {
            "epochs": 200,
            "initial_lr": 1e-3,
            "min_lr": 1e-6,
            "weight_decay": 1e-4,
            "kl_weight": 0.1,
            "grad_clip": 1.0,
            "patience": 25,
            "save_dir": "checkpoints",
            "log_dir": "runs",
            "project_name": "vae-cifar",
        }
        if config:
            self.config.update(config)

        # Initialize device
        self.device = next(model.parameters()).device

        # Setup logging with unique name
        self.save_dir = Path(self.config["save_dir"])
        self.save_dir.mkdir(exist_ok=True)
        self.run_id = get_random_name(combo=[ADJECTIVES, NAMES], separator="")
        self.writer = self.init_logging()

        # Initialize reconstruction logger
        self.recon_logger = ReconstructionLogger(run_id=self.run_id)

        # Setup training components
        self.setup_training()

    def init_logging(self):
        """Initialize wandb and tensorboard with unique run name"""
        writer = SummaryWriter(log_dir=f"{self.config['log_dir']}/{self.run_id}")

        # Initialize wandb with unique name
        wandb.init(
            project=self.config["project_name"],
            name=self.run_id,
            id=self.run_id,
            config=self.config,
        )
        return writer

    def setup_training(self):
        """Initialize optimizer, schedulers, and training utilities"""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["initial_lr"],
            weight_decay=self.config["weight_decay"],
        )

        # Learning rate schedulers
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=5
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=self.config["min_lr"],
            verbose=True,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=self.config["patience"])

        # Gradient scaler for mixed precision
        self.scaler = GradScaler()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint with unique name"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "run_id": self.run_id,
        }

        # Save latest checkpoint with unique name
        torch.save(checkpoint, self.save_dir / f"{self.run_id}_latest.pt")

        # Save best model if specified
        if is_best:
            torch.save(checkpoint, self.save_dir / f"{self.run_id}_best.pt")

    def train_epoch(self, epoch):
        """Run training loop for one epoch"""
        self.model.train()
        train_metrics = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for images, _ in pbar:
            images = images.to(self.device)

            # Forward pass with mixed precision
            with autocast():
                recons, mu, logvar = self.model(images)
                loss, loss_dict = self.utils.compute_vae_loss(
                    recons, images, mu, logvar, kl_weight=self.config["kl_weight"]
                )

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Clip gradients
            if self.config["grad_clip"] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Calculate metrics
            metrics = self.utils.compute_metrics(recons, images)
            metrics.update(loss_dict)

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{metrics['loss']:.4f}", "ssim": f"{metrics['ssim']:.4f}"}
            )

            train_metrics.append(metrics)

        # Average metrics
        avg_metrics = {
            k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0].keys()
        }

        # Log training metrics
        for name, value in avg_metrics.items():
            self.writer.add_scalar(f"train/{name}", value, epoch)
            wandb.log({f"train_{name}": value}, step=epoch)
        return avg_metrics

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation loop"""
        self.model.eval()
        val_metrics = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        for images, _ in pbar:
            images = images.to(self.device)

            # Forward pass
            recons, mu, logvar = self.model(images)

            # Calculate losses
            loss, loss_dict = self.utils.compute_vae_loss(
                recons, images, mu, logvar, kl_weight=self.config["kl_weight"]
            )

            # Calculate metrics
            metrics = self.utils.compute_metrics(recons, images)
            metrics.update(loss_dict)

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{metrics['loss']:.4f}", "ssim": f"{metrics['ssim']:.4f}"}
            )

            val_metrics.append(metrics)

        # Average metrics
        avg_metrics = {
            k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()
        }

        # Log validation metrics
        for name, value in avg_metrics.items():
            self.writer.add_scalar(f"val/{name}", value, epoch)
            wandb.log({f"val_{name}": value}, step=epoch)

        return avg_metrics

    def log_reconstructions(self, epoch):
        """Log image reconstructions for all subsets"""
        data_loaders = {"train": self.train_loader, "val": self.val_loader}

        save_paths = self.recon_logger.log_all_subsets(
            self.model, data_loaders, self.utils, self.device, epoch, self.writer
        )

        return save_paths

    def train(self):
        """Main training loop"""
        best_val_loss = float("inf")
        print(f"Starting training with run ID: {self.run_id}")

        for epoch in range(self.config["epochs"]):
            # Training phase
            train_metrics = self.train_epoch(epoch)

            # Validation phase
            val_metrics = self.validate(epoch)

            # Log reconstructions for both train and val sets
            if epoch % 5 == 0:
                save_paths = self.log_reconstructions(epoch)
                print(f"\nSaved reconstructions for epoch {epoch}:")
                for subset, path in save_paths.items():
                    print(f"{subset}: {path}")

            # Update learning rate
            if epoch < 5:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step(val_metrics["loss"])

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint(epoch, val_metrics, is_best=True)

            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics)

            # Early stopping check
            if self.early_stopping(val_metrics["loss"]):
                print("Early stopping triggered!")
                break

            # Log summary
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

        # Cleanup
        self.writer.close()
        wandb.finish()

        return {
            "run_id": self.run_id,
            "best_val_loss": best_val_loss,
            "early_stopped": self.early_stopping.early_stop,
        }


class VAEEvaluator:
    def __init__(
        self,
        model_or_path,  # Can be either a model instance or path to checkpoint
        data_module,
        utils,
        device="cuda",
        project_name="vae-cifar",
    ):
        self.device = device
        self.data_module = data_module
        self.utils = utils
        self.project_name = project_name

        # Generate unique run ID for evaluation
        self.run_id = get_random_name(combo=[ADJECTIVES, NAMES], separator="")

        # Initialize model based on input type
        self.model = self._initialize_model(model_or_path)

        # Setup logging
        self.setup_logging()

        # Initialize reconstruction logger
        self.recon_logger = ReconstructionLogger(run_id=self.run_id)

    def _initialize_model(self, model_or_path):
        """Initialize model either from checkpoint or direct instance"""
        if isinstance(model_or_path, torch.nn.Module):
            print("Using provided model instance for evaluation")
            model = model_or_path
            self.original_config = getattr(model, "config", {})
            self.original_run_id = getattr(model, "run_id", "unknown")
            self.checkpoint_source = "direct_model"
        else:
            print(f"Loading model from checkpoint: {model_or_path}")
            checkpoint = torch.load(model_or_path, map_location=self.device)
            model = VAE().to(self.device)

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                self.original_config = checkpoint.get("config", {})
                self.original_run_id = checkpoint.get("run_id", "unknown")
            else:
                model.load_state_dict(checkpoint)
                self.original_config = {}
                self.original_run_id = "unknown"

            self.checkpoint_source = str(model_or_path)

        model = model.to(self.device)
        model.eval()
        return model

    def setup_logging(self):
        """Initialize wandb and tensorboard logging"""
        log_dir = Path("eval_runs") / self.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

        wandb.init(
            project=self.project_name,
            name=f"eval_{self.run_id}",
            id=self.run_id,
            config={
                "original_run_id": self.original_run_id,
                "original_config": self.original_config,
                "checkpoint_source": self.checkpoint_source,
            },
        )

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation on test set"""
        self.model.eval()
        test_metrics = []
        test_loader = self.data_module.test_dataloader()

        print(f"\nStarting evaluation (Run ID: {self.run_id})")
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(self.device)

            # Forward pass
            recons, mu, logvar = self.model(images)

            # Calculate metrics
            loss, loss_dict = self.utils.compute_vae_loss(
                recons,
                images,
                mu,
                logvar,
                kl_weight=self.original_config.get("kl_weight", 0.1),
            )
            metrics = self.utils.compute_metrics(recons, images)
            metrics.update(loss_dict)

            test_metrics.append(metrics)

            # Save first batch reconstructions
            if batch_idx == 0:
                data_loaders = {"test": test_loader}
                save_paths = self.recon_logger.log_all_subsets(
                    self.model,
                    data_loaders,
                    self.utils,
                    self.device,
                    epoch=0,  # Single evaluation run
                    writer=self.writer,
                )
                print("\nSaved test reconstructions:")
                for subset, path in save_paths.items():
                    print(f"{subset}: {path}")

        # Calculate average metrics
        avg_metrics = {
            k: np.mean([m[k] for m in test_metrics]) for k in test_metrics[0].keys()
        }

        # Log metrics
        for name, value in avg_metrics.items():
            self.writer.add_scalar(f"test/{name}", value, 0)
            wandb.log({f"test_{name}": value})

        return avg_metrics

    def cleanup(self):
        """Clean up logging"""
        self.writer.close()
        wandb.finish()
