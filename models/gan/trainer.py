import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    InceptionScore,
    FrechetInceptionDistance,
    KernelInceptionDistance,
)
from pathlib import Path
from tqdm.auto import tqdm
import wandb
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES
from models.gan.model import GAN
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from common.utils import GANEarlyStopping, log_metrics
from common.logging import GANReconstructionLogger


class GANTrainingUtils:
    """Training utilities for GAN"""
    def __init__(self, device="cuda"):
        self.device = device

        # Initialize metrics using torchmetrics
        self.fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)
        self.inception_score = InceptionScore(normalize=True).to(device)
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.psnr = PeakSignalNoiseRatio().to(device)

        # For converting between different value ranges
        self.register_ranges()

    def register_ranges(self):
        """Register tensors for converting between different value ranges"""
        self.norm_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.norm_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)

    def denormalize(self, x):
        """Convert from normalized space [-1, 1] to [0, 1] range"""
        return (x * self.norm_std + self.norm_mean).clamp(0, 1)

    def prepare_for_metrics(self, images):
        """Convert images to uint8 format required by metrics"""
        # First denormalize to [0, 1]
        images = self.denormalize(images)

        # Convert to range [0, 255] and uint8
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        return images

    def compute_adversarial_loss(
        self,
        discriminator: nn.Module,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        device: str,
    ) -> tuple:
        """Compute GAN losses for both generator and discriminator using BCE with logits"""
        batch_size = real_batch.size(0)

        # Labels for real and fake data
        real_label = torch.ones(batch_size, 1, device=device)
        fake_label = torch.zeros(batch_size, 1, device=device)

        # Discriminator forward pass on real data
        real_output = discriminator(real_batch)
        d_loss_real = F.binary_cross_entropy_with_logits(real_output, real_label)

        # Discriminator forward pass on fake data
        fake_output = discriminator(fake_batch.detach())
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_output, fake_label)

        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2

        # Generator loss
        g_output = discriminator(fake_batch)
        g_loss = F.binary_cross_entropy_with_logits(g_output, real_label)

        return d_loss, g_loss

    def compute_metrics(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        d_loss: float,
        g_loss: float,
    ) -> dict:
        """
        Compute all metrics for GAN evaluation
        Returns dictionary with all metrics
        """
        with torch.no_grad():
            # Convert images to uint8 format
            real_uint8 = self.prepare_for_metrics(real_images)
            fake_uint8 = self.prepare_for_metrics(fake_images)

            # Keep float versions for SSIM and PSNR
            real_float = self.denormalize(real_images)
            fake_float = self.denormalize(fake_images)

            # Update FID
            self.fid.update(real_uint8, real=True)
            self.fid.update(fake_uint8, real=False)

            # Update Inception Score (only on fake images)
            self.inception_score.update(fake_uint8)

            # Update KID
            self.kid.update(real_uint8, real=True)
            self.kid.update(fake_uint8, real=False)

            # Compute SSIM and PSNR (using float tensors)
            ssim_val = self.ssim(fake_float, real_float)
            psnr_val = self.psnr(fake_float, real_float)

            # Get inception score mean and std
            inception_mean, inception_std = self.inception_score.compute()

            # Get KID mean and std
            kid_mean, kid_std = self.kid.compute()

            # Gather all metrics
            metrics = {
                "fid": self.fid.compute(),
                "inception_score": (
                    inception_mean,
                    inception_std,
                ),  # Return both mean and std
                "kid_mean": kid_mean,  # Split KID into mean and std
                "kid_std": kid_std,
                "ssim": ssim_val,
                "psnr": psnr_val,
                "d_loss": d_loss,
                "g_loss": g_loss,
            }

            # Reset metrics for next computation
            self.fid.reset()
            self.inception_score.reset()
            self.kid.reset()

            return metrics


# class GANTrainer:
#     def __init__(
#         self, model, train_loader, utils, config=None  # GANTrainingUtils instance
#     ):
#         self.model = model
#         self.train_loader = train_loader
#         self.utils = utils

#         # Default configuration
#         self.config = {
#             "epochs": 200,
#             "g_lr": 4e-4,
#             "d_lr": 1e-4,
#             "beta1": 0.5,
#             "beta2": 0.999,
#             "grad_clip": 1.0,
#             "n_critic": 2,  # Number of discriminator updates per generator update
#             "patience": 25,
#             "save_dir": "checkpoints",
#             "log_dir": "runs",
#             "project_name": "gan-cifar10",
#         }
#         if config:
#             self.config.update(config)

#         # Initialize device
#         self.device = next(model.parameters()).device

#         # Setup logging
#         self.save_dir = Path(self.config["save_dir"])
#         self.save_dir.mkdir(exist_ok=True)
#         self.run_id, self.writer = init_logging(
#             self.config["save_dir"], self.config["log_dir"]
#         )

#         # Setup training components
#         self.setup_training()

#     def setup_training(self):
#         """Initialize optimizers, schedulers, and training utilities"""
#         # Optimizers
#         self.g_optimizer = optim.Adam(
#             self.model.generator.parameters(),
#             lr=self.config["g_lr"],
#             betas=(self.config["beta1"], self.config["beta2"]),
#         )

#         self.d_optimizer = optim.Adam(
#             self.model.discriminator.parameters(),
#             lr=self.config["d_lr"],
#             betas=(self.config["beta1"], self.config["beta2"]),
#         )

#         # Learning rate schedulers
#         self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.g_optimizer, mode="min", factor=0.5, patience=10, verbose=True
#         )

#         self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.d_optimizer, mode="min", factor=0.5, patience=10, verbose=True
#         )

#         # Early stopping (monitoring FID score)
#         self.early_stopping = GANEarlyStopping(
#             patience=self.config["patience"], monitor="fid"
#         )

#         # Gradient scaler for mixed precision
#         self.g_scaler = GradScaler()
#         self.d_scaler = GradScaler()

#     def save_checkpoint(self, epoch, metrics, is_best=False):
#         """Save model checkpoint"""
#         checkpoint = {
#             "epoch": epoch,
#             "generator_state_dict": self.model.generator.state_dict(),
#             "discriminator_state_dict": self.model.discriminator.state_dict(),
#             "g_optimizer_state_dict": self.g_optimizer.state_dict(),
#             "d_optimizer_state_dict": self.d_optimizer.state_dict(),
#             "metrics": metrics,
#             "config": self.config,
#             "run_id": self.run_id,
#         }

#         # Save latest checkpoint
#         torch.save(checkpoint, self.save_dir / f"latest_{self.run_id}.pt")

#         # Save best model if specified
#         if is_best:
#             torch.save(checkpoint, self.save_dir / f"best_{self.run_id}.pt")

#     def train_epoch(self, epoch):
#         """Run training loop for one epoch"""
#         self.model.train()
#         train_metrics = []

#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
#         for batch_idx, real_images in enumerate(pbar):
#             batch_size = real_images.size(0)
#             real_images = real_images.to(self.device)

#             # Train Discriminator
#             self.d_optimizer.zero_grad()

#             with autocast():
#                 # Generate fake images
#                 z = self.model.generate_latent(batch_size)
#                 fake_images = self.model.generator(z)

#                 # Calculate discriminator loss
#                 d_loss, g_loss = self.utils.compute_adversarial_loss(
#                     self.model.discriminator, real_images, fake_images, self.device
#                 )

#             # Backward pass for discriminator
#             self.d_scaler.scale(d_loss).backward()

#             # Clip discriminator gradients
#             if self.config["grad_clip"] > 0:
#                 self.d_scaler.unscale_(self.d_optimizer)
#                 torch.nn.utils.clip_grad_norm_(
#                     self.model.discriminator.parameters(), self.config["grad_clip"]
#                 )

#             self.d_scaler.step(self.d_optimizer)
#             self.d_scaler.update()

#             # Train Generator every n_critic steps
#             if batch_idx % self.config["n_critic"] == 0:
#                 self.g_optimizer.zero_grad()

#                 with autocast():
#                     # Generate new fake images
#                     z = self.model.generate_latent(batch_size)
#                     fake_images = self.model.generator(z)

#                     # Calculate generator loss
#                     _, g_loss = self.utils.compute_adversarial_loss(
#                         self.model.discriminator, real_images, fake_images, self.device
#                     )

#                 # Backward pass for generator
#                 self.g_scaler.scale(g_loss).backward()

#                 # Clip generator gradients
#                 if self.config["grad_clip"] > 0:
#                     self.g_scaler.unscale_(self.g_optimizer)
#                     torch.nn.utils.clip_grad_norm_(
#                         self.model.generator.parameters(), self.config["grad_clip"]
#                     )

#                 self.g_scaler.step(self.g_optimizer)
#                 self.g_scaler.update()

#                 # Calculate metrics
#                 metrics = self.utils.compute_metrics(
#                     real_images, fake_images, d_loss.item(), g_loss.item()
#                 )

#                 # Update progress bar
#                 pbar.set_postfix(
#                     {
#                         "d_loss": f"{d_loss.item():.4f}",
#                         "g_loss": f"{g_loss.item():.4f}",
#                         "fid": f"{metrics['fid']:.4f}",
#                     }
#                 )

#                 train_metrics.append(metrics)

#         # Average metrics properly handling inception score tuple
#         avg_metrics = {}
#         num_metrics = len(train_metrics)

#         for k in train_metrics[0].keys():
#             if k == "inception_score":
#                 # Handle inception score mean and std separately
#                 means = [m[k][0] for m in train_metrics]
#                 stds = [m[k][1] for m in train_metrics]
#                 avg_metrics[k] = (sum(means) / num_metrics, sum(stds) / num_metrics)
#             else:
#                 # Regular averaging for other metrics
#                 avg_metrics[k] = sum(m[k] for m in train_metrics) / num_metrics

#         # Log metrics and images
#         log_metrics(self.writer, avg_metrics, epoch)
#         log_images(
#             self.writer,
#             self.utils.denormalize(real_images[:8]),
#             self.utils.denormalize(fake_images[:8]),
#             epoch,
#         )

#         return avg_metrics

#     def train(self):
#         """Main training loop"""
#         best_fid = float("inf")

#         try:
#             for epoch in range(self.config["epochs"]):
#                 # Training phase
#                 metrics = self.train_epoch(epoch)

#                 # Update learning rates
#                 self.g_scheduler.step(metrics["g_loss"])
#                 self.d_scheduler.step(metrics["d_loss"])

#                 # Save best model
#                 if metrics["fid"] < best_fid:
#                     best_fid = metrics["fid"]
#                     self.save_checkpoint(epoch, metrics, is_best=True)

#                 # Regular checkpoint
#                 if epoch % 10 == 0:
#                     self.save_checkpoint(epoch, metrics)

#                 # Early stopping check
#                 if self.early_stopping(metrics):
#                     print("Early stopping triggered!")
#                     break

#                 # Get inception score mean and std
#                 is_mean, is_std = metrics["inception_score"]

#                 # Log summary
#                 print(f"\nEpoch {epoch+1} Summary:")
#                 print(f"FID Score: {metrics['fid']:.4f}")
#                 print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
#                 print(f"KID kid_mean Score: {metrics['kid_mean']:.4f}")
#                 print(f"KID kid_std Score: {metrics['kid_std']:.4f}")
#                 print(f"D Loss: {metrics['d_loss']:.4f}")
#                 print(f"G Loss: {metrics['g_loss']:.4f}")
#                 print(f"SSIM: {metrics['ssim']:.4f}")
#                 print(f"PSNR: {metrics['psnr']:.4f}")
#                 print(f"G LR: {self.g_optimizer.param_groups[0]['lr']:.6f}")
#                 print(f"D LR: {self.d_optimizer.param_groups[0]['lr']:.6f}")

#         except KeyboardInterrupt:
#             print("Training interrupted by user")

#         # Cleanup
#         self.writer.close()
#         wandb.finish()

#         return {
#             "run_id": self.run_id,
#             "best_fid": best_fid,
#             "early_stopped": self.early_stopping.should_stop,
#         }


class GANEvaluator:
    def __init__(
        self,
        model_or_path,
        data_module,
        utils,
        device="cuda",
        project_name="gan-cifar10",
    ):
        self.device = device
        self.data_module = data_module
        self.utils = utils
        self.project_name = project_name

        # Generate unique run ID
        self.run_id = get_random_name(combo=[ADJECTIVES, NAMES], separator="")

        # Initialize model based on input type
        self.model = self._initialize_model(model_or_path)

        # Setup logging
        self.setup_logging()

        # Initialize reconstruction logger
        self.recon_logger = GANReconstructionLogger(run_id=self.run_id)

    def _initialize_model(self, model_or_path):
        if isinstance(model_or_path, torch.nn.Module):
            print("Using provided model instance for evaluation")
            model = model_or_path
            self.original_config = getattr(model, "config", {})
            self.original_run_id = getattr(model, "run_id", "unknown")
            self.checkpoint_source = "direct_model"
        else:
            print(f"Loading model from checkpoint: {model_or_path}")
            checkpoint = torch.load(model_or_path, map_location=self.device)
            model = GAN().to(self.device)

            if isinstance(checkpoint, dict):
                if "generator_state_dict" in checkpoint:
                    model.generator.load_state_dict(checkpoint["generator_state_dict"])
                    model.discriminator.load_state_dict(
                        checkpoint["discriminator_state_dict"]
                    )
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
    def evaluate(self, num_samples=1000):
        """Run evaluation"""
        self.model.eval()
        test_metrics = []
        test_loader = self.data_module.test_dataloader()

        print(f"\nStarting evaluation (Run ID: {self.run_id})")

        batch_idx = 0
        total_samples = 0

        for real_images in tqdm(test_loader, desc="Evaluating"):
            if total_samples >= num_samples:
                break

            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)

            # Generate fake images
            z = self.model.generate_latent(batch_size)
            fake_images = self.model.generator(z)

            # Calculate metrics
            metrics = self.utils.compute_metrics(
                real_images,
                fake_images,
                0.0,  # placeholder for d_loss during evaluation
                0.0,  # placeholder for g_loss during evaluation
            )

            test_metrics.append(metrics)

            # Save first batch reconstructions
            if batch_idx == 0:
                self._save_sample_images(real_images, fake_images)

            batch_idx += 1
            total_samples += batch_size

        # Calculate average metrics
        avg_metrics = self._average_metrics(test_metrics)

        # Log metrics
        self._log_metrics(avg_metrics)

        return avg_metrics

    def _average_metrics(self, metrics_list):
        """Average metrics properly handling inception score tuple"""
        avg_metrics = {}
        num_metrics = len(metrics_list)

        for k in metrics_list[0].keys():
            if k == "inception_score":
                means = [m[k][0] for m in metrics_list]
                stds = [m[k][1] for m in metrics_list]
                avg_metrics[k] = (sum(means) / num_metrics, sum(stds) / num_metrics)
            else:
                avg_metrics[k] = sum(m[k] for m in metrics_list) / num_metrics

        return avg_metrics

    def _log_metrics(self, metrics):
        """Log metrics to tensorboard and wandb"""
        is_mean, is_std = metrics["inception_score"]

        metric_dict = {
            "fid": metrics["fid"],
            "inception_score_mean": is_mean,
            "inception_score_std": is_std,
            "kid_mean": metrics["kid_mean"],
            "kid_std": metrics["kid_std"],
            "ssim": metrics["ssim"],
            "psnr": metrics["psnr"],
        }

        # Log to tensorboard
        for name, value in metric_dict.items():
            self.writer.add_scalar(f"test/{name}", value, 0)

        # Log to wandb
        wandb.log(metric_dict)

    def _save_sample_images(self, real_images, fake_images):
        """Save sample images during evaluation"""
        save_dir = Path("GAN_images") / self.run_id / "test"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create comparison grid
        n_images = min(real_images.size(0), 8)
        comparison = torch.cat(
            [
                self.utils.denormalize(real_images[:n_images]),
                self.utils.denormalize(fake_images[:n_images]),
            ]
        )

        save_image(
            comparison, save_dir / f"comparison.png", nrow=n_images, normalize=True
        )

        # Log to tensorboard and wandb
        grid = make_grid(comparison, nrow=n_images, normalize=True)
        self.writer.add_image("test/real_vs_fake", grid, 0)
        wandb.log({"test/real_vs_fake": wandb.Image(grid)})

    def cleanup(self):
        """Clean up logging"""
        self.writer.close()
        wandb.finish()


class GANTrainer:
    def __init__(self, model, train_loader, utils, config=None):
        self.model = model
        self.train_loader = train_loader
        self.utils = utils

        self.config = {
            "epochs": 200,
            "g_lr": 4e-4,
            "d_lr": 1e-4,
            "beta1": 0.5,
            "beta2": 0.999,
            "grad_clip": 1.0,
            "n_critic": 2,
            "patience": 25,
            "save_dir": "checkpoints",
            "log_dir": "runs",
            "project_name": "gan-cifar10",
            "fixed_samples_size": 16,
        }
        if config:
            self.config.update(config)

        self.device = next(model.parameters()).device
        self.save_dir = Path(self.config["save_dir"])
        self.save_dir.mkdir(exist_ok=True)
        self.run_id = get_random_name(combo=[ADJECTIVES, NAMES], separator="")

        # Initialize logging
        self.setup_logging()

        # Initialize reconstruction logger
        self.recon_logger = GANReconstructionLogger(run_id=self.run_id)

        # Create fixed noise for tracking progress
        self.fixed_latent = self.model.generate_latent(
            self.config["fixed_samples_size"]
        )

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
        self.g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=self.config["g_lr"],
            betas=(self.config["beta1"], self.config["beta2"]),
        )

        self.d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.config["d_lr"],
            betas=(self.config["beta1"], self.config["beta2"]),
        )

        self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        self.early_stopping = GANEarlyStopping(
            patience=self.config["patience"], monitor="fid"
        )

        self.g_scaler = GradScaler()
        self.d_scaler = GradScaler()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.model.generator.state_dict(),
            "discriminator_state_dict": self.model.discriminator.state_dict(),
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
            "d_optimizer_state_dict": self.d_optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "run_id": self.run_id,
        }

        torch.save(checkpoint, self.save_dir / f"latest_{self.run_id}.pt")
        if is_best:
            torch.save(checkpoint, self.save_dir / f"best_{self.run_id}.pt")

    def train_epoch(self, epoch):
        self.model.train()
        train_metrics = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, real_images in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)

            # Train Discriminator
            self.d_optimizer.zero_grad()
            with autocast():
                z = self.model.generate_latent(batch_size)
                fake_images = self.model.generator(z)
                d_loss, g_loss = self.utils.compute_adversarial_loss(
                    self.model.discriminator, real_images, fake_images, self.device
                )

            self.d_scaler.scale(d_loss).backward()
            if self.config["grad_clip"] > 0:
                self.d_scaler.unscale_(self.d_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.discriminator.parameters(), self.config["grad_clip"]
                )
            self.d_scaler.step(self.d_optimizer)
            self.d_scaler.update()

            # Train Generator
            if batch_idx % self.config["n_critic"] == 0:
                self.g_optimizer.zero_grad()
                with autocast():
                    z = self.model.generate_latent(batch_size)
                    fake_images = self.model.generator(z)
                    _, g_loss = self.utils.compute_adversarial_loss(
                        self.model.discriminator, real_images, fake_images, self.device
                    )

                self.g_scaler.scale(g_loss).backward()
                if self.config["grad_clip"] > 0:
                    self.g_scaler.unscale_(self.g_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.generator.parameters(), self.config["grad_clip"]
                    )
                self.g_scaler.step(self.g_optimizer)
                self.g_scaler.update()

                metrics = self.utils.compute_metrics(
                    real_images, fake_images, d_loss.item(), g_loss.item()
                )
                train_metrics.append(metrics)

                pbar.set_postfix(
                    {
                        "d_loss": f"{d_loss.item():.4f}",
                        "g_loss": f"{g_loss.item():.4f}",
                        "fid": f"{metrics['fid']:.4f}",
                    }
                )

        avg_metrics = self._average_metrics(train_metrics)
        self._log_training_step(avg_metrics, real_images, fake_images, epoch)
        return avg_metrics

    def _average_metrics(self, metrics_list):
        avg_metrics = {}
        num_metrics = len(metrics_list)

        for k in metrics_list[0].keys():
            if k == "inception_score":
                means = [m[k][0] for m in metrics_list]
                stds = [m[k][1] for m in metrics_list]
                avg_metrics[k] = (sum(means) / num_metrics, sum(stds) / num_metrics)
            else:
                avg_metrics[k] = sum(m[k] for m in metrics_list) / num_metrics

        return avg_metrics

    def _log_training_step(self, metrics, real_images, fake_images, epoch):
        # Log metrics
        log_metrics(self.writer, metrics, epoch)

        # Log images
        if epoch % 5 == 0:
            # Save reconstructions
            data_loaders = {"train": self.train_loader}
            save_paths = self.recon_logger.log_all_subsets(
                self.model, data_loaders, self.utils, self.device, epoch, self.writer
            )

            # Save fixed sample generations
            fixed_samples_path = self.recon_logger.save_fixed_samples(
                self.model.generator, self.fixed_latent, epoch
            )

            print(f"\nSaved images for epoch {epoch}:")
            for subset, path in save_paths.items():
                print(f"{subset}: {path}")
            print(f"Fixed samples: {fixed_samples_path}")

    def train(self):
        best_fid = float("inf")
        try:
            for epoch in range(self.config["epochs"]):
                metrics = self.train_epoch(epoch)

                self.g_scheduler.step(metrics["g_loss"])
                self.d_scheduler.step(metrics["d_loss"])

                if metrics["fid"] < best_fid:
                    best_fid = metrics["fid"]
                    self.save_checkpoint(epoch, metrics, is_best=True)

                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, metrics)

                if self.early_stopping(metrics):
                    print("Early stopping triggered!")
                    break

                is_mean, is_std = metrics["inception_score"]
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"FID Score: {metrics['fid']:.4f}")
                print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
                print(f"KID mean: {metrics['kid_mean']:.4f}")
                print(f"KID std: {metrics['kid_std']:.4f}")
                print(f"G Loss: {metrics['g_loss']:.4f}")
                print(f"D Loss: {metrics['d_loss']:.4f}")
                print(f"G LR: {self.g_optimizer.param_groups[0]['lr']:.6f}")
                print(f"D LR: {self.d_optimizer.param_groups[0]['lr']:.6f}")

        except KeyboardInterrupt:
            print("Training interrupted by user")

        self.writer.close()
        wandb.finish()

        return {
            "run_id": self.run_id,
            "best_fid": best_fid,
            "early_stopped": self.early_stopping.should_stop,
        }
