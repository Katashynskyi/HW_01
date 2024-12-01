from pathlib import Path
import torch
import torchvision.utils as vutils
from torchvision.utils import save_image
import wandb
import numpy as np

class ReconstructionLogger:
    def __init__(self, base_save_dir="VAE_images", run_id=None):
        """
        Initialize reconstruction logger
        Args:
            base_save_dir: Base directory for saving images
            run_id: Unique identifier for the run
        """
        self.base_dir = Path(base_save_dir)
        self.run_id = run_id if run_id else "default_run"

        # Create directories
        self.save_dir = self.base_dir / self.run_id
        for subset in ["train", "val", "test"]:
            (self.save_dir / subset).mkdir(parents=True, exist_ok=True)

    def save_reconstructions(self, images, recons, subset, epoch, writer=None):
        """
        Save and log reconstruction samples
        Args:
            images: Original images
            recons: Reconstructed images
            subset: Data subset ('train', 'val', or 'test')
            epoch: Current epoch number
            writer: TensorBoard SummaryWriter instance
        """
        # Create comparison grid
        comparison = torch.cat([images, recons])
        grid = vutils.make_grid(comparison, nrow=len(images))

        # Save images
        filename = f"epoch_{epoch}_{subset}_reconstructions.png"
        save_path = self.save_dir / subset / filename
        save_image(grid, save_path)

        # Log to tensorboard
        if writer is not None:
            writer.add_image(f"{subset}/reconstructions", grid, epoch)

        # Log to wandb
        wandb.log({f"{subset}_reconstructions": wandb.Image(grid), "epoch": epoch})

        return save_path

    def log_all_subsets(self, model, data_loaders, utils, device, epoch, writer=None):
        """
        Log reconstructions for all data subsets
        Args:
            model: VAE model
            data_loaders: Dict containing train, val, test dataloaders
            utils: VAETrainingUtils instance
            device: torch device
            epoch: Current epoch number
            writer: TensorBoard SummaryWriter instance
        """
        model.eval()
        save_paths = {}

        with torch.no_grad():
            for subset, loader in data_loaders.items():
                # Get first batch
                images, _ = next(iter(loader))
                images = images.to(device)

                # Generate reconstructions
                recons, _, _ = model(images)

                # Prepare images
                images, recons = utils.prepare_for_metrics(images, recons)

                # Save first 8 images
                save_path = self.save_reconstructions(
                    images[:8], recons[:8], subset, epoch, writer
                )
                save_paths[subset] = save_path

        return save_paths


class GANReconstructionLogger:
    def __init__(self, run_id):
        self.run_id = run_id
        self.base_dir = Path("GAN_images") / run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log_all_subsets(
        self, model, data_loaders, utils, device, epoch, writer, num_samples=8
    ):
        save_paths = {}

        for subset_name, loader in data_loaders.items():
            save_dir = self.base_dir / subset_name
            save_dir.mkdir(exist_ok=True)

            # Get a batch of real images
            real_images = next(iter(loader))
            real_images = real_images.to(device)[:num_samples]

            # Generate fake images
            with torch.no_grad():
                z = model.generate_latent(num_samples)
                fake_images = model.generator(z)

            # Prepare images for saving
            real_images = utils.denormalize(real_images)
            fake_images = utils.denormalize(fake_images)

            # Create comparison grid
            comparison = torch.cat([real_images, fake_images])
            filename = save_dir / f"epoch_{epoch}.png"
            save_image(comparison, filename, nrow=num_samples, normalize=True)

            # Log to tensorboard and wandb
            grid = vutils.make_grid(comparison, nrow=num_samples, normalize=True)
            writer.add_image(f"{subset_name}/real_vs_fake", grid, epoch)
            wandb.log({f"{subset_name}/real_vs_fake": wandb.Image(grid)}, step=epoch)

            save_paths[subset_name] = filename

        return save_paths

    def save_fixed_samples(self, generator, latent_vectors, epoch):
        """Save generations from fixed latent vectors to track progress"""
        with torch.no_grad():
            fake_images = generator(latent_vectors)
            fake_images = (fake_images + 1) / 2  # [-1, 1] -> [0, 1]

            save_dir = self.base_dir / "fixed_samples"
            save_dir.mkdir(exist_ok=True)
            filename = save_dir / f"epoch_{epoch}.png"

            save_image(fake_images, filename, nrow=int(np.sqrt(len(latent_vectors))))
            return filename
