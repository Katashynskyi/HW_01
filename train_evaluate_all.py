"""Common utilities and data modules for deep generative models.

This module contains shared functionality used across different model implementations:
- Base data module classes for loading and processing datasets
- Configuration management utilities
- Common training utilities and helper functions
"""

from pathlib import Path
import yaml
import torch

# Autoencoder
from models.autoencoder.model import Autoencoder
from models.autoencoder.trainer import (
    AutoencoderTrainingUtils,
    AutoencoderTrainer,
    AutoencoderEvaluator,
)

# VAE
from models.vae.model import VAE
from common.data import VAEDataModule
from models.vae.trainer import VAETrainingUtils, VAEEvaluator, VAETrainer

# GAN
from models.gan.model import GAN
from common.data import GANDataModule
from models.gan.trainer import GANTrainingUtils, GANTrainer, GANEvaluator

# Normalizing Flow
from models.NormalizingFlow.model import RealNVP
from common.data import FlowDataModule
from common.utils import FlowTrainingUtils
from models.NormalizingFlow.trainer import FlowEvaluator, FlowTrainer

with open("common/configs/AE_config.yaml", "r", encoding="UTF-8") as file:
    config = yaml.safe_load(file)
# Ensure device is set correctly
config["device"] = "cuda" if torch.cuda.is_available() else "cpu"


def train_autoencoder():
    data_module = VAEDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        subset_size=config["subset_size"],
    )
    data_module.setup()
    model = Autoencoder(latent_dim=config["latent_dim"]).to(config["device"])
    utils = AutoencoderTrainingUtils(device=config["device"])
    trainer = AutoencoderTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        utils=utils,
        config=config,
    )
    results = trainer.train()
    return Path(config["save_dir"]) / f"best_{results['run_id']}.pt"


def evaluate_autoencoder(checkpoint_path):
    data_module = VAEDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        subset_size=config["subset_size"],
    )
    data_module.setup()
    utils = AutoencoderTrainingUtils(device=config["device"])
    evaluator = AutoencoderEvaluator(
        checkpoint_path=checkpoint_path,
        data_module=data_module,
        utils=utils,
        device=config["device"],
        project_name=config["project_name"],
    )
    metrics = evaluator.evaluate_and_visualize(num_vis_samples=10)
    return metrics


def train_vae():
    """Training function for VAE"""
    # Load config
    with open("common/configs/VAE_config.yaml", "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize components
    model = VAE().to(device)
    data_module = VAEDataModule(subset_size=config["subset_size"])
    data_module.setup()
    utils = VAETrainingUtils(device=device)

    # Create base directory for images
    Path("VAE_images").mkdir(exist_ok=True)

    # Train model
    trainer = VAETrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        utils=utils,
        config=config,
    )

    results = trainer.train()

    # Save best checkpoint path to config
    best_checkpoint_path = str(trainer.save_dir / f"{trainer.run_id}_best.pt")
    config["checkpoint_path"] = best_checkpoint_path

    with open("common/configs/VAE_config.yaml", "w", encoding="UTF-8") as file:
        yaml.safe_dump(config, file)

    return model, results, best_checkpoint_path


def evaluate_vae(model=None, checkpoint_path=None):
    """
    Evaluation function for VAE
    Args:
        model: Optional model instance to evaluate
        checkpoint_path: Optional path to model checkpoint
    """
    with open("common/configs/VAE_config.yaml", "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data and utils
    data_module = VAEDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    data_module.setup()
    utils = VAETrainingUtils(device=device)

    # Determine model source
    if model is not None:
        model_source = model
    elif checkpoint_path is not None:
        model_source = checkpoint_path
    else:
        config_checkpoint = config.get("checkpoint_path")
        if not config_checkpoint:
            raise ValueError("No checkpoint path found in config and no model provided")
        model_source = config_checkpoint

    # Initialize evaluator
    evaluator = VAEEvaluator(
        model_or_path=model_source,
        data_module=data_module,
        utils=utils,
        device=device,
        project_name=config["project_name"],
    )

    # Run evaluation
    try:
        metrics = evaluator.evaluate()
        print("\nEvaluation Results:")
        for name, value in metrics.items():
            print(f"{name:15s}: {value:.4f}")

        print(f"\nResults saved in:")
        print(f"- VAE_images/{evaluator.run_id}/")
        print(f"- WandB project: {evaluator.project_name}")
        print(f"- Run ID: {evaluator.run_id}")

    finally:
        evaluator.cleanup()

    return metrics


def train_gan():
    with open("common/configs/GAN_config.yaml", "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAN().to(device)
    data_module = GANDataModule(subset_size=config["subset_size"])
    data_module.setup()
    utils = GANTrainingUtils(device=device)

    Path("GAN_images").mkdir(exist_ok=True)

    trainer = GANTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        utils=utils,
        config=config,
    )

    results = trainer.train()
    best_checkpoint_path = str(trainer.save_dir / f"best_{trainer.run_id}.pt")
    config["checkpoint_path"] = best_checkpoint_path

    with open("common/configs/GAN_config.yaml", "w", encoding="UTF-8") as file:
        yaml.safe_dump(config, file)

    return model, results, best_checkpoint_path


def evaluate_gan(model=None, checkpoint_path=None):
    with open("common/configs/GAN_config.yaml", "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = GANDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    data_module.setup()
    utils = GANTrainingUtils(device=device)

    model_source = (
        model
        if model is not None
        else (
            checkpoint_path
            if checkpoint_path is not None
            else config.get("checkpoint_path")
        )
    )

    if not model_source:
        raise ValueError("No checkpoint path found in config and no model provided")

    evaluator = GANEvaluator(
        model_or_path=model_source,
        data_module=data_module,
        utils=utils,
        device=device,
        project_name=config["project_name"],
    )

    try:
        metrics = evaluator.evaluate()
        print("\nEvaluation Results:")
        is_mean, is_std = metrics["inception_score"]

        print(f"FID Score: {metrics['fid']:.4f}")
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        print(f"KID mean: {metrics['kid_mean']:.4f}")
        print(f"KID std: {metrics['kid_std']:.4f}")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"PSNR: {metrics['psnr']:.4f}")

        print(f"\nResults saved in:")
        print(f"- GAN_images/{evaluator.run_id}/")
        print(f"- WandB project: {evaluator.project_name}")
        print(f"- Run ID: {evaluator.run_id}")

    finally:
        evaluator.cleanup()

    return metrics


def train_flow() -> Path:
    """
    Train a normalizing flow model and return the path to the best checkpoint.
    """
    # # Load configuration
    with open("common/configs/FLOW_config.yaml", "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)
    # Initialize components
    data_module = FlowDataModule(
        batch_size=config["batch_size"], subset_size=config["subset_size"]
    )
    data_module.setup()

    model = RealNVP(
        hidden_dim=config["hidden_dim"], num_layers=config["num_layers"]
    ).to(config["device"])

    utils = FlowTrainingUtils(device=config["device"])

    trainer = FlowTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        utils=utils,
        project_name=config["project_name"],
        save_dir=config["save_dir"],
        log_dir=config["log_dir"],
        learning_rate=config["learning_rate"],
        max_epochs=config["max_epochs"],
    )

    trainer.train()
    return Path(config["save_dir"]) / f"best_{trainer.run_id}.pt"


def evaluate_flow(checkpoint_path: str) -> dict:
    """
    Evaluate a trained flow model and return metrics.
    """
    # # Load configuration
    with open("common/configs/FLOW_config.yaml", "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)
    # Initialize components
    data_module = FlowDataModule(
        batch_size=config["batch_size"], subset_size=config["subset_size"]
    )
    data_module.setup()

    model = RealNVP(
        hidden_dim=config["hidden_dim"], num_layers=config["num_layers"]
    ).to(config["device"])

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config["device"])
    # model.load_state_dict(checkpoint["model_state_dict"])

    utils = FlowTrainingUtils(device=config["device"])

    evaluator = FlowEvaluator(
        model=model,
        test_loader=data_module.test_dataloader(),
        utils=utils,
        results_dir=config["results_dir"],
    )

    metrics = evaluator.evaluate()
    return metrics
