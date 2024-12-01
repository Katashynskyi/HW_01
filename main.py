"""Main script for training and evaluating various generative models.

This script provides a command-line interface for training and evaluating different types
of generative models including Autoencoders (AE), Variational Autoencoders (VAE), 
Generative Adversarial Networks (GAN), and Normalizing Flows (FLOW). It handles model
configuration loading, training orchestration, and evaluation procedures.

The script uses configuration files stored in 'common/configs/' directory with model-specific
settings stored in corresponding YAML files.

Usage:
    python script.py --model MODEL_TYPE --mode MODE [--checkpoint CHECKPOINT_PATH]

Arguments:
    --model: Type of generative model to train/evaluate
            Choices: 'AE', 'VAE', 'GAN', 'FLOW'
    --mode: Operation mode
            Choices: 'train' (train and evaluate model), 'evaluate' (evaluate only)
    --checkpoint: Path to model checkpoint for evaluation (required for evaluate mode)

Examples:
    python main.py --model AE --mode train 
	-will do training on train subset and evaluation on test subset.
    python main.py --model AE --mode evaluate --checkpoint checkpoints/best/best_SocialNikoletta.pt
        -will load best AE checkpoint so far and evaluate it.


    python main.py --model VAE --mode train
        -will do training on train subset and evaluation on test subset.
    python main.py --model VAE --mode evaluate --checkpoint checkpoints/best/best_IcyTotem.pt
        -will load best VAE checkpoint so far and evaluate it.


    python main.py --model GAN --mode train
        -will do training on train subset and evaluation on test subset.
    python main.py --model GAN --mode evaluate --checkpoint checkpoints/best/best_UnlikelyMarlane.pt
        -will load best GAN checkpoint so far and evaluate it.


    python main.py --model FLOW --mode train
        -will do training on train subset and evaluation on test subset.

    python main.py --model FLOW --mode evaluate --checkpoint checkpoints/best/latest_InterestedSianna.pt
        -will load best NORMLFOW checkpoint so far and evaluate it.

Functions:
    load_config(model_type: str) -> dict:
        Loads model-specific configuration from YAML file.
        Args:
            model_type: Type of model ('AE', 'VAE', 'GAN', 'FLOW')
        Returns:
            Dictionary containing model configuration parameters

    main():
        Main execution function that parses command-line arguments and orchestrates
        training/evaluation workflow.

Dependencies:
    - argparse: Command-line argument parsing
    - yaml: Configuration file parsing
    - torch: PyTorch deep learning framework
    - train_evaluate_all: Custom module containing model-specific training/evaluation functions

Notes:
    - GPU acceleration is automatically enabled if available
    - Training mode automatically runs evaluation after training completion
    - Checkpoint path can be specified either via command-line or in config file
"""

import argparse
import yaml
import torch

from train_evaluate_all import (
    train_autoencoder,
    evaluate_autoencoder,
    train_vae,
    evaluate_vae,
    train_gan,
    evaluate_gan,
    train_flow,
    evaluate_flow,
)


def load_config(model_type):
    """Load and process the configuration for a specific model type.

    Args:
        model_type (str): Type of the model to load configuration for.
                         Must be one of: 'AE', 'VAE', 'GAN', 'FLOW'

    Returns:
        dict: Configuration dictionary containing model parameters and settings.
              Includes an additional 'device' key specifying 'cuda' or 'cpu'
              based on hardware availability.

    Raises:
        FileNotFoundError: If the configuration file for the specified model type
                          doesn't exist in the common/configs directory.
        yaml.YAMLError: If the configuration file contains invalid YAML syntax.
    """
    config_path = f"common/configs/{model_type}_config.yaml"
    with open(config_path, "r", encoding="UTF-8") as file:
        config = yaml.safe_load(file)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def main():
    """Main execution function for the generative models training/evaluation pipeline.

    This function handles:
    1. Command-line argument parsing
    2. Configuration loading
    3. Model training or evaluation based on specified mode
    4. Automatic evaluation after training
    5. Checkpoint management for evaluation mode

    The function implements the following workflow:
    - For training mode: Trains the specified model and evaluates it
    - For evaluation mode: Loads a checkpoint and evaluates the model

    Raises:
        ValueError: If checkpoint path is not provided for evaluation mode
        argparse.ArgumentError: If invalid arguments are provided
    """
    parser = argparse.ArgumentParser(description="Train or evaluate generative models")
    parser.add_argument("--model", choices=["AE", "VAE", "GAN", "FLOW"])
    parser.add_argument("--mode", choices=["train", "evaluate"])
    parser.add_argument("--checkpoint", help="Path to model checkpoint for evaluation")

    args = parser.parse_args()

    config = load_config(args.model)

    if args.mode == "train":
        if args.model == "AE":
            best_path = train_autoencoder()
            evaluate_autoencoder(best_path)
        elif args.model == "VAE":
            model, _, checkpoint = train_vae()
            evaluate_vae(model=model)
        elif args.model == "GAN":
            model, _, checkpoint = train_gan()
            evaluate_gan(model=model)
        elif args.model == "FLOW":
            best_path = train_flow()
            evaluate_flow(best_path)
    else:  # evaluate
        if not args.checkpoint:
            args.checkpoint = config.get("checkpoint_path")
            if not args.checkpoint:
                raise ValueError("Checkpoint path required for evaluation")

    if args.model == "AE":
        evaluate_autoencoder(args.checkpoint)
    elif args.model == "VAE":
        evaluate_vae(checkpoint_path=args.checkpoint)
    elif args.model == "GAN":
        evaluate_gan(checkpoint_path=args.checkpoint)
    elif args.model == "FLOW":
        evaluate_flow(args.checkpoint)


if __name__ == "__main__":
    main()
from torch.utils.tensorboard import SummaryWriter
