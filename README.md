# Generative Models Project

This project implements and evaluates various generative models on the CIFAR-10 dataset, including Autoencoders (AE), Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), and Normalizing Flows.

## Status

- ✅ Autoencoder (AE)
- ✅ Variational Autoencoder (VAE) 
- ✅ Generative Adversarial Network (GAN)
- ❌ Normalizing Flow (implementation incomplete)

Note: Metrics implementations use external libraries rather than custom implementations.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/generative-models.git
cd generative-models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have PyTorch installed with CUDA support if using GPU acceleration.

## Data

The project uses the CIFAR-10 dataset which will be automatically downloaded when running the models. No additional data setup is required.

## Usage

Each model can be trained and evaluated using the following command structure:

```bash
python main.py --model MODEL_TYPE --mode MODE [--checkpoint CHECKPOINT_PATH]
```

### Quick Start Examples

1. **Autoencoder (AE)**
```bash
# Train AE
python main.py --model AE --mode train 

# Evaluate AE using checkpoint
python main.py --model AE --mode evaluate --checkpoint checkpoints/best/best_SocialNikoletta.pt
```

2. **Variational Autoencoder (VAE)**
```bash
# Train VAE
python main.py --model VAE --mode train

# Evaluate VAE using checkpoint
python main.py --model VAE --mode evaluate --checkpoint checkpoints/best/best_IcyTotem.pt
```

3. **Generative Adversarial Network (GAN)**
```bash
# Train GAN
python main.py --model GAN --mode train

# Evaluate GAN using checkpoint
python main.py --model GAN --mode evaluate --checkpoint checkpoints/best/best_UnlikelyMarlane.pt
```

4. **Normalizing Flow** (Note: Implementation incomplete)
```bash
# Train Flow
python main.py --model FLOW --mode train

# Evaluate Flow using checkpoint
python main.py --model FLOW --mode evaluate --checkpoint checkpoints/best/latest_InterestedSianna.pt
```

### Command Arguments

- `--model`: Model type (`AE`, `VAE`, `GAN`, `FLOW`)
- `--mode`: Operation mode (`train`, `evaluate`)
- `--checkpoint`: Path to model checkpoint (required for evaluation mode)

## Project Structure

```
├── README.md
├── checkpoints/              # Model checkpoints
├── common/                   # Shared utilities and configs
│   ├── configs/             # Model configuration files
│   ├── data.py              # Data loading utilities
│   ├── logging.py           # Logging functionality
│   └── utils.py             # Common utilities
├── main.py                  # Main training/evaluation script
├── models/                  # Model implementations
│   ├── autoencoder/
│   ├── vae/
│   ├── gan/
│   └── NormalizingFlow/
├── notebooks/               # Training and evaluation notebooks
└── requirements.txt
```

## Configuration

Model-specific configurations are stored in YAML files under `common/configs/`. Modify these files to adjust model architecture, training parameters, and other settings.

## Logging

Training progress and metrics are logged using TensorBoard. Logs are stored in the project directory.

## Notes

- GPU acceleration is automatically used if available
- Training mode automatically runs evaluation after completion
- Pre-trained model checkpoints are provided in the `checkpoints/best/` directory
- The Normalizing Flow implementation is currently incomplete and may not function as expected

