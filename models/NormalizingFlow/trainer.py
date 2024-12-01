# All required imports
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import wandb
from tqdm.auto import tqdm
from pathlib import Path
import json
from typing import Optional, Dict, Any

from common.utils import FlowTrainingUtils, EarlyStopping
from models.NormalizingFlow.model import RealNVP
from common.data import FlowDataModule


class FlowTrainer:
    def __init__(
        self,
        model: RealNVP,
        train_loader: DataLoader,
        utils: FlowTrainingUtils,
        project_name: str = "normalizing-flow-cifar10",
        save_dir: str = "checkpoints",
        log_dir: str = "runs",
        learning_rate: float = 1e-4,
        max_epochs: int = 100,
    ):
        self.model = model
        self.train_loader = train_loader
        self.utils = utils
        self.device = next(model.parameters()).device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize unique run ID and logging
        self.run_id = get_random_name(combo=[ADJECTIVES, NAMES], separator="")
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{self.run_id}")
        wandb.init(project=project_name, name=self.run_id, id=self.run_id)

        # Training components
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=4, verbose=True
        )
        self.early_stopping = EarlyStopping(patience=25)
        self.max_epochs = max_epochs
        self.best_fid = float("inf")

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "run_id": self.run_id,
        }

        torch.save(checkpoint, self.save_dir / f"latest_{self.run_id}.pt")
        if is_best:
            torch.save(checkpoint, self.save_dir / f"best_{self.run_id}.pt")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            real_images = (
                batch[0].to(self.device)
                if isinstance(batch, list)
                else batch.to(self.device)
            )

            # Forward pass and loss computation
            self.optimizer.zero_grad()
            loss, _ = self.model.compute_loss(real_images)

            # Generate samples for metrics
            generated_images = self.model.sample(real_images.size(0))

            # Update model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update metrics
            self.utils.update_metrics(real_images, generated_images)
            total_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return self.utils.compute_metrics(total_loss / len(self.train_loader))

    def train(self):
        try:
            for epoch in range(self.max_epochs):
                metrics = self.train_epoch()

                # Log metrics and update scheduler
                self.utils.log_metrics(self.writer, metrics, epoch)
                self.scheduler.step(metrics["fid"])

                # Generate and log sample images
                if epoch % 5 == 0:
                    with torch.no_grad():
                        real_batch = next(iter(self.train_loader))[0].to(self.device)
                        fake_batch = self.model.sample(real_batch.size(0))
                        self.utils.log_images(
                            self.writer, real_batch, fake_batch, epoch
                        )

                # Save checkpoints
                is_best = metrics["fid"] < self.best_fid
                if is_best:
                    self.best_fid = metrics["fid"]
                self.save_checkpoint(metrics, is_best)

                # Early stopping check
                if self.early_stopping(metrics["fid"]):
                    print("Early stopping triggered!")
                    break

                print(f"Epoch {epoch+1}/{self.max_epochs} - FID: {metrics['fid']:.4f}")

        except KeyboardInterrupt:
            print("Training interrupted by user")

        self.writer.close()
        wandb.finish()


class FlowEvaluator:
    def __init__(
        self,
        model: RealNVP,
        test_loader: DataLoader,
        utils: FlowTrainingUtils,
        results_dir: str = "eval_results",
    ):
        self.model = model
        self.test_loader = test_loader
        self.utils = utils
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.run_id = get_random_name(combo=[ADJECTIVES, NAMES], separator="")

        # Setup logging
        self.writer = SummaryWriter(f"eval_runs/{self.run_id}")
        wandb.init(project="normalizing-flow-cifar10", name=f"eval_{self.run_id}")

    @torch.no_grad()
    def evaluate(self, num_samples: int = 1000) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        num_batches = len(self.test_loader)

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            real_images = batch[0].to(next(self.model.parameters()).device)
            loss, _ = self.model.compute_loss(real_images)
            generated_images = self.model.sample(real_images.size(0))

            self.utils.update_metrics(real_images, generated_images)
            total_loss += loss.item()

            # Save sample visualizations
            if batch is next(iter(self.test_loader)):
                self.save_samples(real_images, generated_images)

        metrics = self.utils.compute_metrics(total_loss / num_batches)
        self.save_results(metrics)

        self.writer.close()
        wandb.finish()
        return metrics

    def save_samples(self, real_images: torch.Tensor, generated_images: torch.Tensor):
        self.utils.log_images(self.writer, real_images, generated_images, 0)

        comparison = torch.cat(
            [
                self.utils.denormalize(real_images[:8]),
                self.utils.denormalize(generated_images[:8]),
            ]
        )
        grid = vutils.make_grid(comparison, nrow=8, normalize=True)
        save_image(grid, self.results_dir / f"samples_{self.run_id}.png")

    def save_results(self, metrics: Dict[str, float]):
        results_file = self.results_dir / f"metrics_{self.run_id}.json"
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=4)


class FlowIntegration:
    def __init__(
        self,
        batch_size: int = 64,
        subset_size: int = 100,
        hidden_dim: int = 64,
        num_layers: int = 8,
        learning_rate: float = 1e-4,
        max_epochs: int = 10,
        project_name: str = "normalizing-flow-cifar10",
        save_dir: str = "checkpoints",
        log_dir: str = "runs",
    ):
        self.config = {
            "batch_size": batch_size,
            "subset_size": subset_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "project_name": project_name,
            "save_dir": save_dir,
            "log_dir": log_dir,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        # Initialize components
        data_module = FlowDataModule(
            batch_size=self.config["batch_size"], subset_size=self.config["subset_size"]
        )
        data_module.setup()

        model = RealNVP(
            hidden_dim=self.config["hidden_dim"], num_layers=self.config["num_layers"]
        ).to(self.device)

        utils = FlowTrainingUtils(device=self.device)

        trainer = FlowTrainer(
            model=model,
            train_loader=data_module.train_dataloader(),
            utils=utils,
            project_name=self.config["project_name"],
            save_dir=self.config["save_dir"],
            log_dir=self.config["log_dir"],
            learning_rate=self.config["learning_rate"],
            max_epochs=self.config["max_epochs"],
        )

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])

        trainer.train()

        evaluator = FlowEvaluator(
            model=model, test_loader=data_module.test_dataloader(), utils=utils
        )
        test_metrics = evaluator.evaluate()

        return {
            "run_id": trainer.run_id,
            "train_metrics": trainer.best_fid,
            "test_metrics": test_metrics,
        }

