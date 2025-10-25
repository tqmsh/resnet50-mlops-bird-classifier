import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class MetricsCalculator:
    """Calculate and track training metrics."""

    @staticmethod
    def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            outputs: Model predictions (logits)
            targets: Ground truth labels
            num_classes: Number of classes

        Returns:
            Dictionary of metrics including accuracy, precision, recall, F1
        """
        # Get predicted classes
        preds = torch.argmax(outputs, dim=1)

        # Convert to numpy for sklearn metrics
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Calculate accuracy
        accuracy = accuracy_score(targets_np, preds_np)

        # Calculate precision, recall, F1 (macro averaged for multi-class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_np, preds_np, average='macro', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

class Trainer:
    """
    PyTorch trainer for bird classification with MLflow integration.

    Handles training loop, validation, and metrics tracking.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            config: Training configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()

        # Training state
        self.best_val_f1 = 0.0

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        training_config = self.config['training']

        if training_config['optimizer'] == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config.get('weight_decay', 0)
            )
        elif training_config['optimizer'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config.get('weight_decay', 0)
            )
        elif training_config['optimizer'] == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                momentum=training_config.get('momentum', 0.9),
                weight_decay=training_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_targets = []

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for data, targets in progress_bar:
            data, targets = data.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())

            # Update progress bar
            current_loss = total_loss / total_samples
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

        # Calculate epoch metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        epoch_metrics = self.metrics_calculator.calculate_metrics(
            all_outputs, all_targets, self.config['model']['num_classes']
        )
        epoch_metrics['loss'] = total_loss / total_samples

        return epoch_metrics

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)

            for data, targets in progress_bar:
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                # Accumulate metrics
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

                all_outputs.append(outputs.detach())
                all_targets.append(targets.detach())

                # Update progress bar
                current_loss = total_loss / total_samples
                progress_bar.set_postfix({'val_loss': f'{current_loss:.4f}'})

        # Calculate epoch metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        epoch_metrics = self.metrics_calculator.calculate_metrics(
            all_outputs, all_targets, self.config['model']['num_classes']
        )
        epoch_metrics['loss'] = total_loss / total_samples

        return epoch_metrics

    def log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str):
        """
        Log metrics (placeholder for MLflow integration).

        Args:
            metrics: Metrics dictionary
            epoch: Current epoch
            phase: Training phase ('train' or 'val')
        """
        print(f"Epoch {epoch} {phase}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: Optional[int] = None):
        """
        Train the model for specified number of epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (uses config value if None)
        """
        if epochs is None:
            epochs = self.config['training']['epochs']

        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Optimizer: {type(self.optimizer).__name__}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            train_metrics = self.train_epoch(train_loader)
            self.log_metrics(train_metrics, epoch + 1, 'train')

            # Validation
            val_metrics = self.validate_epoch(val_loader)
            self.log_metrics(val_metrics, epoch + 1, 'val')

            # Save best model based on F1 score
            current_val_f1 = val_metrics.get('f1_score', 0.0)
            if current_val_f1 > self.best_val_f1:
                self.best_val_f1 = current_val_f1
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved with F1: {current_val_f1:.4f}")

        print(f"\nTraining completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")