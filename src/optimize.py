import optuna
import torch
import torch.optim as optim
import torch.nn as nn
import yaml
import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional
import tempfile
import os

from model import ResNet50BirdClassifier
from train import Trainer, MetricsCalculator
from data import create_dataloaders

class HyperparameterOptimizer:
    """
    Automated hyperparameter tuning using Optuna with PyTorch and MLflow integration.

    This class handles the complete optimization workflow:
    - Suggests hyperparameters from defined search space
    - Trains PyTorch models with suggested parameters
    - Tracks all experiments in MLflow
    - Returns best hyperparameters and model
    """

    def __init__(self, data_dir: str, config_path: str = 'configs/config.yaml',
                 search_space_path: str = 'configs/search_space.yaml'):
        """Initialize the optimizer with data and configuration."""
        self.data_dir = data_dir

        # Load base configuration
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        # Load search space configuration
        with open(search_space_path, 'r') as f:
            self.search_space = yaml.safe_load(f)

        # Setup MLflow
        mlflow.set_tracking_uri(self.base_config['mlflow']['tracking_uri'])
        mlflow.set_experiment(f"{self.base_config['mlflow']['experiment_name']}_optimization")

        # Create dataloaders once (shared across trials)
        self.train_loader, self.val_loader, self.class_names = create_dataloaders(
            data_dir=data_dir,
            batch_size=32,  # Will be updated per trial
            train_split=self.base_config['data']['train_split'],
            image_size=self.base_config['data']['image_size']
        )

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on search space configuration.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']

            if param_type == 'float':
                if param_config.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'], log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )

            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )

            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )

            # Handle conditional parameters (e.g., momentum for SGD)
            if 'depends_on' in param_config:
                condition = param_config['depends_on']
                param_value, expected_value = condition.split(':')
                if params.get(param_value) == expected_value:
                    # Apply the parameter
                    if param_type == 'float':
                        if param_config.get('log', False):
                            params[param_name] = trial.suggest_float(
                                param_name, param_config['low'], param_config['high'], log=True
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name, param_config['low'], param_config['high']
                            )

        return params

    def create_optimizer(self, model: nn.Module, optimizer_name: str,
                        learning_rate: float, **kwargs) -> optim.Optimizer:
        """Create PyTorch optimizer based on hyperparameters."""
        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=kwargs.get('weight_decay', 0))
        elif optimizer_name == 'AdamW':
            return optim.AdamW(model.parameters(), lr=learning_rate,
                              weight_decay=kwargs.get('weight_decay', 0))
        elif optimizer_name == 'SGD':
            return optim.SGD(model.parameters(), lr=learning_rate,
                           momentum=kwargs.get('momentum', 0.9),
                           weight_decay=kwargs.get('weight_decay', 0))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation F1 score (to be maximized)
        """
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)

        # Create config for this trial
        trial_config = self.base_config.copy()
        trial_config['training'].update(params)

        # Log hyperparameters to MLflow
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)

            try:
                # Create model
                model = ResNet50BirdClassifier(
                    num_classes=trial_config['model']['num_classes'],
                    pretrained=trial_config['model']['pretrained']
                )

                # Create custom optimizer
                optimizer = self.create_optimizer(
                    model,
                    params['optimizer'],
                    params['learning_rate'],
                    weight_decay=params.get('weight_decay', 0),
                    momentum=params.get('momentum', 0.9)
                )

                # Create trainer with custom optimizer
                trainer = Trainer(model, trial_config)
                trainer.optimizer = optimizer

                # Create new dataloaders with the batch size for this trial
                batch_size = params['batch_size']
                train_loader, val_loader, _ = create_dataloaders(
                    data_dir=self.data_dir,
                    batch_size=batch_size,
                    train_split=self.base_config['data']['train_split'],
                    image_size=self.base_config['data']['image_size']
                )
                trainer.train_loader = train_loader
                trainer.val_loader = val_loader

                # Train for limited epochs to speed up optimization
                max_epochs = min(params['epochs'], 20)  # Cap at 20 for faster trials

                best_val_f1 = 0.0
                patience_counter = 0
                patience = params.get('early_stopping_patience', 10)

                for epoch in range(max_epochs):
                    # Train
                    train_metrics = trainer.train_epoch(trainer.train_loader)
                    trainer.log_metrics(train_metrics, epoch, 'train')

                    # Validate
                    val_metrics = trainer.validate_epoch(trainer.val_loader)
                    trainer.log_metrics(val_metrics, epoch, 'val')

                    # Track best validation F1
                    current_val_f1 = val_metrics.get('f1_score', 0)
                    if current_val_f1 > best_val_f1:
                        best_val_f1 = current_val_f1
                        patience_counter = 0

                        # Save best model for this trial
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                            torch.save(model.state_dict(), tmp_file.name)
                            mlflow.log_artifact(tmp_file.name, 'best_model.pth')
                    else:
                        patience_counter += 1

                    # Early stopping
                    if patience_counter >= patience:
                        mlflow.log_metric('early_stopped_epoch', epoch)
                        break

                # Report intermediate value to Optuna
                trial.report(best_val_f1, epoch)

                # Check if trial should be pruned
                if trial.should_prune():
                    mlflow.log_metric('pruned', True)
                    raise optuna.exceptions.TrialPruned()

                mlflow.log_metric('best_val_f1', best_val_f1)
                mlflow.log_metric('trial_completed', True)

                return best_val_f1

            except Exception as e:
                mlflow.log_metric('trial_failed', True)
                mlflow.log_param('error', str(e))
                print(f"Trial {trial.number} failed: {e}")
                return 0.0

    def optimize(self, n_trials: int = 50, study_name: str = None) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study

        Returns:
            Dictionary containing best hyperparameters and study results
        """
        if study_name is None:
            study_name = "bird_classifier_optimization"

        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        print(f"Starting optimization with {n_trials} trials...")
        print(f"Study name: {study_name}")
        print(f"MLflow experiment: {self.base_config['mlflow']['experiment_name']}_optimization")
        print("-" * 50)

        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)

        # Results
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial

        print("\n" + "=" * 50)
        print("OPTIMIZATION COMPLETED")
        print("=" * 50)
        print(f"Best validation F1 score: {best_value:.4f}")
        print(f"Best trial number: {best_trial.number}")
        print("\nBest hyperparameters:")
        for param_name, param_value in best_params.items():
            print(f"  {param_name}: {param_value}")

        print(f"\nNumber of completed trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

        # Log final results to MLflow
        with mlflow.start_run(run_name="optimization_summary"):
            mlflow.log_params(best_params)
            mlflow.log_metric('best_f1_score', best_value)
            mlflow.log_metric('n_trials', len(study.trials))

        return {
            'best_params': best_params,
            'best_value': best_value,
            'best_trial': best_trial,
            'study': study,
            'n_trials': len(study.trials)
        }

    def train_final_model(self, best_params: Dict[str, Any], epochs: int = None) -> str:
        """
        Train final model with best hyperparameters.

        Args:
            best_params: Best hyperparameters from optimization
            epochs: Number of training epochs (defaults to config value)

        Returns:
            Path to saved final model
        """
        # Create final config
        final_config = self.base_config.copy()
        final_config['training'].update(best_params)

        if epochs:
            final_config['training']['epochs'] = epochs

        # Create and train final model
        model = ResNet50BirdClassifier(
            num_classes=final_config['model']['num_classes'],
            pretrained=final_config['model']['pretrained']
        )

        # Create optimizer with best parameters
        optimizer = self.create_optimizer(
            model,
            best_params['optimizer'],
            best_params['learning_rate'],
            weight_decay=best_params.get('weight_decay', 0),
            momentum=best_params.get('momentum', 0.9)
        )

        # Create trainer
        trainer = Trainer(model, final_config)
        trainer.optimizer = optimizer

        # Create dataloaders with best batch size
        batch_size = best_params['batch_size']
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=self.data_dir,
            batch_size=batch_size,
            train_split=self.base_config['data']['train_split'],
            image_size=self.base_config['data']['image_size']
        )
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader

        # Train final model
        with mlflow.start_run(run_name="final_model_training"):
            mlflow.log_params(best_params)
            mlflow.log_param('training_type', 'final_model')

            trainer.train(trainer.train_loader, trainer.val_loader)

        # Save final model
        model_path = 'final_bird_classifier.pth'
        model.save_model(model_path)

        print(f"\nFinal model saved to: {model_path}")
        return model_path