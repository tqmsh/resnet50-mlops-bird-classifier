import optuna
import torch
import torch.optim as optim
import torch.nn as nn
from src.config_loader import load_config_with_env_vars
import mlflow
import mlflow.pytorch
from typing import Dict, Any
import tempfile
from datetime import datetime

from src.model import ResNetBirdClassifier
from src.train import Trainer
from src.data import create_dataloaders

class HyperparameterOptimizer:
    """
    Automated hyperparameter tuning using Optuna with PyTorch and MLflow integration.

    This class handles the complete optimization workflow:
    - Suggests hyperparameters from defined search space
    - Trains PyTorch models with suggested parameters
    - Tracks all experiments in MLflow
    - Returns best hyperparameters and model
    """

    def __init__(self, data_dir: str, config_path: str = 'configs/training_config.yaml',
                 search_space_path: str = 'configs/search_space.yaml'):
        """Initialize the optimizer with data and configuration."""
        self.data_dir = data_dir

        # Load base configuration with environment variable substitution
        self.base_config = load_config_with_env_vars(config_path)

        # Load search space configuration with environment variable substitution
        self.search_space = load_config_with_env_vars(search_space_path)

        # Setup MLflow
        base_experiment_name = self.base_config['mlflow']['experiment_name']
        mlflow.set_tracking_uri(self.base_config['mlflow']['tracking_uri'])

        # Create timestamped experiment for this optimization session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{base_experiment_name}_optimization_{timestamp}"
        mlflow.set_experiment(self.experiment_name)

        # Create dataloaders once (shared across trials)
        self.train_loader, self.val_loader, self.class_names = create_dataloaders(
            data_dir=data_dir,
            config=self.base_config
        )

        # Store base experiment name for consistent use
        self.base_experiment_name = base_experiment_name

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
                    elif param_type == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high']
                        )

        return params

    def create_optimizer(self, model: nn.Module, optimizer_name: str,
                        learning_rate: float, weight_decay: float, momentum: float) -> optim.Optimizer:
        """Create PyTorch optimizer based on hyperparameters."""
        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(model.parameters(), lr=learning_rate,
                              weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            return optim.SGD(model.parameters(), lr=learning_rate,
                           momentum=momentum,
                           weight_decay=weight_decay)
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
        trial_config['training'] = params  # Add training section from Optuna params

        # Log hyperparameters to MLflow
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)

            try:
                # Create model with trial config
                model = ResNetBirdClassifier(model_config=trial_config['model'])

                # Create custom optimizer
                optimizer = self.create_optimizer(
                    model,
                    params['optimizer'],
                    params['learning_rate'],
                    weight_decay=params['weight_decay'],
                    momentum=params['momentum']
                )

                # Create trainer with custom optimizer
                trainer = Trainer(model, trial_config)
                trainer.optimizer = optimizer

                # Create new dataloaders with the batch size for this trial
                batch_size = params['batch_size']
                trial_config = self.base_config.copy()
                trial_config['data']['batch_size'] = batch_size
                train_loader, val_loader, _ = create_dataloaders(
                    data_dir=self.data_dir,
                    config=trial_config
                )
                trainer.train_loader = train_loader
                trainer.val_loader = val_loader

                # Use epochs from Optuna-suggested parameters
                max_epochs = params['epochs']

                best_val_f1 = 0.0
                patience_counter = 0
                patience = params['early_stopping_patience']

                for epoch in range(max_epochs):
                    # Train
                    train_metrics = trainer.train_epoch(trainer.train_loader)
                    trainer.log_metrics(train_metrics, epoch, 'train')

                    # Validate
                    val_metrics = trainer.validate_epoch(trainer.val_loader)
                    trainer.log_metrics(val_metrics, epoch, 'val')

                    # Track best validation F1
                    current_val_f1 = val_metrics['f1_score']
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

    def optimize(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of optimization trials

        Returns:
            Dictionary containing best hyperparameters and study results
        """
        # Get optimization settings from config
        optimization_config = self.base_config.get('optimization', {})
        study_name = optimization_config.get('study_name', 'bird_classifier_optimization')
        direction = optimization_config.get('direction', 'maximize')
        sampler_type = optimization_config.get('sampler', 'TPE')
        pruner_type = optimization_config.get('pruner', 'Median')

        # Create sampler based on config
        if sampler_type == 'TPE':
            sampler = optuna.samplers.TPESampler(seed=42)
        else:
            raise ValueError(f"Unsupported sampler type: {sampler_type}")

        # Create pruner based on config
        if pruner_type == 'Median':
            pruner = optuna.pruners.MedianPruner()
        else:
            raise ValueError(f"Unsupported pruner type: {pruner_type}")

        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )

        print(f"Starting optimization with {n_trials} trials...")
        print(f"Study name: {study_name}")
        print(f"Direction: {direction}")
        print(f"Sampler: {sampler_type}")
        print(f"Pruner: {pruner_type}")
        print(f"MLflow experiment: {self.experiment_name}")
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
        with mlflow.start_run(run_name="optimization_summary", nested=True):
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

    def train_final_model(self, best_params: Dict[str, Any]) -> str:
        """
        Train final model with best hyperparameters.

        Args:
            best_params: Best hyperparameters from optimization

        Returns:
            Path to saved final model
        """
        # Create final config with best parameters
        final_config = self.base_config.copy()
        final_config['training'] = best_params  # Add training section from best params

        # Create and train final model with config
        model = ResNetBirdClassifier(model_config=final_config['model'])

        # Create optimizer with best parameters
        optimizer = self.create_optimizer(
            model,
            best_params['optimizer'],
            best_params['learning_rate'],
            weight_decay=best_params['weight_decay'],
            momentum=best_params['momentum']
        )

        # Create trainer
        trainer = Trainer(model, final_config)
        trainer.optimizer = optimizer

        # Create dataloaders with best batch size
        batch_size = best_params['batch_size']
        final_config = self.base_config.copy()
        final_config['data']['batch_size'] = batch_size
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=self.data_dir,
            config=final_config
        )
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader

        # Train final model
        final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_experiment_name = f"{self.base_experiment_name}_final_model_{final_timestamp}"
        mlflow.set_experiment(final_experiment_name)
        with mlflow.start_run(run_name="final_model_training"):
            mlflow.log_params(best_params)
            mlflow.log_param('training_type', 'final_model')

            trainer.train(trainer.train_loader, trainer.val_loader)

            # Register model with MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="bird_classifier"
            )

        # Save final model locally
        model_path = 'final_bird_classifier.pth'
        model.save_model(model_path)

        print(f"\nFinal model saved to: {model_path}")
        return model_path