import os
import sys
import argparse
from typing import Dict, Any

from src.optimize import HyperparameterOptimizer
from src.model import ResNetBirdClassifier
from src.data import create_dataloaders
from src.train import Trainer
from src.config_loader import load_config_with_env_vars

class MLPipeline:
    """
    Complete MLOps pipeline combining MLflow experiment tracking with Optuna
    hyperparameter optimization for PyTorch bird classification.

    This pipeline provides:
    - Automated hyperparameter tuning
    - Comprehensive experiment tracking
    - Best model selection and training
    - Complete MLflow integration
    """

    def __init__(self, data_dir: str, config_path: str = 'configs/training_config.yaml', search_space_path: str = 'configs/search_space.yaml'):
        """Initialize the MLOps pipeline."""
        self.data_dir = data_dir
        self.config_path = config_path
        self.search_space_path = search_space_path

        # Validate data directory
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")

        # Load configuration with environment variable substitution
        self.config = load_config_with_env_vars(config_path)

        print("=" * 60)
        print("BIRD CLASSIFIER MLOPS PIPELINE")
        print("=" * 60)
        print(f"Data directory: {data_dir}")
        print(f"Configuration: {config_path}")
        print(f"Number of classes: {self.config['model']['num_classes']}")
        print(f"MLflow tracking: {self.config['mlflow']['tracking_uri']}")
        print("=" * 60)

    def run_hyperparameter_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Run automated hyperparameter optimization.

        Args:
            n_trials: Number of optimization trials

        Returns:
            Dictionary containing optimization results
        """
        print(f"\nStarting Hyperparameter Optimization...")
        print(f"Number of trials: {n_trials}")
        print("-" * 40)

        print(f"Search space: {self.search_space_path}")
        print("-" * 50)

        # Create optimizer
        optimizer = HyperparameterOptimizer(
            data_dir=self.data_dir,
            config_path=self.config_path,
            search_space_path=self.search_space_path
        )

        # Run optimization
        results = optimizer.optimize(n_trials=n_trials)

        print(f"\nHyperparameter optimization completed!")
        print(f"Best F1 score: {results['best_value']:.4f}")
        print(f"Total trials: {results['n_trials']}")

        return results

    def run_complete_pipeline(self, n_trials: int = 50,
                            train_final_model: bool = True,
                            final_epochs: int = None) -> Dict[str, Any]:
        """
        Run the complete MLOps pipeline:
        1. Hyperparameter optimization
        2. Final model training with best parameters

        Args:
            n_trials: Number of optimization trials
            train_final_model: Whether to train final model
            final_epochs: Number of epochs for final training

        Returns:
            Dictionary containing complete pipeline results
        """
        print("\nStarting Complete MLOps Pipeline...")
        print("=" * 60)

        # Step 1: Hyperparameter optimization
        optimizer = HyperparameterOptimizer(
            data_dir=self.data_dir,
            config_path=self.config_path,
            search_space_path=self.search_space_path
        )
        optimization_results = optimizer.optimize(n_trials)

        # Step 2: Train final model with best parameters
        if train_final_model:
            print(f"\nTraining Final Model with Best Hyperparameters...")
            print("-" * 40)

            final_model_path = optimizer.train_final_model(
                optimization_results['best_params']
            )

            optimization_results['final_model_path'] = final_model_path

        print(f"\nComplete MLOps Pipeline Finished!")
        print("=" * 60)
        print("Results:")
        print(f"  Best F1 Score: {optimization_results['best_value']:.4f}")
        print(f"  Optimization Trials: {optimization_results['n_trials']}")
        if train_final_model:
            print(f"  Final Model: {optimization_results['final_model_path']}")
        print(f"  MLflow Experiment: {self.config['mlflow']['experiment_name']}")
        print("=" * 60)

        return optimization_results

    def evaluate_model(self, model_path: str) -> Dict[str, Any]:
        """
        Evaluate a trained model on validation set.

        Args:
            model_path: Path to trained model

        Returns:
            Evaluation metrics
        """
        print(f"\nEvaluating Model: {model_path}")
        print("-" * 40)

        # Load model with config
        model = ResNetBirdClassifier(model_config=self.config['model'])
        model.load_model(model_path)

        # Create dataloaders
        _, val_loader, class_names = create_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.config['data']['batch_size'],
            train_split=self.config['data']['train_split'],
            image_size=self.config['data']['image_size']
        )

        # Create trainer for evaluation
        trainer = Trainer(model, self.config)

        # Evaluate
        val_metrics = trainer.validate_epoch(val_loader)

        print(f"Evaluation Results:")
        for metric_name, metric_value in val_metrics.items():
            if metric_name != 'confusion_matrix':
                print(f"  {metric_name}: {metric_value:.4f}")

        return val_metrics

def main():
    parser = argparse.ArgumentParser(description='Bird Classifier MLOps Pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--search_space', type=str, default='configs/search_space.yaml',
                        help='Path to hyperparameter search space file')
    parser.add_argument('--mode', type=str, default='complete',
                        choices=['basic', 'optimize', 'complete', 'evaluate'],
                        help='Pipeline mode to run')
    parser.add_argument('--model_path', type=str,
                        help='Path to model for evaluation')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--final_epochs', type=int,
                        help='Number of epochs for final model training')
    parser.add_argument('--no_final_training', action='store_true',
                        help='Skip final model training in complete mode')

    args = parser.parse_args()

    try:
        # Create pipeline
        pipeline = MLPipeline(args.data_dir, args.config, args.search_space)

        # Run based on mode
        if args.mode == 'basic':
            model_path = pipeline.run_basic_training()

        elif args.mode == 'optimize':
            results = pipeline.run_hyperparameter_optimization(args.n_trials)

        elif args.mode == 'complete':
            results = pipeline.run_complete_pipeline(
                n_trials=args.n_trials,
                train_final_model=not args.no_final_training,
                final_epochs=args.final_epochs
            )

        elif args.mode == 'evaluate':
            if not args.model_path:
                print("Error: --model_path required for evaluation mode")
                sys.exit(1)
            results = pipeline.evaluate_model(args.model_path)

        print(f"\nPipeline completed successfully!")
        print("Check MLflow UI for detailed experiment tracking and visualizations.")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()