import os
from typing import Dict, Any

from src.optimize import HyperparameterOptimizer
from src.config_loader import load_config_with_env_vars

class MLPipeline:
    def __init__(self, data_dir: str, config_path: str = 'configs/training_config.yaml', search_space_path: str = 'configs/search_space.yaml'):
        self.data_dir = data_dir
        self.config_path = config_path
        self.search_space_path = search_space_path

        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")

        self.config = load_config_with_env_vars(config_path)

    def run_hyperparameter_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        optimizer = HyperparameterOptimizer(
            data_dir=self.data_dir,
            config_path=self.config_path,
            search_space_path=self.search_space_path
        )
        return optimizer.optimize(n_trials=n_trials)

    def run_complete_pipeline(self, n_trials: int = 50,
                            train_final_model: bool = True) -> Dict[str, Any]:
        optimizer = HyperparameterOptimizer(
            data_dir=self.data_dir,
            config_path=self.config_path,
            search_space_path=self.search_space_path
        )
        optimization_results = optimizer.optimize(n_trials)

        if train_final_model:
            final_model_path = optimizer.train_final_model(
                optimization_results['best_params']
            )
            optimization_results['final_model_path'] = final_model_path

        return optimization_results

