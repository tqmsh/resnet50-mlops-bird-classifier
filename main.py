#!/usr/bin/env python3
"""
Bird Classifier - Main Entry Point

This script provides a simple interface to run the complete bird classification
pipeline with automated hyperparameter tuning using PyTorch, MLflow, and Optuna.

Usage Examples:
    # Complete automated pipeline (recommended)
    python src/main.py --data_dir /path/to/bird_dataset --optimize

    # Basic training only
    python src/main.py --data_dir /path/to/bird_dataset --basic

    # Hyperparameter optimization only
    python src/main.py --data_dir /path/to/bird_dataset --optimize_only

    # Make predictions with trained model
    python src/main.py --predict --model_path final_bird_classifier.pth --image_path test_image.jpg
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from pipeline import MLPipeline
from predict import BirdPredictor
from config_loader import load_config_with_env_vars

def main():
    parser = argparse.ArgumentParser(
        description='Bird Classification with Automated Hyperparameter Tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --optimize
  %(prog)s --predict --model_path model.pth --image_path image.jpg
        """
    )

    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to configuration file. Use configs/test_config.yaml for quick testing')

    parser.add_argument('--search_space', type=str, default='configs/search_space.yaml',
                        help='Path to hyperparameter search space file. Use configs/test_search_space.yaml for quick testing')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--optimize', action='store_true',
                           help='Run complete pipeline with optimization')
    mode_group.add_argument('--predict', action='store_true',
                           help='Run inference with trained model')

    parser.add_argument('--model_path', type=str,
                        help='Path to trained model for prediction')
    parser.add_argument('--image_path', type=str,
                        help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str,
                        help='Path to directory of images for batch prediction')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to return')

    args = parser.parse_args()

    if args.predict and not args.model_path:
        print("Error: --model_path required for prediction mode")
        sys.exit(1)

    try:
        if args.predict:
            print("Prediction Mode")
            print("-" * 30)

            if not args.image_path and not args.image_dir:
                print("Error: Please provide either --image_path or --image_dir")
                sys.exit(1)

            predictor = BirdPredictor(args.model_path, args.config)

            # Single image prediction
            if args.image_path:
                result = predictor.predict_single(args.image_path, args.top_k)
                print(f"\nPrediction for: {args.image_path}")
                print(f"Top prediction: {result['top_prediction']['class_name']} "
                      f"(confidence: {result['top_prediction']['confidence']:.4f})")

                print("\nTop predictions:")
                for i, pred in enumerate(result['predictions'], 1):
                    print(f"  {i}. {pred['class_name']}: {pred['confidence']:.4f}")

            # Batch prediction
            elif args.image_dir:
                results = predictor.predict_batch([args.image_dir])
                print(f"Processed {len(results)} images")

        else:
            print("Bird Classification Pipeline")
            print("=" * 50)

            config = load_config_with_env_vars(args.config)
            data_dir = config.get('data', {}).get('train_path')

            if not data_dir:
                print("Error: No train_path found in config")
                sys.exit(1)

            pipeline = MLPipeline(data_dir, args.config, args.search_space)

            print(f"Mode: Complete Pipeline with Optimization")
            results = pipeline.run_complete_pipeline(
                n_trials=config.get('optimization', {}).get('n_trials', 10),
                train_final_model=True,
                final_epochs=config.get('training', {}).get('epochs', 50)
            )
            print(f"\nComplete pipeline finished!")
            print(f"Best F1 Score: {results['best_value']:.4f}")
            print(f"Final Model: {results.get('final_model_path', 'N/A')}")

            print(f"\nView detailed results in MLflow UI:")
            print(f"   Tracking URI: {pipeline.config['mlflow']['tracking_uri']}")
            print(f"   Experiment: {pipeline.config['mlflow']['experiment_name']}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()