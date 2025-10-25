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
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from pipeline import MLPipeline
from predict import BirdPredictor

def main():
    parser = argparse.ArgumentParser(
        description='Bird Classification with Automated Hyperparameter Tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data_dir /path/to/data --optimize
  %(prog)s --data_dir /path/to/data --basic
  %(prog)s --predict --model_path model.pth --image_path image.jpg
        """
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        help='Path to training data directory')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--basic', action='store_true',
                           help='Run basic training without optimization')
    mode_group.add_argument('--optimize_only', action='store_true',
                           help='Run hyperparameter optimization only')
    mode_group.add_argument('--optimize', action='store_true',
                           help='Run complete pipeline with optimization (recommended)')
    mode_group.add_argument('--predict', action='store_true',
                           help='Run inference with trained model')

    # Prediction arguments
    parser.add_argument('--model_path', type=str,
                        help='Path to trained model for prediction')
    parser.add_argument('--image_path', type=str,
                        help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str,
                        help='Path to directory of images for batch prediction')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to return')

    # Optimization arguments
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of hyperparameter optimization trials')
    parser.add_argument('--final_epochs', type=int,
                        help='Number of epochs for final model training')

    args = parser.parse_args()

    # Validate arguments
    if args.predict and not args.model_path:
        print("Error: --model_path required for prediction mode")
        sys.exit(1)

    if not args.predict and not args.data_dir:
        print("Error: --data_dir required for training modes")
        sys.exit(1)

    try:
        # Prediction mode
        if args.predict:
            print("🔮 Prediction Mode")
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

        # Training modes
        else:
            print("🚀 Bird Classification Pipeline")
            print("=" * 50)

            # Create pipeline
            pipeline = MLPipeline(args.data_dir, args.config)

            # Run based on mode
            if args.basic:
                print("Mode: Basic Training")
                model_path = pipeline.run_basic_training()
                print(f"\n✅ Training completed! Model saved: {model_path}")

            elif args.optimize_only:
                print(f"Mode: Hyperparameter Optimization ({args.n_trials} trials)")
                results = pipeline.run_hyperparameter_optimization(args.n_trials)
                print(f"\n✅ Optimization completed! Best F1: {results['best_value']:.4f}")

            elif args.optimize:
                print(f"Mode: Complete Pipeline with Optimization ({args.n_trials} trials)")
                results = pipeline.run_complete_pipeline(
                    n_trials=args.n_trials,
                    train_final_model=True,
                    final_epochs=args.final_epochs
                )
                print(f"\n✅ Complete pipeline finished!")
                print(f"Best F1 Score: {results['best_value']:.4f}")
                print(f"Final Model: {results.get('final_model_path', 'N/A')}")

            print(f"\n📊 View detailed results in MLflow UI:")
            print(f"   Tracking URI: {pipeline.config['mlflow']['tracking_uri']}")
            print(f"   Experiment: {pipeline.config['mlflow']['experiment_name']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()