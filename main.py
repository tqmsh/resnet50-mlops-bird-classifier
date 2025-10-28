#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from pipeline import MLPipeline
from predict import BirdPredictor
from config_loader import load_config_with_env_vars

def main():
    parser = argparse.ArgumentParser(description='Bird Classification Pipeline')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--search_space', type=str, default='configs/search_space.yaml')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--optimize', action='store_true')
    mode_group.add_argument('--predict', action='store_true')

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--top_k', type=int, default=5)

    args = parser.parse_args()

    if args.predict and not args.model_path:
        print("Error: --model_path required for prediction")
        sys.exit(1)

    try:
        if args.predict:
            if not args.image_path:
                print("Error: --image_path required")
                sys.exit(1)

            predictor = BirdPredictor(args.model_path, args.config)
            result = predictor.predict_single(args.image_path, args.top_k)

            print(f"Prediction for: {args.image_path}")
            print(f"Top: {result['top_prediction']['class_name']} "
                  f"({result['top_prediction']['confidence']:.4f})")

        else:
            config = load_config_with_env_vars(args.config)
            data_dir = config.get('data', {}).get('train_path')

            if not data_dir:
                print("Error: No train_path found in config")
                sys.exit(1)

            pipeline = MLPipeline(data_dir, args.config, args.search_space)
            results = pipeline.run_complete_pipeline(
                n_trials=config.get('optimization', {}).get('n_trials', 10),
                train_final_model=True
            )

            print(f"Best F1 Score: {results['best_value']:.4f}")
            print(f"Final Model: {results.get('final_model_path', 'N/A')}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()