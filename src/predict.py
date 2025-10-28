import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from config_loader import load_config_with_env_vars
import argparse
from typing import List, Tuple, Dict, Any
import json

class BirdPredictor:
    """
    PyTorch-compatible inference class for bird species classification.

    Provides easy-to-use prediction functionality with support for
    single images and batch processing.
    """

    def __init__(self, model_path: str, config_path: str = 'configs/training_config.yaml'):
        """Initialize the predictor with trained model and config."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load configuration with environment variable substitution
        self.config = load_config_with_env_vars(config_path)

        # Setup transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((self.config['data']['image_size'],
                             self.config['data']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Class names (placeholder - should be saved during training)
        self.class_names = [f"class_{i}" for i in range(self.config['model']['num_classes'])]

    def _load_model(self, model_path: str):
        """Load PyTorch model from checkpoint."""
        from model import ResNetBirdClassifier

        model = ResNetBirdClassifier(model_config=self.config['model'])

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)

        return model

    def predict_single(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Predict bird species for a single image.

        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            Dictionary containing predictions and confidence scores
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)

        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            class_name = self.class_names[class_idx]

            results.append({
                'class_id': class_idx,
                'class_name': class_name,
                'confidence': prob
            })

        return {
            'image_path': image_path,
            'predictions': results,
            'top_prediction': results[0]
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict bird species for multiple images.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of prediction dictionaries
        """
        results = []

        for image_path in image_paths:
            try:
                prediction = self.predict_single(image_path)
                results.append(prediction)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })

        return results

    def predict_with_threshold(self, image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict with confidence threshold filtering.

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for valid prediction

        Returns:
            Prediction dictionary with threshold filtering
        """
        prediction = self.predict_single(image_path)

        # Filter predictions by threshold
        valid_predictions = [
            p for p in prediction['predictions']
            if p['confidence'] >= confidence_threshold
        ]

        if not valid_predictions:
            return {
                'image_path': image_path,
                'predictions': [],
                'top_prediction': None,
                'message': f'No predictions above confidence threshold {confidence_threshold}'
            }

        prediction['predictions'] = valid_predictions
        prediction['top_prediction'] = valid_predictions[0]

        return prediction

def main():
    parser = argparse.ArgumentParser(description='Bird Species Classification Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str,
                        help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str,
                        help='Path to directory containing images')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to return')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Confidence threshold for filtering predictions')
    parser.add_argument('--output', type=str,
                        help='Output JSON file for batch predictions')

    args = parser.parse_args()

    if not args.image_path and not args.image_dir:
        print("Error: Please provide either --image_path or --image_dir")
        return

    # Initialize predictor
    predictor = BirdPredictor(args.model_path, args.config)

    # Single image prediction
    if args.image_path:
        if args.threshold > 0:
            result = predictor.predict_with_threshold(args.image_path, args.threshold)
        else:
            result = predictor.predict_single(args.image_path, args.top_k)

        print(f"Prediction for {args.image_path}:")
        print(f"Top prediction: {result['top_prediction']['class_name']} "
              f"(confidence: {result['top_prediction']['confidence']:.4f})")

        print("\nTop predictions:")
        for pred in result['predictions']:
            print(f"  {pred['class_name']}: {pred['confidence']:.4f}")

    # Batch prediction
    elif args.image_dir:
        import os
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(image_extensions)
        ]

        print(f"Processing {len(image_paths)} images...")
        results = predictor.predict_batch(image_paths)

        # Print summary
        successful_predictions = [r for r in results if 'error' not in r]
        failed_predictions = [r for r in results if 'error' in r]

        print(f"Successfully processed: {len(successful_predictions)}")
        print(f"Failed: {len(failed_predictions)}")

        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()