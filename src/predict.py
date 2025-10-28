import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.config_loader import load_config_with_env_vars
from typing import Dict, Any

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

