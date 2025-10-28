import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any

class ResNetBirdClassifier(nn.Module):
    """
    Dynamic ResNet-based bird classifier with configurable architecture.

    Supports different ResNet variants based on configuration:
    - resnet18, resnet34, resnet50, resnet101, resnet152
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the bird classifier with dynamic model selection.

        Args:
            model_config: Model configuration dictionary containing:
                - name: Model name (e.g., 'resnet18', 'resnet50')
                - num_classes: Number of bird species classes
                - pretrained: Whether to use ImageNet pretrained weights
                - dropout: Dropout rate for classification head
        """
        super(ResNetBirdClassifier, self).__init__()

        # Extract config parameters
        model_name = model_config['name']
        num_classes = model_config['num_classes']
        pretrained = model_config['pretrained']
        dropout = model_config['dropout']

        # Dynamic model creation based on config
        if not hasattr(models, model_name):
            raise ValueError(f"Unsupported model: {model_name}. Available: {self._get_available_models()}")

        # Get the model constructor and weights class
        model_constructor = getattr(models, model_name)
        weights_enum = self._get_weights_enum(model_name, pretrained)

        # Load the specified model
        self.backbone = model_constructor(weights=weights_enum if pretrained else None)
        self.model_name = model_name

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features

        # Replace the final classification layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )

    def _get_available_models(self) -> list:
        """Get list of available ResNet models."""
        return ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

    def _get_weights_enum(self, model_name: str, pretrained: bool):
        """Get the appropriate weights enum for the model."""
        if not pretrained:
            return None

        weights_map = {
            'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
            'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
            'resnet50': models.ResNet50_Weights.IMAGENET1K_V2,
            'resnet101': models.ResNet101_Weights.IMAGENET1K_V2,
            'resnet152': models.ResNet152_Weights.IMAGENET1K_V2,
        }

        return weights_map.get(model_name, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.backbone(x)

    def save_model(self, path: str):
        """Save model state dict."""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str, device: Optional[str] = None):
        """Load model state dict."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.load_state_dict(torch.load(path, map_location=device))
        return self.to(device)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            'model_name': self.model_name,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }