import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class ResNet50BirdClassifier(nn.Module):
    """
    ResNet50-based bird classifier with customizable number of classes.

    Uses pretrained ResNet50 as backbone with custom classification head.
    """

    def __init__(self, num_classes: int = 220, pretrained: bool = True, dropout: float = 0.5):
        """
        Initialize the bird classifier.

        Args:
            num_classes: Number of bird species classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate for classification head
        """
        super(ResNet50BirdClassifier, self).__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

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