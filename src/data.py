import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, List

def get_data_transforms(image_size: int = 224) -> dict:
    """
    Get data transforms for training and validation.

    Args:
        image_size: Target image size for resizing

    Returns:
        Dictionary containing train and val transforms
    """
    # ImageNet normalization constants
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }

def create_dataloaders(data_dir: str, batch_size: int = 32, train_split: float = 0.8,
                      image_size: int = 224, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create PyTorch dataloaders for training and validation.

    Args:
        data_dir: Path to the training data directory
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        image_size: Target image size for resizing
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Get data transforms
    transforms_dict = get_data_transforms(image_size)

    # Load the full dataset
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transforms_dict['train']
    )

    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    # Split dataset
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Update validation dataset transforms
    val_dataset.dataset.transform = transforms_dict['val']

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset loaded successfully!")
    print(f"Total samples: {total_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names[:5]}...")  # Show first 5 classes

    return train_loader, val_loader, class_names