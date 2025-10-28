import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, List

def get_data_transforms(image_size: int) -> dict:
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

def create_dataloaders(data_dir: str, config: dict) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create PyTorch dataloaders for training and validation.

    Args:
        data_dir: Path to the dataset root directory (contains Train/Test subdirectories)
        config: Configuration dictionary containing all parameters

    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Extract parameters from config
    data_config = config['data']
    batch_size = data_config['batch_size']
    train_split = data_config['train_split']
    num_workers = data_config['num_workers']
    pin_memory = data_config['pin_memory']
    random_seed = data_config['random_seed']
    train_subdir = data_config['train_subdir']

    # Get data transforms
    image_size = data_config['image_size']
    transforms_dict = get_data_transforms(image_size)

    # Path to the training data containing species folders
    train_data_path = os.path.join(data_dir, train_subdir)

    # Load the full training dataset
    full_dataset = datasets.ImageFolder(
        root=train_data_path,
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
        generator=torch.Generator().manual_seed(random_seed)
    )

    # Update validation dataset transforms
    val_dataset.dataset.transform = transforms_dict['val']

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Dataset loaded successfully!")
    print(f"Total samples: {total_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names[:5]}...")  # Show first 5 classes

    return train_loader, val_loader, class_names