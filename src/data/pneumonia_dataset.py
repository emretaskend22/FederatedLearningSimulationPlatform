"""
PneumoniaMNIST Dataset for Federated Learning

This module provides the PneumoniaMNIST dataset wrapper for use in federated learning
experiments. Uses the medmnist library for data access.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    import medmnist
    from medmnist import PneumoniaMNIST
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    print("Warning: medmnist not installed. Install with: pip install medmnist")


class PneumoniaMNISTDataset(Dataset):
    """
    PneumoniaMNIST Dataset wrapper for Federated Learning.
    
    PneumoniaMNIST contains 28x28 grayscale chest X-ray images for binary 
    classification: Normal (0) vs Pneumonia (1).
    
    Attributes:
        X: Image tensor of shape [N, 1, 28, 28]
        y: Label tensor of shape [N, 1]
    """
    
    def __init__(self, split='train', download=True, transform=None):
        """
        Args:
            split: 'train', 'val', or 'test'
            download: Whether to download the dataset if not present
            transform: Optional transform to apply to images
        """
        if not MEDMNIST_AVAILABLE:
            raise ImportError("medmnist is required. Install with: pip install medmnist")
        
        self.split = split
        self.transform = transform
        
        # Load PneumoniaMNIST using medmnist
        self.dataset = PneumoniaMNIST(
            split=split,
            download=download,
            as_rgb=False  # Keep as grayscale
        )
        
        # Extract images and labels
        # medmnist stores images as [N, 28, 28] uint8 and labels as [N, 1]
        images = self.dataset.imgs  # numpy array [N, 28, 28]
        labels = self.dataset.labels  # numpy array [N, 1]
        
        # Normalize to [0, 1] and add channel dimension
        images = images.astype(np.float32) / 255.0
        images = np.expand_dims(images, axis=1)  # [N, 1, 28, 28]
        
        self.X = torch.tensor(images, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        
        # For binary classification, squeeze labels if needed
        if self.y.dim() == 2 and self.y.size(1) == 1:
            # Keep as [N, 1] for consistency with AdultDataset
            pass
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_pneumonia_data(download=True):
    """
    Helper function to load PneumoniaMNIST train and test datasets.
    
    Returns:
        train_dataset: Training TensorDataset
        test_dataset: Test TensorDataset
    """
    train_ds = PneumoniaMNISTDataset(split='train', download=download)
    test_ds = PneumoniaMNISTDataset(split='test', download=download)
    
    from torch.utils.data import TensorDataset
    train_tensor = TensorDataset(train_ds.X, train_ds.y)
    test_tensor = TensorDataset(test_ds.X, test_ds.y)
    
    return train_tensor, test_tensor


# Quick test when run directly
if __name__ == "__main__":
    print("Testing PneumoniaMNISTDataset...")
    ds = PneumoniaMNISTDataset(split='train', download=True)
    print(f"Train samples: {len(ds)}")
    print(f"Image shape: {ds.X.shape}")
    print(f"Label shape: {ds.y.shape}")
    print(f"Label distribution: {torch.unique(ds.y, return_counts=True)}")
    
    # Test single item
    img, label = ds[0]
    print(f"Single image shape: {img.shape}, label: {label}")
