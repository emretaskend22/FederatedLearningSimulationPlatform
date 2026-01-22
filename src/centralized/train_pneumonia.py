#!/usr/bin/env python3
"""
Centralized Training for PneumoniaMNIST

This script trains the CNN model on the full PneumoniaMNIST dataset in a
centralized (non-federated) manner to establish a baseline for comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.core.model import SimpleCNN
from src.data.pneumonia_dataset import PneumoniaMNISTDataset


def train_centralized(epochs=20, batch_size=32, lr=0.001):
    """
    Train CNN on full PneumoniaMNIST dataset (centralized baseline).
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """
    print("Loading PneumoniaMNIST data...")
    train_ds = PneumoniaMNISTDataset(split='train', download=True)
    test_ds = PneumoniaMNISTDataset(split='test', download=True)
    
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    model = SimpleCNN(input_channels=1, num_classes=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    metrics = {"loss": [], "accuracy": []}
    
    print("Starting centralized training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for X, y in loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)
        
        epoch_loss /= len(train_ds)
        metrics["loss"].append(epoch_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                out = model(X)
                predicted = (out > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        metrics["accuracy"].append(acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {acc:.4f}")

    # Save results with pneumonia prefix
    os.makedirs('results', exist_ok=True)
    with open('results/centralized_pneumonia_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Results saved to results/centralized_pneumonia_metrics.json")
    
    return metrics


if __name__ == "__main__":
    train_centralized()
