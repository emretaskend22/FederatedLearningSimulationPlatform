#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import json
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.core.model import SimpleMLP
from src.data.datasets import load_data, AdultDataset

def train_centralized(epochs=20, batch_size=32, lr=0.01):
    print("Loading data...")
    # Use load_data helper to ensure consistent feature encoding across train/test
    train_ds, test_ds = load_data('src/data')
    
    # In centralized, we train on all training data
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Get dimension from data
    input_dim = train_ds[0][0].shape[0]
    print(f"Data Input Dimension: {input_dim}")
    
    model = SimpleMLP(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    metrics = {"loss": [], "accuracy": []}
    
    print("Starting training...")
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
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in DataLoader(test_ds, batch_size=batch_size):
                out = model(X)
                predicted = (out > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        metrics["accuracy"].append(acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {acc:.4f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/centralized_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Results saved to results/centralized_metrics.json")

if __name__ == "__main__":
    train_centralized()
