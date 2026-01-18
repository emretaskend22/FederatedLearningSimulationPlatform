"""
PneumoniaMNIST FL Client - Extends base FLClient for image classification.

This module provides a specialized FL client for training CNN models on
PneumoniaMNIST chest X-ray images in a federated learning setting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

import sys
import os
sys.path.append(os.getcwd())

from src.core.privacy import GaussianMechanism


class FLClientPneumonia:
    """
    Federated Learning Client for PneumoniaMNIST image classification.
    
    Similar to FLClient but optimized for CNN models and image data.
    """
    
    def __init__(self, client_id, dataset, model_class, device='cpu', **model_kwargs):
        """
        Args:
            client_id: Unique client identifier
            dataset: PyTorch Dataset with image data
            model_class: CNN model class (e.g., SimpleCNN)
            device: 'cpu' or 'cuda'
            **model_kwargs: Arguments passed to model constructor
        """
        self.client_id = client_id
        self.dataset = dataset
        self.model = model_class(**model_kwargs).to(device)
        self.device = device
        self.loader = None

    def set_parameters(self, state_dict):
        """Load global model parameters."""
        self.model.load_state_dict(state_dict)

    def get_parameters(self):
        """Get current model parameters."""
        return self.model.state_dict()

    def train(self, config, privacy_engine=None):
        """
        Train the model on local data.
        
        Args:
            config: dict containing 'local_epochs', 'batch_size', 'learning_rate', 'proximal_mu'
            privacy_engine: Optional GaussianMechanism instance
            
        Returns:
            num_samples: Number of training samples
            metrics: dict with 'loss' and 'accuracy'
        """
        epochs = config.get('local_epochs', 1)
        batch_size = config.get('batch_size', 32)
        lr = config.get('learning_rate', 0.01)
        proximal_mu = config.get('proximal_mu', 0.0)  # For FedProx
        
        # Save initial global model state for FedProx
        if proximal_mu > 0:
            global_model_params = copy.deepcopy(list(self.model.parameters()))

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.BCELoss()  # Binary classification
        optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Adam often better for CNNs
        
        self.model.train()
        epoch_loss = 0.0
        
        for epoch in range(epochs):
            batch_loss = 0.0
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                output = self.model(X)
                loss = criterion(output, y)
                
                # FedProx Proximal Term
                if proximal_mu > 0:
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_model_params):
                        prox_term += (param - global_param).norm(2) ** 2
                    loss += (proximal_mu / 2) * prox_term

                loss.backward()
                optimizer.step()
                batch_loss += loss.item() * X.size(0)
            
            epoch_loss += batch_loss / len(self.dataset)

        # Evaluate on local data
        val_loss, val_acc = self.evaluate()

        # Apply differential privacy if enabled
        if privacy_engine:
            with torch.no_grad():
                for param in self.model.parameters():
                    privacy_engine.apply(param)

        return len(self.dataset), {"loss": val_loss, "accuracy": val_acc}

    def evaluate(self, batch_size=32):
        """
        Evaluate the model on local dataset.
        
        Returns:
            loss: Average loss
            accuracy: Classification accuracy
        """
        dataloader = DataLoader(self.dataset, batch_size=batch_size)
        criterion = nn.BCELoss()
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
                
                predicted = (outputs > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
                
        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0
