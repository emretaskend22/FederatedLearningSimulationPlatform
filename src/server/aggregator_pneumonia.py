"""
PneumoniaMNIST FL Server Aggregator

This module provides the FL server for aggregating CNN model updates
from clients training on PneumoniaMNIST data.
"""

import torch
import copy
import logging
import json
import os

from src.core.model import SimpleCNN
from src.core.strategy import FedAvg, FedProx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FLServerPneumonia:
    """
    Federated Learning Server for PneumoniaMNIST CNN models.
    
    Similar to FLServer but configured for image classification models.
    """
    
    def __init__(self, model_class, model_kwargs=None, min_clients=2, max_rounds=10, 
                 strategy_name='FedAvg', config=None):
        """
        Args:
            model_class: Model class to use (e.g., SimpleCNN)
            model_kwargs: Arguments for model initialization
            min_clients: Minimum clients required to start training
            max_rounds: Maximum number of FL rounds
            strategy_name: 'FedAvg' or 'FedProx'
            config: Additional configuration dict
        """
        model_kwargs = model_kwargs or {}
        self.global_model = model_class(**model_kwargs)
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.min_clients = min_clients
        self.max_rounds = max_rounds
        self.strategy_name = strategy_name
        self.config = config or {}
        
        # Strategy
        if strategy_name == 'FedProx':
            self.strategy = FedProx()
        else:
            self.strategy = FedAvg()
            
        self.registered_clients = {}
        self.current_round = 0
        self.updates_buffer = []
        self.metrics_history = {"loss": [], "accuracy": []}
        
        # Early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def get_global_model_state(self):
        """Return current global model state dict."""
        return self.global_model.state_dict()
        
    def register_client(self, client_id):
        """Register a new client."""
        if client_id not in self.registered_clients:
            self.registered_clients[client_id] = True
            logger.info(f"Client {client_id} registered. Total: {len(self.registered_clients)}")
            return True
        return False
        
    def check_start_condition(self):
        """Check if enough clients are registered to start training."""
        return len(self.registered_clients) >= self.min_clients

    def receive_update(self, client_id, state_dict, num_samples, metrics=None, dp_epsilon=None):
        """
        Store update and check if we can aggregate.
        
        Args:
            client_id: Client identifier
            state_dict: Model state dict from client
            num_samples: Number of samples client trained on
            metrics: Optional dict with 'loss' and 'accuracy'
            
        Returns:
            True if aggregation was performed
        """
        self.updates_buffer.append({
            'client_id': client_id,
            'state_dict': state_dict,
            'num_samples': num_samples,
            'metrics': metrics or {},
            'dp_epsilon': dp_epsilon
        })
        logger.info(f"Received update from {client_id}. Buffer size: {len(self.updates_buffer)}")
        
        if dp_epsilon is not None:
            self.config['epsilon'] = float(dp_epsilon)


        if len(self.updates_buffer) >= len(self.registered_clients):
            return self.aggregate_and_step()
        return False

    def aggregate_and_step(self):
        """Aggregate client updates and advance to next round."""
        logger.info(f"Aggregating round {self.current_round}...")
        
        # Unpack buffer
        client_models = [u['state_dict'] for u in self.updates_buffer]
        client_weights = [u['num_samples'] for u in self.updates_buffer]
        
        # Aggregate using strategy
        new_state = self.strategy.aggregate(self.global_model, client_models, client_weights)
        self.global_model.load_state_dict(new_state)
        
        # Aggregate Metrics (Weighted Average)
        total_samples = sum(u['num_samples'] for u in self.updates_buffer)
        if total_samples > 0:
            round_loss = sum(u['metrics'].get('loss', 0) * u['num_samples'] for u in self.updates_buffer) / total_samples
            round_acc = sum(u['metrics'].get('accuracy', 0) * u['num_samples'] for u in self.updates_buffer) / total_samples
        else:
            round_loss, round_acc = 0.0, 0.0
        
        self.metrics_history['loss'].append(round_loss)
        self.metrics_history['accuracy'].append(round_acc)
        
        logger.info(f"Round {self.current_round} complete. Loss: {round_loss:.4f} Acc: {round_acc:.4f}")

        # Clear buffer and advance round
        self.updates_buffer = []
        self.current_round += 1
        
        # Early Stopping Logic
        patience = 3
        min_delta = 0.001
        
        if round_loss < (self.best_loss - min_delta):
            self.best_loss = round_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            logger.info(f"Early Stopping: No improvement ({self.patience_counter}/{patience})")
            
        if self.patience_counter >= patience:
            logger.info(f"Early Stopping Triggered! Loss hasn't improved for {patience} rounds.")
            self.current_round = self.max_rounds
            self.save_results()
            return True

        # Save results every round for monitoring
        self.save_results()
        
        if self.current_round >= self.max_rounds:
            logger.info("Max rounds reached. Training complete.")
            
        return True

    def save_results(self):
        """Save experiment results to JSON file."""
        os.makedirs('results', exist_ok=True)
        results = {
            "dataset": "PneumoniaMNIST",
            "model": "SimpleCNN",
            "strategy": self.strategy_name,
            "rounds": self.current_round,
            "clients": self.min_clients,
            "loss": self.metrics_history['loss'],
            "accuracy": self.metrics_history['accuracy'],
            "partition": self.config.get('partition', 'iid'),
            "epsilon": self.config.get('epsilon', 0.0),
            "status": "complete" if self.current_round >= self.max_rounds else "running"
        }
        
        # Filename format: fl_pneumonia_{strategy}_N{clients}_{partition}_eps{epsilon}.json
        part = self.config.get('partition', 'iid')
        eps = self.config.get('epsilon', 0.0)
        filename = f"results/fl_pneumonia_{self.strategy_name}_N{self.min_clients}_{part}_eps{eps}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")
