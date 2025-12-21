import torch
import copy
import logging
from src.core.model import SimpleMLP
from src.core.strategy import FedAvg, FedProx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLServer:
    def __init__(self, input_dim, min_clients=2, max_rounds=10, strategy_name='FedAvg', config=None):
        self.global_model = SimpleMLP(input_dim)
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
        self.updates_buffer = [] # Store (client_id, parameters, num_samples)
        self.metrics_history = {"loss": [], "accuracy": []}
        
    def get_global_model_state(self):
        return self.global_model.state_dict()
        
    def register_client(self, client_id):
        if client_id not in self.registered_clients:
            self.registered_clients[client_id] = True
            logger.info(f"Client {client_id} registered. Total: {len(self.registered_clients)}")
            return True
        return False
        
    def check_start_condition(self):
        return len(self.registered_clients) >= self.min_clients

    def receive_update(self, client_id, state_dict, num_samples, metrics=None):
        """Store update and check if we can aggregate"""
        self.updates_buffer.append({
            'client_id': client_id,
            'state_dict': state_dict,
            'num_samples': num_samples,
            'metrics': metrics or {}
        })
        logger.info(f"Received update from {client_id}. Buffer size: {len(self.updates_buffer)}")
        
        if len(self.updates_buffer) >= len(self.registered_clients):
            return self.aggregate_and_step()
        return False

    def aggregate_and_step(self):
        logger.info(f"Aggregating round {self.current_round}...")
        
        # Unpack buffer
        client_models = [u['state_dict'] for u in self.updates_buffer]
        client_weights = [u['num_samples'] for u in self.updates_buffer]
        
        # Aggregate
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

        # Clear buffer
        self.updates_buffer = []

        
        # Save metrics (Placeholder for now, just saving round completion)
        # In a real system we would aggregate validation metrics here. 
        # For simulation, we assume clients trained successfully.
        
        # Save partial results every round for live dashboard monitoring
        self.save_results()
        
        if self.current_round >= self.max_rounds:
             logger.info("Max rounds reached. Training complete.")
             # save_results is already called above
             
        return True

    def save_results(self):
        import json
        import os
        os.makedirs('results', exist_ok=True)
        results = {
            "strategy": self.strategy_name,
            "rounds": self.current_round,
            "clients": self.min_clients,
            "clients": self.min_clients,
            "loss": self.metrics_history['loss'],
            "accuracy": self.metrics_history['accuracy'],
            "partition": self.config.get('partition', 'unknown'),
            "epsilon": self.config.get('epsilon', 0.0),
            "status": "complete" if self.current_round >= self.max_rounds else "running"
        }
        # Unique standardized filename for dashboard parsing
        # Format: fl_{strategy}_N{clients}_{partition}_eps{epsilon}.json
        part = self.config.get('partition', 'iid')
        eps = self.config.get('epsilon', 0.0)
        filename = f"results/fl_{self.strategy_name}_N{self.min_clients}_{part}_eps{eps}.json"
        with open(filename, 'w') as f:
            json.dump(results, f)
        logger.info(f"Results saved to {filename}")
