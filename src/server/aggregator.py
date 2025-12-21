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

    def receive_update(self, client_id, state_dict, num_samples):
        """Store update and check if we can aggregate"""
        self.updates_buffer.append({
            'client_id': client_id,
            'state_dict': state_dict,
            'num_samples': num_samples
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
        
        # Clear buffer
        self.updates_buffer = []
        self.current_round += 1
        logger.info(f"Round {self.current_round} complete.")
        
        if self.current_round >= self.max_rounds:
             logger.info("Max rounds reached. Training complete.")
             # We could trigger a shutdown flag here
             
        return True
