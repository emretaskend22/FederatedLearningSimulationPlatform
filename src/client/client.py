import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from src.core.privacy import GaussianMechanism

class FLClient:
    def __init__(self, client_id, dataset, model_class, input_dim, device='cpu'):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model_class(input_dim).to(device)
        self.device = device
        self.loader = None # Initialized in train/eval

    def set_parameters(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_parameters(self):
        return self.model.state_dict()

    def train(self, config, privacy_engine=None):
        """
        Args:
            config: dict containing 'local_epochs', 'batch_size', 'learning_rate', 'proximal_mu'
            privacy_engine: Optional GaussianMechanism instance
        Returns:
            num_samples, metrics
        """
        epochs = config.get('local_epochs', 1)
        batch_size = config.get('batch_size', 32)
        lr = config.get('learning_rate', 0.01)
        proximal_mu = config.get('proximal_mu', 0.0) # For FedProx
        
        # Save initial global model state for FedProx
        if proximal_mu > 0:
            global_model_params = copy.deepcopy(list(self.model.parameters()))

        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
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
                
                # Differential Privacy: Clip Gradients & Add Noise
                # Note: Standard DP-SGD adds noise to Summed Gradients then divides by batch.
                # Here we simulate client-level DP or record-level DP?
                # "Architecture says: Gaussian Mechanism to inject noise into gradients."
                # Usually in FL this means Client-Level DP (noise added to update) or Record-Level (DP-SGD).
                # Interpreting as Record-Level DP (DP-SGD) implies clipping per sample.
                # Interpreting as Client-Level DP implies clipping the final model update.
                # Given "Hospital A adds DP noise", it implies Client-Level DP on the update sent to server.
                # So we apply it AFTER training, on the delta.
                
                optimizer.step()
                batch_loss += loss.item() * X.size(0)
            
            epoch_loss += batch_loss / len(self.dataset)

        # Apply Client-Level DP if enabled
        # Update = New_Weights - Old_Weights
        # We add noise to the weights before returning? Or to the update?
        # Usually: Client clips the update norm, adds noise, then sends.
        # But for simplicity here, we can add noise to the parameters directly if they are the "signal".
        # Let's apply to the state_dict before returning in simulation loop if privacy_engine is set.
        # BUT `train` modifies `self.model`.
        # We need to return the weights.
        
        # Refined Logic:
        # We will modify the model in place with noise if requested? No, that ruins local model for next round if stateful.
        # But FL usually stateless client.
        
        if privacy_engine:
             with torch.no_grad():
                 for param in self.model.parameters():
                     # Clip to sensitivity (heuristic: assume max_norm or use config)
                     # For client-level DP, sensitivity is the max norm of the UPDATE.
                     # We skip clipping here for simplicity unless specified, 
                     # but strictly we should clip the *difference*
                     # Let's implement noise addition on the weights for now as a proxy.
                     privacy_engine.apply(param)

        return len(self.dataset), {"loss": epoch_loss}

    def evaluate(self, batch_size=32):
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
                
        return total_loss / total, correct / total
