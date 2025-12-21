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

        # Calculate Training Accuracy (approximate on last epoch) for metrics
        # Ideally we run a separate eval on local test set, but for training plots we can use this.
        # Or better, let's just make a quick pass for accuracy on training set or return separate val accuracy.
        # Let's use the evaluate method which is cleaner.
        val_loss, val_acc = self.evaluate() # Evaluate on local data

        if privacy_engine:
             with torch.no_grad():
                 for param in self.model.parameters():
                     privacy_engine.apply(param)

        return len(self.dataset), {"loss": val_loss, "accuracy": val_acc}

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
