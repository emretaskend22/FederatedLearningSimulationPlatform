import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class SimpleCNN(nn.Module):
    """
    Simple CNN for 28x28 grayscale image classification (PneumoniaMNIST).
    
    Architecture:
        Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2)
        Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
        Flatten -> FC(64*5*5, 128) -> ReLU -> FC(128, 1) -> Sigmoid
    
    Output: Binary classification probability
    """
    
    def __init__(self, input_channels=1, num_classes=1):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After conv1 + pool: 28 -> 14
        # After conv2 + pool: 14 -> 7
        # So feature map is 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x


def get_model(model_name, **kwargs):
    """
    Factory function to get model by name.
    
    Args:
        model_name: 'SimpleMLP' or 'SimpleCNN'
        **kwargs: Model-specific arguments
        
    Returns:
        nn.Module instance
    """
    if model_name == 'SimpleMLP':
        return SimpleMLP(**kwargs)
    elif model_name == 'SimpleCNN':
        return SimpleCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

