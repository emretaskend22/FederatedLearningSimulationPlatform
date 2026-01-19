import torch
import numpy as np

class GaussianMechanism:
    def __init__(self, epsilon, delta, sensitivity=0.1):
        """
        Args:
            epsilon: Privacy budget
            delta: Probability of failure
            sensitivity: Sensitivity of the function (usually clipped grad norm)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # Calibrate noise scale (sigma)
        # Using analytic gaussian mechanism approximation or standard formula
        # Sigma = sqrt(2 * log(1.25/delta)) * sensitivity / epsilon
        if epsilon > 0:
             self.sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        else:
             self.sigma = 0.0

    def apply(self, tensor):
        """Add Gaussian noise to tensor"""
        if self.sigma == 0:
            return tensor
            
        noise = torch.normal(0, self.sigma, size=tensor.shape, device=tensor.device)
        return tensor + noise

    def clip(self, tensor, max_norm):
        """Clip tensor by norm"""
        norm = torch.norm(tensor)
        if norm > max_norm:
            return tensor * (max_norm / norm)
        return tensor
