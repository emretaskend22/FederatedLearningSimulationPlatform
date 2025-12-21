import torch
import copy

class AggregationStrategy:
    def aggregate(self, global_model, client_models, client_weights):
        raise NotImplementedError

class FedAvg(AggregationStrategy):
    def aggregate(self, global_model, client_models, client_weights):
        """
        Args:
            global_model: The current global model (nn.Module)
            client_models: List of client models (nn.Module) or state_dicts
            client_weights: List of weights (e.g. number of samples) for each client
        Returns:
            Averaged state_dict
        """
        # Normalize weights
        total_weight = sum(client_weights)
        norm_weights = [w / total_weight for w in client_weights]
        
        # Create weighted average
        new_state_dict = copy.deepcopy(global_model.state_dict())
        
        for key in new_state_dict.keys():
            # Initialize with 0
            new_state_dict[key] = torch.zeros_like(new_state_dict[key])
            
            for i, client_model in enumerate(client_models):
                # Handle if client_model is module or state_dict
                if isinstance(client_model, torch.nn.Module):
                     client_state = client_model.state_dict()
                else:
                     client_state = client_model
                
                # Check data type consistency
                if new_state_dict[key].dtype != client_state[key].dtype:
                    # Cast client param to global param type (usually float32)
                    client_param = client_state[key].to(new_state_dict[key].dtype)
                else:
                    client_param = client_state[key]

                new_state_dict[key] += client_param * norm_weights[i]
                
        return new_state_dict

class FedProx(FedAvg):
    """
    FedProx aggregation is identical to FedAvg. 
    The difference is in the local training objective (proximal term).
    We inherit aggregate from FedAvg.
    """
    pass
