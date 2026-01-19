"""
PneumoniaMNIST Training Script for Federated Learning

This script runs a federated learning client for PneumoniaMNIST image classification.
Similar to train.py but configured for CNN models and chest X-ray data.
"""

import argparse
import requests
import torch
import io
import time
import base64
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.client.client_pneumonia import FLClientPneumonia
from src.core.model import SimpleCNN
from src.data.pneumonia_dataset import PneumoniaMNISTDataset
from src.data.partitioning import DataPartitioner
from src.core.privacy import GaussianMechanism

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8001")  # Different port for pneumonia experiments


def main(args):
    """Main training loop for a single federated learning client."""
    
    # 1. Load Data & Partition
    print(f"Client {args.client_id}: Loading PneumoniaMNIST dataset...")
    full_dataset = PneumoniaMNISTDataset(split='train', download=True)
    
    # Partitioning (Deterministic based on seed)
    torch.manual_seed(42)  # Ensure consistent partitioning across clients
    partitioner = DataPartitioner(
        full_dataset, 
        num_clients=args.total_clients, 
        partition=args.partition, 
        beta=args.beta
    )
    
    # Get subset for this client
    client_indices = partitioner.use(args.client_id)
    client_dataset = torch.utils.data.Subset(full_dataset, client_indices)
    
    print(f"Client {args.client_id}: Received {len(client_dataset)} samples")
    
    # 2. Initialize Client with CNN model
    client = FLClientPneumonia(
        client_id=str(args.client_id),
        dataset=client_dataset,
        model_class=SimpleCNN,
        input_channels=1,  # Grayscale images
        num_classes=1      # Binary classification
    )
    
    # 3. Register with server
    print(f"Client {args.client_id}: Registering with {SERVER_URL}...")
    try:
        resp = requests.post(f"{SERVER_URL}/register", json={"client_id": str(args.client_id)})
        resp.raise_for_status()
        print(f"Client {args.client_id}: Registered.")
    except Exception as e:
        print(f"Failed to register: {e}")
        return

    # 4. Training Loop - Poll for round info
    last_round = -1
    
    while True:
        try:
            # Get Config from server
            config_resp = requests.get(f"{SERVER_URL}/config")
            config = config_resp.json()
            
            if config.get('finished', False):
                print(f"Client {args.client_id}: Training finished (Max rounds reached).")
                break
            
            server_round = config['round']
            
            if server_round > last_round:
                print(f"Client {args.client_id}: Starting Round {server_round}")
                
                # Fetch Global Model
                model_resp = requests.get(f"{SERVER_URL}/model")
                encoded = model_resp.json()['encoded_state']
                decoded = base64.b64decode(encoded)
                buffer = io.BytesIO(decoded)
                global_state = torch.load(buffer, weights_only=False)
                
                client.set_parameters(global_state)
                
                # Setup Privacy Engine if enabled
                privacy_engine = None
                if args.dp_epsilon > 0:
                    privacy_engine = GaussianMechanism(epsilon=args.dp_epsilon, delta=1e-5)

                # Train locally
                num_samples, metrics = client.train(config, privacy_engine=privacy_engine)
                print(f"Client {args.client_id}: Train complete. Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
                
                # Upload updated model to server
                new_state = client.get_parameters()
                buffer = io.BytesIO()
                torch.save(new_state, buffer)
                buffer.seek(0)
                
                files = {'file': buffer}
                data = {
                    'client_id': str(args.client_id), 
                    'num_samples': num_samples,
                    'loss': metrics['loss'],
                    'accuracy': metrics['accuracy'],
                    'dp_epsilon': args.dp_epsilon
                }
                
                resp = requests.post(f"{SERVER_URL}/update", data=data, files=files)
                resp.raise_for_status()
                print(f"Client {args.client_id}: Update sent.")
                
                last_round = server_round
            else:
                time.sleep(1)  # Wait for round to advance
                
        except Exception as e:
            print(f"Error in loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PneumoniaMNIST FL Client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (0-indexed)")
    parser.add_argument("--total-clients", type=int, default=2, help="Total number of clients")
    parser.add_argument("--partition", type=str, default='iid', choices=['iid', 'non-iid'], help="Data partitioning strategy")
    parser.add_argument("--beta", type=float, default=0.5, help="Dirichlet beta for non-IID")
    parser.add_argument("--dp-epsilon", type=float, default=0.0, help="DP epsilon (0 = disabled)")
    
    args = parser.parse_args()
    main(args)
