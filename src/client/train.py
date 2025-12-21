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

from src.client.client import FLClient
from src.core.model import SimpleMLP
from src.data.datasets import AdultDataset
from src.data.partitioning import DataPartitioner
from src.core.privacy import GaussianMechanism

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

def main(args):
    # 1. Load Data & Partition
    # In a real deployed scenario, data would be local. 
    # Here we simulate by loading all and selecting partition.
    full_dataset = AdultDataset('src/data', split='train')
    
    # Partitioning (Deterministic based on seed/args)
    # We need to know total clients to partition correctly.
    # Assuming args.total_clients is passed or fixed
    partitioner = DataPartitioner(full_dataset, num_clients=args.total_clients, partition=args.partition, beta=args.beta)
    
    # Subset
    client_indices = partitioner.use(args.client_id)
    client_dataset = torch.utils.data.Subset(full_dataset, client_indices)
    
    # 2. Initialize Client
    client = FLClient(
        client_id=str(args.client_id),
        dataset=client_dataset,
        model_class=SimpleMLP,
        input_dim=96 # matched with Adult features
    )
    
    # 3. Register
    print(f"Client {args.client_id}: Registering with {SERVER_URL}...")
    try:
        resp = requests.post(f"{SERVER_URL}/register", json={"client_id": str(args.client_id)})
        resp.raise_for_status()
        print(f"Client {args.client_id}: Registered.")
    except Exception as e:
        print(f"Failed to register: {e}")
        return

    # 4. Training Loop
    # Poll for round info
    # In this simple simulation, we assume synchronous turns.
    # ideally we use websockets or long polling. 
    # for simplicity: fetch config, if round > last_round, train.
    
    last_round = -1
    
    while True:
        try:
            # Get Config
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
                global_state = torch.load(buffer)
                
                client.set_parameters(global_state)
                
                # Setup Privacy
                privacy_engine = None
                if args.dp_epsilon > 0:
                     # Sensitivity 1.0 is default, delta 1e-5
                     privacy_engine = GaussianMechanism(epsilon=args.dp_epsilon, delta=1e-5)

                # Train
                num_samples, metrics = client.train(config, privacy_engine=privacy_engine)
                print(f"Client {args.client_id}: Train complete. Loss: {metrics['loss']:.4f}")
                
                # Upload
                new_state = client.get_parameters()
                buffer = io.BytesIO()
                torch.save(new_state, buffer)
                buffer.seek(0)
                
                files = {'file': buffer}
                data = {'client_id': str(args.client_id), 'num_samples': num_samples}
                
                resp = requests.post(f"{SERVER_URL}/update", data=data, files=files)
                resp.raise_for_status()
                print(f"Client {args.client_id}: Update sent.")
                
                last_round = server_round
            else:
                time.sleep(1) # Wait for round to advance
                
        except Exception as e:
            print(f"Error in loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--total-clients", type=int, default=2)
    parser.add_argument("--partition", type=str, default='iid')
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--dp-epsilon", type=float, default=0.0)
    
    args = parser.parse_args()
    main(args)
