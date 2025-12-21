#!/usr/bin/env python3
import yaml
import argparse
import os

def generate_compose(num_clients, strategy='FedAvg', dp_epsilon=0.0, partition='iid', rounds=20):
    services = {}
    
    # Server Service
    services['server'] = {
        'build': '.',
        'image': 'fl-platform:latest', # Use local build
        'command': 'uvicorn src.server.app:app --host 0.0.0.0 --port 8000',
        'ports': ['8000:8000'],
        'environment': {
            'MIN_CLIENTS': str(num_clients),
            'MAX_ROUNDS': str(rounds),
            'STRATEGY': strategy,
            'PARTITION': partition,
            'DP_EPSILON': str(dp_epsilon)
        },
        'volumes': [
            './results:/app/results'
        ],
        'networks': ['fl-net']
    }
    
    # Client Services
    for i in range(num_clients):
        services[f'client_{i}'] = {
            'image': 'fl-platform:latest',
            'depends_on': ['server'],
            'command': f'python src/client/train.py --client-id {i} --total-clients {num_clients} --partition {partition} --dp-epsilon {dp_epsilon}',
            'environment': {
                'SERVER_URL': 'http://server:8000'
            },
            'networks': ['fl-net']
        }
        
    compose_data = {
        'version': '3.8',
        'services': services,
        'networks': {
            'fl-net': {'driver': 'bridge'}
        }
    }
    
    with open('docker-compose.yml', 'w') as f:
        yaml.dump(compose_data, f, sort_keys=False)
    
    print(f"Generated docker-compose.yml for {num_clients} clients, strategy={strategy}, dp={dp_epsilon}, rounds={rounds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--strategy", type=str, default='FedAvg')
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--partition", type=str, default='iid')
    parser.add_argument("--rounds", type=int, default=20)
    
    args = parser.parse_args()
    generate_compose(args.clients, args.strategy, args.epsilon, args.partition, args.rounds)
