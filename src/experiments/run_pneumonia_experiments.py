#!/usr/bin/env python3
"""
PneumoniaMNIST Experiment Runner

This script runs a grid of federated learning experiments for PneumoniaMNIST:
- Different client counts (5, 10, 20)
- Different strategies (FedAvg, FedProx)
- Different data partitions (IID, non-IID)

Saves results in the same format as the Adult dataset experiments for
dashboard compatibility.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import json
import os
import sys
import copy

# Add project root to path
sys.path.append(os.getcwd())

from src.core.model import SimpleCNN
from src.data.pneumonia_dataset import PneumoniaMNISTDataset
from src.data.partitioning import DataPartitioner
from src.core.strategy import FedAvg, FedProx


def local_train(model, dataset, config, global_params=None):
    """
    Train model locally on client data.
    
    Args:
        model: CNN model
        dataset: Client's data subset
        config: Training config dict
        global_params: For FedProx, the global model parameters
        
    Returns:
        num_samples, metrics dict
    """
    epochs = config.get('local_epochs', 1)
    batch_size = config.get('batch_size', 32)
    lr = config.get('learning_rate', 0.001)
    proximal_mu = config.get('proximal_mu', 0.0)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            
            # FedProx proximal term
            if proximal_mu > 0 and global_params:
                prox_term = 0.0
                for param, global_param in zip(model.parameters(), global_params):
                    prox_term += (param - global_param).norm(2) ** 2
                loss += (proximal_mu / 2) * prox_term
            
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            predicted = (out > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    return len(dataset), {
        "loss": total_loss / total if total > 0 else 0,
        "accuracy": correct / total if total > 0 else 0
    }


def run_fl_experiment(num_clients, strategy_name, partition, rounds=20, local_epochs=1):
    """
    Run a single FL experiment.
    
    Args:
        num_clients: Number of FL clients
        strategy_name: 'FedAvg' or 'FedProx'
        partition: 'iid' or 'non-iid'
        rounds: Number of FL rounds
        local_epochs: Local training epochs per round
        
    Returns:
        Results dict
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {strategy_name} | N={num_clients} | {partition}")
    print(f"{'='*60}")
    
    # Load full dataset
    full_dataset = PneumoniaMNISTDataset(split='train', download=True)
    test_dataset = PneumoniaMNISTDataset(split='test', download=True)
    
    # Partition data
    torch.manual_seed(42)
    partitioner = DataPartitioner(
        full_dataset,
        num_clients=num_clients,
        partition=partition,
        beta=0.5
    )
    
    # Create client datasets
    client_datasets = []
    for i in range(num_clients):
        indices = partitioner.use(i)
        client_datasets.append(Subset(full_dataset, indices))
        print(f"  Client {i}: {len(indices)} samples")
    
    # Initialize global model
    global_model = SimpleCNN(input_channels=1, num_classes=1)
    
    # Strategy
    strategy = FedProx() if strategy_name == 'FedProx' else FedAvg()
    proximal_mu = 0.01 if strategy_name == 'FedProx' else 0.0
    
    config = {
        'local_epochs': local_epochs,
        'batch_size': 32,
        'learning_rate': 0.001,
        'proximal_mu': proximal_mu
    }
    
    metrics = {"loss": [], "accuracy": []}
    
    # FL Training Loop
    for round_num in range(rounds):
        client_models = []
        client_weights = []
        round_metrics = []
        
        global_params = list(global_model.parameters()) if proximal_mu > 0 else None
        
        for client_id, client_data in enumerate(client_datasets):
            # Clone global model for client
            client_model = SimpleCNN(input_channels=1, num_classes=1)
            client_model.load_state_dict(global_model.state_dict())
            
            # Local training
            num_samples, m = local_train(
                client_model, 
                client_data, 
                config,
                global_params=[p.clone() for p in global_model.parameters()] if proximal_mu > 0 else None
            )
            
            client_models.append(client_model.state_dict())
            client_weights.append(num_samples)
            round_metrics.append((num_samples, m))
        
        # Aggregate
        new_state = strategy.aggregate(global_model, client_models, client_weights)
        global_model.load_state_dict(new_state)
        
        # Compute weighted average metrics
        total_samples = sum(w for w, _ in round_metrics)
        round_loss = sum(w * m['loss'] for w, m in round_metrics) / total_samples
        round_acc = sum(w * m['accuracy'] for w, m in round_metrics) / total_samples
        
        metrics['loss'].append(round_loss)
        metrics['accuracy'].append(round_acc)
        
        print(f"  Round {round_num+1}/{rounds} - Loss: {round_loss:.4f} - Acc: {round_acc:.4f}")
    
    # Evaluate on test set
    test_loader = DataLoader(test_dataset, batch_size=32)
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            out = global_model(X)
            predicted = (out > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    test_acc = correct / total
    print(f"  Final Test Accuracy: {test_acc:.4f}")
    
    # Save results
    results = {
        "dataset": "PneumoniaMNIST",
        "strategy": strategy_name,
        "rounds": rounds,
        "clients": num_clients,
        "loss": metrics['loss'],
        "accuracy": metrics['accuracy'],
        "partition": partition,
        "epsilon": 0.0,
        "status": "complete"
    }
    
    os.makedirs('results', exist_ok=True)
    filename = f"results/fl_pneumonia_{strategy_name}_N{num_clients}_{partition}_eps0.0.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {filename}")
    
    return results


def run_all_experiments():
    """Run full grid of experiments."""
    client_counts = [5, 10, 20]
    strategies = ['FedAvg', 'FedProx']
    partitions = ['iid', 'non-iid']
    
    print("="*60)
    print("PneumoniaMNIST Federated Learning Experiment Grid")
    print("="*60)
    print(f"Client counts: {client_counts}")
    print(f"Strategies: {strategies}")
    print(f"Partitions: {partitions}")
    print(f"Total experiments: {len(client_counts) * len(strategies) * len(partitions)}")
    
    for n in client_counts:
        for strategy in strategies:
            for partition in partitions:
                run_fl_experiment(
                    num_clients=n,
                    strategy_name=strategy,
                    partition=partition,
                    rounds=10,  # 10 rounds for faster experiments
                    local_epochs=2
                )
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PneumoniaMNIST FL Experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--strategy", type=str, default="FedAvg", choices=["FedAvg", "FedProx"])
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "non-iid"])
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    
    args = parser.parse_args()
    
    if args.all:
        run_all_experiments()
    else:
        run_fl_experiment(
            num_clients=args.clients,
            strategy_name=args.strategy,
            partition=args.partition,
            rounds=args.rounds
        )
