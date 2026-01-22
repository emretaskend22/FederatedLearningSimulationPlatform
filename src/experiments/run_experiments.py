#!/usr/bin/env python3
import time
import subprocess
import os
import sys
import argparse
# Add project root to sys.path
sys.path.append(os.getcwd())
from src.orchestration.generate_compose import generate_compose

def run_command(command, cwd=None):
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )
    return process

def wait_for_completion(results_dir, expected_filename, timeout=600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(os.path.join(results_dir, expected_filename)):
            print(f"Found result file: {expected_filename}")
            return True
        time.sleep(5)
    print(f"Timeout waiting for {expected_filename}")
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    client_sizes = [5, 10, 20]
    partitions = ['iid', 'non-iid']
    strategies = ['FedAvg', 'FedProx']
    models = ['SimpleMLP', 'SimpleCNN']
    epsilon = 1.0
    rounds = 5 # Using 5 rounds as per config default / quick testing. Adjust if needed.

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    experiments = []
    for m in models:
        for n in client_sizes:
            for p in partitions:
                for s in strategies:
                    experiments.append({'n': n, 'p': p, 's': s, 'eps': epsilon, 'm': m})

    print(f"Found {len(experiments)} experiments to run.")

    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Starting Experiment: Model={exp['m']} N={exp['n']} Partition={exp['p']} Strategy={exp['s']} Epsilon={exp['eps']}")
        
        # 1. Generate Docker Compose
        print("Generating docker-compose.yml...")
        generate_compose(
            num_clients=exp['n'],
            strategy=exp['s'],
            dp_epsilon=exp['eps'],
            partition=exp['p'],
            rounds=rounds,
            model=exp['m']
        )

        expected_filename = f"fl_{exp['s']}_N{exp['n']}_{exp['p']}_eps{exp['eps']}_{exp['m']}.json"
        
        if args.dry_run:
            print(f"DRY RUN: Would execute docker-compose up --build, wait for {expected_filename}, then down.")
            continue

        # 2. Run Docker Compose
        print("Starting containers...")
        # Force build to ensure latest code is used, --remove-orphans to clean up
        cmd = "docker-compose up --build --force-recreate --remove-orphans -d"
        subprocess.run(cmd, shell=True, check=True)

        # 3. Monitor
        print(f"Waiting for results... (Timeout: 10 mins)")
        success = wait_for_completion(results_dir, expected_filename, timeout=600)
        
        if success:
            print("Experiment completed successfully.")
        else:
            print("Experiment failed or timed out.")

        # 4. Cleanup
        print("Stopping containers...")
        subprocess.run("docker-compose down", shell=True, check=True)
        
        # Mild pause between experiments to let ports free up
        time.sleep(5)

if __name__ == "__main__":
    main()
