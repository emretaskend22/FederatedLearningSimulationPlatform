#!/usr/bin/env python3
import subprocess
import time
import os
import sys

# Combinations to run
# Combinations to run
# Full Grid Search (12 Experiments)
CLIENT_COUNTS = [5, 10, 20] 
STRATEGIES = ['FedAvg', 'FedProx']
PARTITIONS = ['iid', 'non-iid']
# Epsilon: Fixed to 0 for now for speed

def run_experiment(clients, strategy, partition, epsilon=0.0):
    print(f"=== Running Experiment: N={clients}, {strategy}, {partition}, eps={epsilon} ===")
    
    # 1. Generate Compose
    cmd_gen = [
        sys.executable, "src/orchestration/generate_compose.py",
        "--clients", str(clients),
        "--strategy", strategy,
        "--partition", partition,
        "--epsilon", str(epsilon),
        "--rounds", "20"
    ]
    subprocess.check_call(cmd_gen)
    
    # 2. Docker Up (Build & Force Recreate)
    cmd_up = ["docker-compose", "up", "-d", "--build", "--force-recreate"]
    subprocess.check_call(cmd_up)
    
    # 3. Wait for completion
    # Poll results file
    filename = f"results/fl_{strategy}_N{clients}_{partition}_eps{epsilon}.json"
    print(f"Waiting for results in {filename}...")
    
    start_time = time.time()
    while True:
        if os.path.exists(filename):
            # Check if status is complete
            try:
                import json
                with open(filename, 'r') as f:
                    data = json.load(f)
                    if data.get('status') == 'complete':
                        print(f"Experiment finished!")
                        break
            except:
                pass
        
        # Timeout safety (e.g. 5 mins)
        if time.time() - start_time > 300: 
            print("Timeout reached!")
            break
            
        time.sleep(5)
        
    # 4. Cleanup
    subprocess.check_call(["docker-compose", "down"])
    print("Cleanup complete.\n")

if __name__ == "__main__":
    # Ensure results dir exists
    os.makedirs("results", exist_ok=True)
    
    # Run Grid
    for c in CLIENT_COUNTS:
        for s in STRATEGIES:
            for p in PARTITIONS:
                run_experiment(c, s, p)
