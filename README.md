# Federated Learning Simulation Platform

A robust, Docker-based platform for simulating and analyzing Federated Learning (FL) algorithms. This project allows you to compare aggregation strategies, analyze scalability, and visualize the impact of data heterogeneity on model performance.

## 🚀 Features

*   **Algorithms**: Support for **FedAvg** (Federated Averaging) and **FedProx** (Federated Proximal Optimization).
*   **Privacy**: Integrated Differential Privacy (DP) mechanism using Gaussian noise.
*   **Simulation**: Fully containerized environment using Docker Compose to orchestrate Server and multiple Clients.
*   **Data Heterogeneity**: Capabilities to simulate IID (Independent and Identically Distributed) and Non-IID data partitions across clients.
*   **Analysis Dashboard**: Interactive Streamlit dashboard for comparative analysis:
    *   **Strategy Comparison**: Benchmarking different algorithms.
    *   **Scale Analysis**: Analyzing performance as client count increases.
    *   **Distribution Impact**: Visualizing robustness to non-IID data.

## 🛠 Project Structure

```bash
.
├── dashboard.py            # Streamlit Analytics Dashboard
├── src/
│   ├── server/             # FastAPI Aggregator (Central Server)
│   ├── client/             # Training Logic (Client)
│   ├── centralized/        # Global Baseline Training Script
│   ├── orchestration/      # Docker Compose Generator
│   ├── experiments/        # Grid Search Automation
│   └── data/               # Dataset Handling (Adult Income)
├── results/                # JSON Logs & Metrics
└── docker-compose.yml      # (Generated) Simulation Config
```

## ⚡ Quick Start

### 1. Prerequisites
*   Docker & Docker Compose
*   Python 3.13+

### 2. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Establish Baseline
Train a centralized model on the full dataset to set a "gold standard" for accuracy.
```bash
python src/centralized/train.py
```

### 4. Run an FL Experiment
You can run a specific simulation configuration manually:
```bash
# 1. Generate Configuration (e.g., 5 Clients, FedProx, Non-IID)
python src/orchestration/generate_compose.py --clients 5 --strategy FedProx --partition non-iid

# 2. Start Simulation
docker-compose up -d --build --force-recreate

# 3. Watch Logs (Optional)
docker-compose logs -f

# 4. Stop & Cleanup
docker-compose down
```

### 5. Automated Grid Search
To generate data for comparative analysis, run the automation script. This runs multiple permutations (N=2/5, FedAvg/FedProx, IID/Non-IID) sequentially.
```bash
python src/experiments/run_grid_search.py
```

## 📊 Dashboard
Visualize and compare your results interactively.

```bash
streamlit run dashboard.py
```
Open the provided URL (usually `http://localhost:8501`) to see:
*   **Tab 1**: FedAvg vs. FedProx Comparison.
*   **Tab 2**: Scalability Analysis (Impact of adding clients).
*   **Tab 3**: Robustness to Data Distribution (IID vs Non-IID).
