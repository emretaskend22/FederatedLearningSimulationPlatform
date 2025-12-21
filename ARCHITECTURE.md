Project Proposal: Comparative Analysis of Privacy Preserving Federated Learning Architectures

1. Executive Summary & Problem Statement
1.1 Motivation
Data Engineering is increasingly moving away from "Big Data Centralization" toward "Distributed Data Governance." Data cannot be moved due to regulation. Federated Learning (FL) is the solution, but its performance is highly sensitive to:
1.	Network Scale: How many clients (5, 10, 50) are participating?
2.	Statistical Heterogeneity: Is the data balanced (IID) or skewed (Non-IID)?
3.	Privacy Overhead: How much noise from Differential Privacy (DP) can the model handle before utility collapses?
This project builds a framework to answer these questions by comparing different FL aggregation strategies across two distinct data domains.
1.2 Scope of Investigation
Instead of a single implementation, this project will execute a 3-dimensional grid search of experiments:
•	Variable Clients: Simulations with $N \in \{5, 10, 20\}$ client nodes.
•	Data Skew: Comparing performance on perfectly balanced data vs. highly skewed (label-skewed) data.
•	Aggregation Strategies: Benchmarking the standard FedAvg against more robust algorithms like FedProx or FedYogi.
2. Technical Methodology & Experimental Design
2.1 The Two-Dataset Strategy
By using two different datasets, we prove the system's generalizability.
•	Dataset 1 Adult Income: [Description of data type, e.g., Tabular, Image, or Time-series].
•	Dataset 2 [Name]: [Description of data type].
2.2 Data Engineering Pipeline
The pipeline will be built as a modular "plug-and-play" system in Python/PyCharm:
1.	Ingestion Layer: Loads raw [DATASET_A/B].
2.	Partitioning Engine: This is a custom script that divides the data into N clients based on two specific scenarios:
o	Scenario I (IID): Random, uniform distribution.
o	Scenario II (Non-IID): Skewed distribution (e.g., Client A only sees Label 1, Client B only sees Label 2).
3.	Orchestration Layer: Using Docker Compose to spin up N containers, each representing a "siloed" client node.
2.3 Federated Approaches to Compare
We will implement and compare at least two distinct aggregation strategies:
•	FedAvg (Baseline): Simple weight averaging.
•	FedProx (Advanced): Adds a "proximal term" to handle the "drift" caused by Non-IID data. This is a superior engineering choice for skewed datasets.
3. Privacy, Evaluation, and Road-map 
3.1 Privacy Layer: Differential Privacy (DP)
Each experiment will be run with and without Differential Privacy. We will use the Gaussian Mechanism to inject noise into the gradients.
•	Goal: Quantify the "Accuracy Drop" caused by privacy. If Hospital A adds DP noise, how much does the global model's F1-score decrease?
3.2 Key Performance Indicators (KPIs)
The success of the engineering framework will be measured by:
•	Convergence Speed: How many communication rounds does it take to reach 80% accuracy?
•	Communication Overhead: How much data (MB) is exchanged between the server and N clients?
•	Privacy-Utility Curve: A plot showing Accuracy vs. Privacy Budget (epsilon).
