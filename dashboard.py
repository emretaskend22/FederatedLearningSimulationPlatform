import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

# Set Plot Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 9}) # Smaller font for plots

st.set_page_config(layout="wide", page_title="FL Comparative Analysis")

# --- Custom CSS for Professional UI ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-bottom: 2px solid transparent; 
        color: #495057;
        font-weight: 600;
        font-size: 14px;
        border-radius: 0px;
        border: none;
        padding: 0px 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        border-bottom: 3px solid #4e73df !important; /* Underline style */
        color: #4e73df !important;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        text-align: center;
        padding-bottom: 30px;
        font-weight: 700;
    }
    h3 {
        color: #4e73df;
        font-size: 1.1rem;
        padding-top: 15px;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Federated Learning Simulation Platform")

# --- Dataset Selection ---
dataset_options = ["Adult (Fraud Detection)", "PneumoniaMNIST (Medical Imaging)"]
selected_dataset = st.selectbox("📁 Select Dataset", dataset_options)

is_pneumonia = "Pneumonia" in selected_dataset

# --- Data Loading ---
results_dir = "results"

# Load appropriate baseline
if is_pneumonia:
    baseline_path = "results/centralized_pneumonia_metrics.json"
    fl_prefix = "fl_pneumonia_"
    dataset_name = "PneumoniaMNIST"
else:
    baseline_path = "results/centralized_metrics.json"
    fl_prefix = "fl_"
    dataset_name = "Adult"

# 1. Load Centralized Baseline
baseline_acc = 0.0
baseline_loss = 0.0
if os.path.exists(baseline_path):
    with open(baseline_path, 'r') as f:
        c_data = json.load(f)
        baseline_acc = max(c_data['accuracy'])
        baseline_loss = min(c_data['loss'])

# 2. Load FL Experiments
experiments = []
if os.path.exists(results_dir):
    # Load ALL files starting with fl_
    files = [f for f in os.listdir(results_dir) if f.startswith("fl_") and f.endswith(".json")]
    for f in files:
        try:
            with open(os.path.join(results_dir, f), 'r') as file:
                data = json.load(file)
                
                # Parsing Logic
                # 1. New Format: fl_{strategy}_N{clients}_{partition}_eps{epsilon}_{model}.json
                match = re.search(r"fl_(.+?)_N(\d+)_(.+?)_eps([\d\.]+)_([A-Za-z0-9]+)\.json", f)
                if match:
                    data['strategy'] = match.group(1)
                    data['clients'] = int(match.group(2))
                    data['partition'] = match.group(3)
                    data['epsilon'] = float(match.group(4))
                    data['model'] = match.group(5)
                
                # 2. Legacy Pneumonia: fl_pneumonia_{strategy}_N{clients}_{partition}_eps{epsilon}.json
                elif f.startswith("fl_pneumonia_"):
                    match_p = re.search(r"fl_pneumonia_(.+?)_N(\d+)_(.+?)_eps([\d\.]+)\.json", f)
                    if match_p:
                        data['strategy'] = match_p.group(1)
                        data['clients'] = int(match_p.group(2))
                        data['partition'] = match_p.group(3)
                        data['epsilon'] = float(match_p.group(4))
                        data['model'] = 'SimpleCNN' # Legacy pneumonia mapped to SimpleCNN
                    else:
                        continue
                
                # 3. Legacy Adult: fl_{strategy}_N{clients}_{partition}_eps{epsilon}.json
                else:
                    match_old = re.search(r"fl_(.+?)_N(\d+)_(.+?)_eps([\d\.]+)\.json", f)
                    if match_old:
                         data['strategy'] = match_old.group(1)
                         data['clients'] = int(match_old.group(2))
                         data['partition'] = match_old.group(3)
                         data['epsilon'] = float(match_old.group(4))
                         data['model'] = 'SimpleMLP' # Legacy default to Adult/SimpleMLP
                    else:
                        continue # Skip unparseable files

                if 'accuracy' in data and data['accuracy']:
                    data['final_accuracy'] = data['accuracy'][-1]
                else:
                    data['final_accuracy'] = 0.0
                    
                if 'loss' in data and data['loss']:
                    data['final_loss'] = data['loss'][-1]
                else:
                    data['final_loss'] = 0.0
                
                experiments.append(data)
        except Exception:
            pass

all_df = pd.DataFrame(experiments)

# Filter by selected dataset
if not all_df.empty:
    if is_pneumonia:
        # Filter for SimpleCNN
        df = all_df[all_df['model'] == 'SimpleCNN'].copy()
    else:
        # Filter for SimpleMLP
        df = all_df[all_df['model'] == 'SimpleMLP'].copy()
else:
    df = pd.DataFrame()

if df.empty:
    st.warning(f"🚀 No experiment results found for {dataset_name}. Please run the simulation first.")
    st.info(f"""
    **To run experiments for {dataset_name}:**
    
    1. **Centralized Baseline:**
    ```bash
    python -m src.centralized.train{'_pneumonia' if is_pneumonia else ''}
    ```
    
    2. **Federated Learning Experiments:**
    ```bash
    python src/experiments/run_{'pneumonia_' if is_pneumonia else ''}experiments.py --all
    ```
    """)
    st.stop()

# Ensure types
df['clients'] = df['clients'].astype(int)
df['final_accuracy'] = df['final_accuracy'].astype(float)
df['final_loss'] = df['final_loss'].astype(float)

# --- Summary Metrics ---
total_exps = len(df)
best_acc = df['final_accuracy'].max()
best_strategy = df.loc[df['final_accuracy'].idxmax()]['strategy']

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Experiments", total_exps)
col2.metric("Best Accuracy (FL)", f"{best_acc:.4f}")
col3.metric("Baseline Accuracy", f"{baseline_acc:.4f}" if baseline_acc > 0 else "N/A")
col4.metric("Baseline Gap", f"{best_acc - baseline_acc:+.4f}" if baseline_acc > 0 else "N/A")

st.markdown("---")

# --- TABS ---
tabs = st.tabs(["📊 Strategy Comparison", "📈 Scale Analysis", "📉 Data Distribution"])

# Helper for subplot logic - REDUCED FIGSIZE
def plot_comparison(df_sub, x_col, y_col, hue_col, title, ax, baseline_val=None):
    if df_sub.empty:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=9, color='grey')
        return
        
    sns.lineplot(data=df_sub, x=x_col, y=y_col, hue=hue_col, style=hue_col, markers=True, dashes=False, ax=ax, linewidth=2.5, markersize=8)
    
    # Baseline
    if baseline_val and baseline_val > 0:
        ax.axhline(y=baseline_val, color='green', linestyle='--', label='Centralized Baseline', alpha=0.5, linewidth=1.5)
    
    ax.set_title(title, fontsize=11, fontweight='600', pad=10)
    ax.set_xlabel(x_col.capitalize().replace("_", " "), fontsize=10)
    ax.set_ylabel(y_col.replace("final_", "").capitalize(), fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    if x_col == 'clients':
         ax.set_xticks(sorted(df_sub['clients'].unique()))
    
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, linestyle=':', alpha=0.4)
    # Remove top and right spines
    sns.despine()

# --- TAB 1: Strategy Comparison ---
with tabs[0]:
    with st.container():
        st.caption(f"Compare how different algorithms perform across IID and Non-IID settings for {dataset_name}.")
        
        # Row 1: Accuracy - Ultra Compact (10, 3.5)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        plot_comparison(df[df['partition'] == 'iid'], 'clients', 'final_accuracy', 'strategy', "IID: Accuracy", axes[0], baseline_acc)
        plot_comparison(df[df['partition'] == 'non-iid'], 'clients', 'final_accuracy', 'strategy', "Non-IID: Accuracy", axes[1], baseline_acc)
        st.pyplot(fig)


# --- TAB 2: Scale Analysis ---
with tabs[1]:
    with st.container():
        st.caption("Analyze how model performance degrades as you add more clients.")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        plot_comparison(df[df['strategy'] == 'FedAvg'], 'clients', 'final_accuracy', 'partition', "FedAvg Scaling", axes[0], baseline_acc)
        plot_comparison(df[df['strategy'] == 'FedProx'], 'clients', 'final_accuracy', 'partition', "FedProx Scaling", axes[1], baseline_acc)
        st.pyplot(fig)

# --- TAB 3: Data Distribution ---
with tabs[2]:
    with st.container():
        st.caption("Assess the robustness of algorithms to data heterogeneity.")
        
        # New Layout: Selection on TOP
        clients_opts = sorted(df['clients'].unique())
        
        col_c1, col_c2, col_c3 = st.columns([1, 2, 1]) # Column 2 is 50% width centered
        with col_c2:
            sel_n = st.selectbox("Select Client Count (N)", clients_opts) if clients_opts else None
        
        if sel_n:
            subset = df[df['clients'] == sel_n]
            
            # Constrain to Middle Column
            with col_c2:
                # Metric Card style Chart
                fig, ax = plt.subplots(figsize=(5, 3)) 
                
                if not subset.empty:
                    sns.barplot(data=subset, x='partition', y='final_accuracy', hue='strategy', ax=ax, palette='viridis')
                    ax.set_ylim(0, 1.05)
                    if baseline_acc > 0:
                        ax.axhline(y=baseline_acc, color='green', linestyle='--', label='Centralized', alpha=0.6)
                    ax.set_title(f"Accuracy Gap @ N={sel_n}", fontsize=10, fontweight='600')
                    ax.legend(loc='lower right', fontsize=8, frameon=True)
                    ax.tick_params(labelsize=8)
                    sns.despine()
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("No data for this client count.")


