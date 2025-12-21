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

st.set_page_config(layout="wide", page_title="FL Comparative Analysis")
st.title("Federated Learning: Comparative Analysis")

# --- Data Loading ---
results_dir = "results"
baseline_path = "results/centralized_metrics.json"

# 1. Load Centralized Baseline
baseline_acc = 0.0
baseline_loss = 0.0
if os.path.exists(baseline_path):
    with open(baseline_path, 'r') as f:
        c_data = json.load(f)
        # Use simple max val for accuracy, min for loss (or last?)
        # "Best value" is usually better reference.
        baseline_acc = max(c_data['accuracy'])
        baseline_loss = min(c_data['loss'])

# 2. Load FL Experiments
experiments = []
if os.path.exists(results_dir):
    files = [f for f in os.listdir(results_dir) if f.startswith("fl_") and f.endswith(".json")]
    for f in files:
        try:
            with open(os.path.join(results_dir, f), 'r') as file:
                data = json.load(file)
                
                # Parse Filename Metadata (robust fallback)
                match = re.search(r"fl_(.+?)_N(\d+)_(.+?)_eps([\d\.]+)\.json", f)
                if match:
                    data['strategy'] = match.group(1)
                    data['clients'] = int(match.group(2))
                    data['partition'] = match.group(3)
                    data['epsilon'] = float(match.group(4))
                else:
                    # Fallback or manual extraction
                    if 'strategy' not in data: data['strategy'] = 'Unknown'
                    if 'clients' not in data: data['clients'] = 0
                    if 'partition' not in data: data['partition'] = 'ios'

                # Extract FINAL metrics
                # We assume the lists are populated.
                if 'accuracy' in data and data['accuracy']:
                    data['final_accuracy'] = data['accuracy'][-1]
                else:
                    data['final_accuracy'] = 0.0
                    
                if 'loss' in data and data['loss']:
                    data['final_loss'] = data['loss'][-1]
                else:
                    data['final_loss'] = 0.0
                
                experiments.append(data)
        except Exception as e:
            # st.warning(f"Error {f}: {e}")
            pass

df = pd.DataFrame(experiments)

if df.empty:
    st.warning("No experiment results found. Run grid search first.")
    st.stop()

# Ensure types
df['clients'] = df['clients'].astype(int)
df['final_accuracy'] = df['final_accuracy'].astype(float)
df['final_loss'] = df['final_loss'].astype(float)

# --- TABS ---
tabs = st.tabs(["1️⃣ Strategy Comparison", "2️⃣ Scale Analysis", "3️⃣ Data Distribution", "📋 Raw Data"])

# Helper for subplot logic to avoid repetition
def plot_comparison(df_sub, x_col, y_col, hue_col, title, ax, baseline_val=None):
    if df_sub.empty:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        return
        
    sns.lineplot(data=df_sub, x=x_col, y=y_col, hue=hue_col, style=hue_col, markers=True, dashes=False, ax=ax, linewidth=2.5, markersize=9)
    
    # Baseline
    if baseline_val:
        ax.axhline(y=baseline_val, color='green', linestyle='--', label='Centralized (Best)', alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel(x_col.capitalize().replace("_", " "))
    # Fix integer ticks for clients
    if x_col == 'clients':
         ax.set_xticks(sorted(df_sub['clients'].unique()))
    
    ax.legend(title=hue_col.capitalize())

# --- TAB 1: Strategy Comparison ---
with tabs[0]:
    st.markdown("### Accuracy/Loss vs Clients (FedAvg vs FedProx)")
    
    # Row 1: Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: IID
    plot_comparison(
        df[df['partition'] == 'iid'], 
        x_col='clients', y_col='final_accuracy', hue_col='strategy', 
        title="IID: Accuracy vs Scale", ax=axes[0], baseline_val=baseline_acc
    )
    
    # Subplot 2: Non-IID
    plot_comparison(
        df[df['partition'] == 'non-iid'], 
        x_col='clients', y_col='final_accuracy', hue_col='strategy', 
        title="Non-IID: Accuracy vs Scale", ax=axes[1], baseline_val=baseline_acc
    )
    st.pyplot(fig)
    
    # Row 2: Loss (Optional toggle?)
    st.markdown("#### Loss View")
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    plot_comparison(
        df[df['partition'] == 'iid'], 
        x_col='clients', y_col='final_loss', hue_col='strategy', 
        title="IID: Loss vs Scale", ax=axes2[0], baseline_val=baseline_loss
    )
    plot_comparison(
        df[df['partition'] == 'non-iid'], 
        x_col='clients', y_col='final_loss', hue_col='strategy', 
        title="Non-IID: Loss vs Scale", ax=axes2[1], baseline_val=baseline_loss
    )
    st.pyplot(fig2)

# --- TAB 2: Scale Analysis ---
with tabs[1]:
    st.markdown("### Scaling Efficiency (IID vs Non-IID)")
    
    # Row 1: Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: FedAvg scale
    plot_comparison(
        df[df['strategy'] == 'FedAvg'],
        x_col='clients', y_col='final_accuracy', hue_col='partition',
        title="FedAvg: Scaling Impact", ax=axes[0], baseline_val=baseline_acc
    )
    
    # Panel 2: FedProx scale
    plot_comparison(
        df[df['strategy'] == 'FedProx'],
        x_col='clients', y_col='final_accuracy', hue_col='partition',
        title="FedProx: Scaling Impact", ax=axes[1], baseline_val=baseline_acc
    )
    st.pyplot(fig)

# --- TAB 3: Data Distribution ---
with tabs[2]:
    st.markdown("### Sensitivity to Heterogeneity")
    
    # Control: Select N
    clients_opts = sorted(df['clients'].unique())
    if clients_opts:
        sel_n = st.selectbox("Select Client Count (N)", clients_opts)
        
        subset = df[df['clients'] == sel_n]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Bar Chart
        if not subset.empty:
            sns.barplot(data=subset, x='partition', y='final_accuracy', hue='strategy', ax=ax, palette='muted')
            ax.set_ylim(0, 1.0)
            ax.axhline(y=baseline_acc, color='green', linestyle='--', label='Centralized')
            ax.set_title(f"Accuracy Gap (IID vs Non-IID) @ N={sel_n}")
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            st.pyplot(fig)
        else:
            st.info("No data for this client count.")

# --- TAB 4: Raw ---
with tabs[3]:
    st.dataframe(df)
