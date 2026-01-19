"""
PneumoniaMNIST FL Server - FastAPI server for CNN model aggregation.

This server handles client registration, model distribution, and aggregation
for PneumoniaMNIST federated learning experiments.
"""

import os
import sys
import io
import base64
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Add src to path
sys.path.append(os.getcwd())

from src.server.aggregator_pneumonia import FLServerPneumonia
from src.core.model import SimpleCNN

app = FastAPI(title="PneumoniaMNIST FL Server")

# Configuration from environment or defaults
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "5"))
STRATEGY = os.getenv("STRATEGY", "FedAvg")
PARTITION = os.getenv("PARTITION", "iid")
EPSILON = float(os.getenv("EPSILON", "0.0"))

# Initialize server with CNN model
server = FLServerPneumonia(
    model_class=SimpleCNN,
    model_kwargs={"input_channels": 1, "num_classes": 1},
    min_clients=NUM_CLIENTS,
    max_rounds=MAX_ROUNDS,
    strategy_name=STRATEGY,
    config={
        "partition": PARTITION,
        "epsilon": EPSILON,
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,  # Lower LR for CNN
        "proximal_mu": 0.01 if STRATEGY == "FedProx" else 0.0
    }
)


@app.get("/")
def root():
    return {"message": "PneumoniaMNIST FL Server", "status": "running"}


@app.get("/config")
def get_config():
    """Return current training configuration and round status."""
    return {
        "round": server.current_round,
        "finished": server.current_round >= server.max_rounds,
        "local_epochs": server.config.get("local_epochs", 1),
        "batch_size": server.config.get("batch_size", 32),
        "learning_rate": server.config.get("learning_rate", 0.001),
        "proximal_mu": server.config.get("proximal_mu", 0.0)
    }


@app.get("/model")
def get_model():
    """Return the current global model state as base64-encoded bytes."""
    state = server.get_global_model_state()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    return {"encoded_state": encoded}


@app.post("/register")
def register_client(data: dict):
    """Register a new client."""
    client_id = data.get("client_id")
    if client_id:
        server.register_client(client_id)
        return {"status": "registered", "client_id": client_id}
    return JSONResponse(status_code=400, content={"error": "client_id required"})


@app.post("/update")
async def receive_update(
    client_id: str = Form(...),
    num_samples: int = Form(...),
    loss: float = Form(...),
    accuracy: float = Form(...),
    dp_epsilon: float = Form(0.0),
    file: UploadFile = File(...)
):
    """Receive model update from a client."""
    content = await file.read()
    buffer = io.BytesIO(content)
    state_dict = torch.load(buffer, weights_only=False)
    
    metrics = {"loss": loss, "accuracy": accuracy}
    aggregated = server.receive_update(client_id, state_dict, num_samples, metrics, dp_epsilon=dp_epsilon)
    
    return {
        "status": "received",
        "aggregated": aggregated,
        "current_round": server.current_round
    }


@app.get("/metrics")
def get_metrics():
    """Return training metrics history."""
    return {
        "rounds": server.current_round,
        "loss": server.metrics_history["loss"],
        "accuracy": server.metrics_history["accuracy"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
