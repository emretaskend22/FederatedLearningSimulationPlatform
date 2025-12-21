from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import torch
import io
import pickle
import base64
from src.server.aggregator import FLServer
import os

app = FastAPI()

# Configuration (Env vars or defaults)
INPUT_DIM = int(os.getenv("INPUT_DIM", 96)) # Adult dataset dimension after processing
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", 2))
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", 5)) # Default to 5 rounds
STRATEGY = os.getenv("STRATEGY", "FedAvg")
PARTITION = os.getenv("PARTITION", "iid")
DP_EPSILON = float(os.getenv("DP_EPSILON", 0.0))


# Global Server Instance
server_config = {
    "partition": PARTITION,
    "epsilon": DP_EPSILON
}
server = FLServer(input_dim=INPUT_DIM, min_clients=MIN_CLIENTS, max_rounds=MAX_ROUNDS, strategy_name=STRATEGY, config=server_config)

class RegisterRequest(BaseModel):
    client_id: str

@app.post("/register")
def register(req: RegisterRequest):
    success = server.register_client(req.client_id)
    return {"status": "registered" if success else "already_registered", "round": server.current_round}

@app.get("/model")
def get_model():
    # Serialize state dict
    state_dict = server.get_global_model_state()
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    # Return as bytes
    # For simplicity in FastAPI, we can return generic bytes/streaming
    return {"encoded_state": base64.b64encode(buffer.getvalue()).decode('utf-8')}

@app.post("/update")
async def update(client_id: str = Form(...), num_samples: int = Form(...), loss: float = Form(0.0), accuracy: float = Form(0.0), file: UploadFile = File(...)):
    # Read uploaded model file
    content = await file.read()
    buffer = io.BytesIO(content)
    state_dict = torch.load(buffer)
    
    metrics = {"loss": loss, "accuracy": accuracy}
    round_complete = server.receive_update(client_id, state_dict, num_samples, metrics)
    return {"status": "accepted", "round_complete": round_complete}

@app.get("/config")
def get_config():
    # Return training config for next round
    return {
        "round": server.current_round,
        "finished": server.current_round >= server.max_rounds,
        "local_epochs": 1,
        "learning_rate": 0.01,
        "batch_size": 32,
        "proximal_mu": 0.01 if STRATEGY == 'FedProx' else 0.0
    }
