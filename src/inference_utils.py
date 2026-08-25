import torch
from src.model import build_model

def load_trained_model(checkpoint_path: str, device: torch.device):
    model = build_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model