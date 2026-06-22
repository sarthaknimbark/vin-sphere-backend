import os
import logging
import pickle
import torch
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

def save_fallback_model(model_dict: dict, model_dir: str) -> str:
    """Saves the fallback frequency/backoff model dictionary to a serialized file."""
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, "fallback_model.pkl")
    logger.info(f"Saving fallback model to {filepath}...")
    with open(filepath, "wb") as f:
        pickle.dump(model_dict, f)
    return filepath

def load_fallback_model(model_dir: str) -> dict:
    """Loads the fallback frequency/backoff model dictionary from the serialized file."""
    filepath = os.path.join(model_dir, "fallback_model.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fallback model file not found at '{filepath}'.")
    logger.info(f"Loading fallback model from {filepath}...")
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_vin_transformer(model_dir: str, device: torch.device) -> Optional[torch.nn.Module]:
    """
    Dynamically instantiates and loads the MultiTaskVINTransformer model from the saved checkpoint.
    
    Returns None if the checkpoint file does not exist, permitting graceful fallback.
    """
    filepath = os.path.join(model_dir, "vin_transformer_best.pt")
    if not os.path.exists(filepath):
        logger.warning(f"Transformer model file not found at '{filepath}'. Inference will operate in Lookup-only mode.")
        return None
        
    logger.info(f"Loading VIN Transformer model from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)
    num_classes_dict = checkpoint["num_classes_dict"]
    
    # Import inside function to prevent circular dependency
    from model import MultiTaskVINTransformer
    
    config = checkpoint.get("config", None)
    if config is not None:
        model = MultiTaskVINTransformer(
            num_classes_dict=num_classes_dict,
            vocab_size=config.get("vocab_size", 39),
            embedding_dim=config.get("embedding_dim", 128),
            num_layers=config.get("num_layers", 4),
            num_heads=config.get("num_heads", 4),
            dim_feedforward=config.get("dim_feedforward", 256),
            dropout=config.get("dropout", 0.1),
            hidden_dim=config.get("hidden_dim", 64)
        )
    else:
        model = MultiTaskVINTransformer(
            num_classes_dict=num_classes_dict,
            vocab_size=39,
            embedding_dim=128,
            num_layers=4,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1,
            hidden_dim=64
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
