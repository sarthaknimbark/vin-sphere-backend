import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Union

VOCAB = "?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
char_to_idx = {char: idx + 1 for idx, char in enumerate(VOCAB)}
vocab_size = len(VOCAB) + 1  # 0 is reserved for padding/unknown characters

def encode_vin(vin_str: str) -> np.ndarray:
    """Encode a 17-character VIN into a sequence of vocabulary indices."""
    vin_str = str(vin_str).upper().strip().replace(" ", "")
    # Standard standard 17-character length
    encoded = []
    for char in vin_str[:17]:
        encoded.append(char_to_idx.get(char, 0))
    # Pad to 17 if shorter
    while len(encoded) < 17:
        encoded.append(0)
    return np.array(encoded, dtype=np.int64)

class MultiTaskVINDataset(Dataset):
    """
    Multi-Task VIN Dataset to load tokenized VIN sequences and target labels
    for the multi-task Transformer architecture.
    """
    def __init__(self, X_vins: Union[List[str], np.ndarray], y_dict: Dict[str, Union[List[int], np.ndarray]]):
        self.X = torch.tensor(np.stack([encode_vin(vin) for vin in X_vins]), dtype=torch.long)
        self.y_dict = {k: torch.tensor(v, dtype=torch.long) for k, v in y_dict.items()}
        self.targets = list(y_dict.keys())

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {'x': self.X[idx]}
        for target in self.targets:
            item[target] = self.y_dict[target][idx]
        return item
