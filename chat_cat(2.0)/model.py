import torch
import torch.nn as nn
from typing import Dict

class MultiTaskVINTransformer(nn.Module):
    """
    Multi-Task Character-Level Transformer Model for VIN Decoding.
    
    Architecture:
    1. Input: (Batch, 17) indices
    2. Embedding Layer
    3. Learnable Positional Encoding
    4. Transformer Encoder (3-6 layers)
    5. Global Pooling (Mean + Max concatenated)
    6. Dense Heads for multiple tasks (Trim, Regional Spec, Engine, Fuel Type, Body Type)
    """
    def __init__(
        self,
        num_classes_dict: Dict[str, int],
        vocab_size: int = 39,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.num_classes_dict = num_classes_dict
        self.config = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "hidden_dim": hidden_dim
        }
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Learnable positional encoding (Sequence length is fixed at 17 characters)
        self.pos_embedding = nn.Parameter(torch.randn(1, 17, embedding_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # We concatenate average pooling and max pooling to enrich the global representation
        pooled_dim = embedding_dim * 2
        
        # Task-specific prediction heads
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ) for target, num_classes in num_classes_dict.items()
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Assert input shape
        assert x.ndim == 2, f"Input tensor must be 2D (batch_size, seq_len), got shape {x.shape}"
        assert x.shape[1] == 17, f"VIN sequence length must be exactly 17, got shape {x.shape}"
        
        batch_size = x.shape[0]
        
        # Embedding and positional encoding additions
        # Shape: (batch_size, 17, embedding_dim)
        h = self.embedding(x) + self.pos_embedding
        
        # Pass through Transformer encoder
        # Shape: (batch_size, 17, embedding_dim)
        h = self.transformer(h)
        
        # Global Pooling along sequence dimension (dim=1)
        mean_pooled = torch.mean(h, dim=1) # (batch_size, embedding_dim)
        max_pooled = torch.max(h, dim=1)[0] # (batch_size, embedding_dim)
        
        # Concatenate mean and max pooled representations
        # Shape: (batch_size, embedding_dim * 2)
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        
        # Predict logits for each task head
        logits = {}
        for target, head in self.heads.items():
            out = head(pooled)
            # Assert output shapes
            expected_classes = self.num_classes_dict[target]
            assert out.shape == (batch_size, expected_classes), \
                f"Head '{target}' output shape {out.shape} does not match expected {(batch_size, expected_classes)}"
            logits[target] = out
            
        return logits
