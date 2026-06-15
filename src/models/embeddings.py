# src/models/embeddings.py
import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learned token embedding with scaling (as in BERT / GPT)."""
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, input_ids):
        return self.embedding(input_ids) * math.sqrt(self.d_model)


class SpatialEmbedding(nn.Module):
    """MLP that maps normalised bounding‑box coordinates to a dense vector."""
    def __init__(self, d_model: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, bboxes):
        """bboxes: (B, L, 4) → (B, L, d_model)"""
        return self.mlp(bboxes)


class PositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings up to max_len positions."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """x: (B, L, D) or just provide positions as (L,) via torch.arange"""
        B, L, _ = x.shape
        positions = torch.arange(L, dtype=torch.long, device=x.device).unsqueeze(0)  # (1, L)
        return self.embedding(positions)