import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerEncoderBlock(nn.Module):
    """
    A single encoder block: Self‑Attention + Feed‑Forward with residual
    connections and layer normalization (post‑norm).
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, bias)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, D)
            mask: optional attention mask (B, 1, 1, L) or broadcastable
        Returns:
            (B, L, D)
        """
        # Self‑attention sub‑layer
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed‑forward sub‑layer
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x