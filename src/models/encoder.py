import math
import torch
import torch.nn as nn
from .embeddings import TokenEmbedding, SpatialEmbedding, PositionalEmbedding
from .encoder_block import TransformerEncoderBlock


class LayoutTransformerEncoder(nn.Module):
    """
    Transformer encoder that fuses text tokens with spatial layout (bounding boxes).
    Combines:
      - Token embeddings
      - Spatial embedding (from normalised bounding boxes)
      - Learned positional embeddings
      - Stack of TransformerEncoderBlock layers
    """
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 12, d_ff: int = 3072, max_len: int = 512,
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx=0)
        self.spatial_embedding = SpatialEmbedding(d_model, dropout=dropout)
        self.position_embedding = PositionalEmbedding(max_len, d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout, bias)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, bboxes, attention_mask=None):
        """
        Args:
            input_ids: (B, L)
            bboxes: (B, L, 4) – normalised coordinates in [0,1]
            attention_mask: (B, L) – 1 for tokens, 0 for padding (optional)
        Returns:
            hidden_states: (B, L, d_model)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.token_embedding(input_ids)          # (B, L, D)
        spa_emb = self.spatial_embedding(bboxes)          # (B, L, D)
        pos_emb = self.position_embedding(tok_emb)        # (1, L, D) → broadcast

        x = tok_emb + spa_emb + pos_emb
        x = self.dropout(x)

        # Attention mask reshaping
        if attention_mask is not None:
            # (B, L) → (B, 1, 1, L) so that MultiHeadAttention can mask
            extended_mask = attention_mask[:, None, None, :]
        else:
            extended_mask = None

        # Pass through each encoder block
        for layer in self.layers:
            x = layer(x, mask=extended_mask)

        # Final layer norm
        x = self.layer_norm(x)
        return x