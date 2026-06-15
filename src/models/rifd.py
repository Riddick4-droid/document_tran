import torch.nn as nn
from .encoder import LayoutTransformerEncoder
from .heads import MLMHead, NERHead
from src.exceptions import ProjectException
import torch

class RIFD(nn.Module):
    """
    Read Info From Documents – Layout‑aware Transformer model.
    Supports:
    - Pre‑training: Masked Language Modelling (MLM)
    - Fine‑tuning: Token classification (NER on forms)
    """
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 12, d_ff: int = 3072, max_len: int = 512,
                 dropout: float = 0.1, bias: bool = True, num_ner_labels: int = None):
        super().__init__()
        self.encoder = LayoutTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            bias=bias
        )
        self.mlm_head = MLMHead(d_model, vocab_size)

        if num_ner_labels is not None:
            self.ner_head = NERHead(d_model, num_ner_labels)
        else:
            self.ner_head = None

    def forward_mlm(self, input_ids, bboxes, attention_mask=None):
        """Returns logits for MLM: (B, L, vocab_size)"""
        hidden = self.encoder(input_ids, bboxes, attention_mask)
        return self.mlm_head(hidden)

    def forward_ner(self, input_ids, bboxes, attention_mask=None):
        """Returns logits for token classification: (B, L, num_labels)"""
        if self.ner_head is None:
            raise ProjectException(
                "NER head has not been initialised. Set num_ner_labels when creating RIFD."
            )
        hidden = self.encoder(input_ids, bboxes, attention_mask)
        return self.ner_head(hidden)

    def forward(self, input_ids, bboxes, attention_mask=None, task="mlm"):
        """Unified forward: task can be 'mlm' or 'ner'."""
        if input_ids.dtype == torch.float:
            input_ids.dtype = torch.long

        if task == "mlm":
            return self.forward_mlm(input_ids, bboxes, attention_mask)
        elif task == "ner":
            return self.forward_ner(input_ids, bboxes, attention_mask)
        else:
            raise ProjectException(f"Unknown task: {task}")