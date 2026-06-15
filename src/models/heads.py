import torch.nn as nn


class MLMHead(nn.Module):
    """
    Linear projection for Masked Language Modelling.
    Maps encoder hidden states to vocabulary logits.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states):
        """hidden_states: (B, L, d_model) → (B, L, vocab_size)"""
        return self.linear(hidden_states)


class NERHead(nn.Module):
    """
    Linear classifier for token‑level entity extraction (BIO tagging).
    """
    def __init__(self, d_model: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, hidden_states):
        """hidden_states: (B, L, d_model) → (B, L, num_labels)"""
        return self.classifier(hidden_states)