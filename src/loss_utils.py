import torch.nn as nn

class MLMLoss(nn.Module):
    """
    Masked Language Modelling loss: cross‑entropy on masked positions only.
    Ignores indices with label = -100.
    """
    def __init__(self, ignore_index:int=-100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    def forward(self, logits, labels):
        """
        Args:
            logits: (B, L, vocab_size)
            labels: (B, L) with -100 for non‑masked tokens
        Returns:
            scalar loss
        """
        return self.loss_fn(logits.view(-1, logits.size(-1)),labels.view(-1))

class TokenClassificationLoss(nn.Module):
    """
    Token‑level classification loss (e.g., NER BIO tagging).
    Ignores positions with label = -100.
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, L, num_labels)
            labels: (B, L) with -100 for special/padding tokens
        Returns:
            scalar loss
        """
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))