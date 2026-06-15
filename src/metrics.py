import math
from typing import List, Optional
import numpy as np
from seqeval.metrics import f1_score as seq_f1_score, classification_report


def compute_perplexity(loss: float) -> float:
    """Convert cross‑entropy loss to perplexity. Safe upper bound for very large loss."""
    try:
        return math.exp(min(loss, 20.0))
    except OverflowError:
        return float('inf')


def compute_token_accuracy(true_labels: List[List[int]], pred_labels: List[List[int]],
                           ignore_index: int = -100) -> float:
    """
    Compute token‑level accuracy, ignoring positions with ignore_index.
    Args:
        true_labels: list of sequences of integer labels
        pred_labels: list of sequences of integer labels
        ignore_index: label value to ignore (default -100)
    Returns:
        accuracy (float)
    """
    flat_true = [t for seq in true_labels for t in seq if t != ignore_index]
    flat_pred = [p for seq in pred_labels for p in seq if p != ignore_index]
    if len(flat_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(flat_true, flat_pred) if t == p)
    return correct / len(flat_true)


def compute_entity_f1(true_labels: List[List[str]], pred_labels: List[List[str]],
                      average: str = 'micro') -> float:
    """
    Compute entity‑level F1 using seqeval.
    Args:
        true_labels: list of sequences of BIO string labels (e.g., [['O', 'B-QUESTION', ...], ...])
        pred_labels: same format
        average: 'micro', 'macro', 'weighted' – passed to seqeval
    Returns:
        F1 score (float)
    """
    try:
        return seq_f1_score(true_labels, pred_labels, average=average)
    except Exception:
        return 0.0


def compute_entity_report(true_labels: List[List[str]], pred_labels: List[List[str]]) -> str:
    """Return seqeval classification report string."""
    return classification_report(true_labels, pred_labels, digits=4)


def integers_to_label_strings(label_lists: List[List[int]], label_map: List[str],
                              ignore_index: int = -100) -> List[List[str]]:
    """
    Convert integer label sequences back to string label sequences, ignoring -100.
    Args:
        label_lists: list of sequences of integers
        label_map: list of label names (e.g., ['B-QUESTION', 'I-QUESTION', ...])
        ignore_index: value to drop
    Returns:
        list of sequences of strings
    """
    out = []
    for seq in label_lists:
        new_seq = [label_map[x] for x in seq if x != ignore_index]
        out.append(new_seq)
    return out