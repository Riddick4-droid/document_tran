import torch
import numpy as np
import cv2
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.logger import get_logger
from src.exceptions import ProjectException
from src.metrics import (compute_token_accuracy, 
                         compute_entity_f1,
                         compute_entity_report, 
                         integers_to_label_strings)
from src.data.datasets import FUNSDDataset
from src.tokenizer_utils import TokenizerWrapper
from src.utils import load_config, random_funsd_visualization

logger = get_logger(__name__)

def evaluate_model(model, dataloader, label_list: List[str], device: str = "cpu") -> Dict:
    model.eval()
    all_int_labels = []
    all_int_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            bboxes = batch['bboxes'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model.forward_ner(input_ids, bboxes, attention_mask)
            preds = torch.argmax(logits, dim=-1)             # (B, L)
            
            # Convert to lists for per‑sample handling
            labels_list = labels.cpu().tolist()              # list of sequences
            preds_list = preds.cpu().tolist()

            # Mask predictions on ignored positions (padding / special tokens)
            for i in range(len(labels_list)):
                for j in range(len(labels_list[i])):
                    if labels_list[i][j] == FUNSDDataset.IGNORE_INDEX:
                        preds_list[i][j] = FUNSDDataset.IGNORE_INDEX

            all_int_preds.extend(preds_list)
            all_int_labels.extend(labels_list)

    # Convert to BIO strings (ignore_index = -100 ensures they are filtered)
    true_str = integers_to_label_strings(all_int_labels, label_list, ignore_index=FUNSDDataset.IGNORE_INDEX)
    pred_str = integers_to_label_strings(all_int_preds, label_list, ignore_index=FUNSDDataset.IGNORE_INDEX)

    token_acc = compute_token_accuracy(all_int_labels, all_int_preds)
    entity_f1 = compute_entity_f1(true_str, pred_str, average='micro')
    report = compute_entity_report(true_str, pred_str)

    return {
        "token_accuracy": token_acc,
        "entity_f1_micro": entity_f1,
        "classification_report": report,
        "true_labels": true_str,
        "pred_labels": pred_str,
    }

def pred_entities_for_documents(model, words:List[Dict], tokenizer: TokenizerWrapper, label_list: List[str], img_shape: Tuple[int,int],
                                device: str= "cpu")->List[Dict]:
    """
    Infer entities for a single document given its words (with pixel boxes).
    Returns list of predicted entities with merged bounding boxes.
    """
    h, w = img_shape
    # Build input sequence (as in FUNSD dataset logic)
    input_ids = [tokenizer.cls_token_id]
    bboxes = [[0.0, 0.0, 1.0, 1.0]]  # [CLS]
    subword_map = []
    for word_idx, word in enumerate(words):
        box = word['box']
        norm_box = [box[0]/w, box[1]/h, box[2]/w, box[3]/h]
        sub_ids = tokenizer.tokenizer.encode(word['text'], add_special_tokens=False)
        for tid in sub_ids:
            input_ids.append(tid)
            bboxes.append(norm_box)
            subword_map.append(word_idx)
    input_ids.append(tokenizer.sep_token_id)
    bboxes.append([0.0, 0.0, 1.0, 1.0])

    input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(device)
    bboxes_t = torch.tensor([bboxes], dtype=torch.float).to(device)
    attn_t = torch.ones_like(input_ids_t)

    model.eval()
    with torch.no_grad():
        logits = model.forward_ner(input_ids_t, bboxes_t, attn_t)
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    # Map subword predictions back to words (first subword label)
    word_labels = [None] * len(words)
    for i, p in enumerate(preds):
        if i == 0 or i == len(preds) - 1:  # skip CLS/SEP
            continue
        w_idx = subword_map[i - 1]  # offset for CLS
        if word_labels[w_idx] is None:
            word_labels[w_idx] = p

    # BIO decoding into entities
    entities = []
    current = None
    for w_idx, (lbl, word) in enumerate(zip(word_labels, words)):
        if lbl is None:
            continue
        tag = label_list[lbl]
        if tag.startswith('B-'):
            if current:
                entities.append(current)
            ent_type = tag[2:]
            current = {'label': ent_type, 'text': word['text'],
                       'box': word['box'][:], 'words': [word]}
        elif tag.startswith('I-'):
            ent_type = tag[2:]
            if current and current['label'] == ent_type:
                current['text'] += ' ' + word['text']
                cb = current['box']
                wb = word['box']
                current['box'] = [min(cb[0], wb[0]), min(cb[1], wb[1]),
                                  max(cb[2], wb[2]), max(cb[3], wb[3])]
                current['words'].append(word)
            else:
                if current:
                    entities.append(current)
                current = {'label': ent_type, 'text': word['text'],
                           'box': word['box'][:], 'words': [word]}
        else:  # 'O'
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    return entities

def visualize_side_by_side(img_path: Path, true_entities: List[Dict],
                           pred_entities: List[Dict],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ground truth vs predicted entities on the same document image.
    Entities are dicts with 'label', 'box' (pixel coords).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ProjectException(f"Could not load image {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    colors = {"QUESTION": "blue", "ANSWER": "green", "HEADER": "orange", "OTHER": "black"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for ax, ents, title in zip(axes, [true_entities, pred_entities],
                                ["Ground Truth", "Predicted"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        for ent in ents:
            box = ent['box']
            color = colors.get(ent['label'], 'black')
            rect = patches.Rectangle((box[0], box[1]),
                                     box[2]-box[0], box[3]-box[1],
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1]-5, ent['label'], color=color, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def run_full_evaluation(config: Dict, model_path: Optional[str] = None,
                        visualize_samples: int = 3, output_dir: Optional[str] = "evaluation", dummy:bool=False):
    """
    High-level function: loads test data, model, runs evaluation,
    prints metrics and saves visualizations.
    If `dummy` is True, creates a randomly initialized model (for testing).
    """
    from src.models.rifd import RIFD
    from src.tokenizer_utils import load_tokenizer

    data_dir = Path(config['data']['root']) / config['data']['datasets']['finetuning']['local_dir']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = load_tokenizer(config)
    label_list = FUNSDDataset.get_label_list()

    # Dataset
    test_dataset = FUNSDDataset(data_dir, tokenizer,
                                max_length=config['data']['datasets']['finetuning'].get("max_length", 512),
                                split='test')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Model
    vocab_size = tokenizer.get_vocab_size()
    model_cfg = config['model']
    model = RIFD(
        vocab_size=vocab_size,
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        d_ff=model_cfg['d_ff'],
        max_len=model_cfg['max_len'],
        dropout=model_cfg['dropout'],
        num_ner_labels=FUNSDDataset.get_num_labels()
    )

    if dummy:
        logger.warning("Using random model weights (dummy=True). Results will be meaningless.")
    else:
        if model_path is None:
            model_path = config['finetuning']['save_path']
        if not Path(model_path).exists():
            raise ProjectException(f"Model checkpoint not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    logger.info("Running evaluation...")
    results = evaluate_model(model, test_loader, label_list, device)

    print("\n=== Evaluation Results ===")
    print(f"Token accuracy: {results['token_accuracy']:.4f}")
    print(f"Entity F1 (micro): {results['entity_f1_micro']:.4f}")
    print("\nClassification report:")
    print(results['classification_report'])

    # Visualize random samples
    if visualize_samples > 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating {visualize_samples} visualizations...")
        indices = random.sample(range(len(test_dataset)), min(visualize_samples, len(test_dataset)))
        for idx in indices:
            img_path, ann_path = test_dataset.samples[idx]
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            with open(ann_path, 'r') as f:
                annot = json.load(f)

            words = []
            true_ents = []
            for item in annot.get('form', []):
                lbl = item.get('label', 'other').upper()
                field_words = item.get('words', [])
                if not field_words:
                    continue
                boxes = [wd['box'] for wd in field_words]
                merged = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                          max(b[2] for b in boxes), max(b[3] for b in boxes)]
                true_ents.append({
                    'label': lbl,
                    'text': ' '.join(wd['text'] for wd in field_words),
                    'box': merged
                })
                for wd in field_words:
                    words.append({'text': wd['text'], 'box': wd['box']})

            pred_ents = pred_entities_for_documents(model, words, tokenizer, label_list,
                                                      (h, w), device)
            save_file = output_dir / f"compare_{img_path.stem}.png"
            visualize_side_by_side(img_path, true_ents, pred_ents, save_path=str(save_file))
            logger.info(f"Saved {save_file}")

if __name__ == "__main__":
    import sys
    # Add project root to path so imports work
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("Running full evaluation test...")
    config_path = "config/configs.yaml"
    if not Path(config_path).exists():
        raise ProjectException(f"Config file not found: {config_path}")

    config = load_config(config_path)

    # Use dummy=True to test without a trained model; set to False when you have a checkpoint
    run_full_evaluation(config, visualize_samples=2, output_dir="evaluation_test", dummy=True)