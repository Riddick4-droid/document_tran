import yaml
import torch
import numpy as np
import random
import json
import cv2
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Optional, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from prettytable import PrettyTable
from src.exceptions import ProjectException

#loading configurations
def load_config(config_path:str)->Dict[str,Any]:
    """Load YAML configuration file"""
    path = Path(config_path)
    if not path.exists():
        raise ProjectException(f"config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
#define model info retriever
def get_model_info(model:torch.nn.Module)->str:
    """Return a string with total trainable parameters and a Prettytable
    listing each module's parameter count and memory size"""

    table = PrettyTable(["Module","Parameters","Size"])
    total_params = 0
    total_bytes = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num = param.numel()
        bytes_ = num * param.element_size()
        if bytes_ < 1024:
            size_str = f"{bytes_} B"
        elif bytes_ < 1024 ** 2:
            size_str = f"{bytes_/1024:.2f} KB"
        else:
            size_str = f"{bytes_/(1024 ** 2):.4f} MB"
        table.add_row([name, num, size_str])
        total_params += num
        total_bytes += bytes_
    total_mb = total_bytes / (1024 ** 2)
    info = f"Total Trainable parameters: {total_params:,}\n"
    info += f"Parameter memory (fp32): {total_mb:.2f} MB\n"
    info += str(table)
    return info


# Helper: load WikiText‑103 articles
def load_wikitext_articles(wiki_dir: Path) -> List[str]:
    """Load WikiText‑103 articles from the downloaded directory."""
    train_file = None
    for pattern in ["wiki.train.tokens", "train.tokens", "wiki.train.raw"]:
        matches = list(wiki_dir.rglob(pattern))
        if matches:
            train_file = matches[0]
            break
    if train_file is None:
        raise ProjectException(f"No training file found in {wiki_dir}")

    articles = []
    current_lines = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n\r')
            if line.strip() == '':   # blank or whitespace-only line = article boundary
                if current_lines:
                    articles.append('\n'.join(current_lines))
                    current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            articles.append('\n'.join(current_lines))
    print(f"Loaded {len(articles)} articles from {train_file}")
    return articles

#synthetic layout visualization(for wikitext 103)
def visualize_synthetic_layout(article:str,tokenizer_wrapper, save_path:Optional[str]=None,figsize: Tuple[int,int]=(12,8))->plt.Figure:
    """
    Draw a 2D grid of the article's tokens and their normalised bounding boxes.
    The article is split by newlines; each line is tokenised into subwords.
    """
    lines = article.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        raise ProjectException("cannot visualize an empty article")
    num_lines = len(lines)

    #tokenize each line (with truncation)
    line_token_ids = []
    for line in lines:
        ids = tokenizer_wrapper.tokenizer.encode(
            line,
            add_special_tokens=False,
            truncation=True,
            max_length=tokenizer_wrapper.max_length
        )
        line_token_ids.append(ids)
    max_tokens_line = max(len(ids) for ids in line_token_ids) if line_token_ids else 1

    #build normalized boxes and decoded token strings
    boxes = []
    texts = []
    for line_idx, ids in enumerate(line_token_ids):
        for tok_idx, tid in enumerate(ids):
            x0 = tok_idx / max_tokens_line
            x1 = (tok_idx+1)/max_tokens_line
            y0 = line_idx / num_lines
            y1 = (line_idx+1)/num_lines
            boxes.append((x0,y0,x1,y1))
            texts.append(tokenizer_wrapper.decode([tid], skip_special_tokens=False))
    #visualization step
    fig,ax = plt.subplots(figsize=figsize)
    for bbox, token in zip(boxes, texts):
        x0,y0,x1,y1 = bbox
        width = x1-x0
        height = y1-y0
        rect = patches.Rectangle((x0,y0), width, height, linewidth=1, edgecolor="black", facecolor="lightblue", alpha=0.4)
        ax.add_patch(rect)
        display_token = token[:10] + ("..." if len(token)>10 else "")
        ax.text(x0 + width/2, y0 +height/2, display_token, ha="center", va="center", fontsize=8, clip_on=True)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks(np.linspace(0,1, max_tokens_line+1))
    ax.set_yticks(np.linspace(0,1), num_lines+1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Synthetic Layout grid", fontsize=15)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname=save_path, dpi=150, format="png")
    return fig

#eda for wikitext-103
def eda_wikitext(data_dir: Path, tokenizer_wrapper):
    """
    Print summary statistics for the WikiText-103 training set,
    plot a histogram of article lengths, and display the synthetic
    layout of the first article.
    """
    wiki_dir = data_dir / "wikitext-103"
    if not wiki_dir.exists():
        raise ProjectException(f"Wikitext-103 directory not found: {wiki_dir}")
    
    #locate the training file
    train_file = None
    for pattern in ["wiki.train.tokens", "train.tokens", "wiki.train.raw"]:
        matches = list(wiki_dir.rglob(pattern=pattern))
        if matches:
            train_file = matches[0]
            break
    if train_file is None:
            raise ProjectException(f"could not find wiki.train.tokens in wikitext-103 dataset")
    
    #read and group articles
    articles = []
    current_lines = []
    with open(train_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            if line.strip() == "":
                if current_lines:
                    articles.append("\n".join(current_lines))
            else:
                current_lines.append(line)
        if current_lines:
            articles.append("\n".join(current_lines))
    num_articles = len(articles)
    token_lengths = [len(art.split) for art in articles]
    total_tokens = sum(token_lengths)
    avg_len = total_tokens / num_articles if num_articles else 0

    print("=== WikiText-103 EDA ===")
    print(f"Articles: {num_articles}")
    print(f"Total tokens (whitespace split): {total_tokens:,}")
    print(f"Avg article length: {avg_len:.1f}, Max: {max(token_lengths):,}, Min: {min(token_lengths):,}")

    # Histogram
    plt.figure(figsize=(10, 4))
    plt.hist(token_lengths, bins=50, edgecolor='black')
    plt.title("Article Length Distribution (WikiText-103)")
    plt.xlabel("Tokens per article")
    plt.ylabel("Frequency")
    plt.show()

    # Show synthetic layout for the first article
    if articles:
        print("\nSynthetic layout of the first article:")
        fig = visualize_synthetic_layout(articles[0], tokenizer_wrapper)
        plt.show()


#eda for funsd

def eda_funsd(data_dir: Path):
    """
    Print FUNSD label distribution and word‑count histogram,
    then show a random annotated document.
    """
    funsd_dir = data_dir / "funsd"
    if not funsd_dir.exists():
        raise ProjectException(f"FUNSD directory not found: {funsd_dir}")
    all_images = list(funsd_dir.rglob("*.png"))
    if not all_images:
        raise ProjectException("No PNG images found in FUNSD dataset")
    print(f"=== FUNSD EDA: {len(all_images)} images found ===")

    # Gather statistics from up to 100 documents
    label_counter = Counter()
    word_counts = []
    for img_path in all_images[:100]:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            found = list(funsd_dir.rglob(f"{img_path.stem}.json"))
            if found:
                json_path = found[0]
            else:
                continue
        with open(json_path, 'r') as f:
            annot = json.load(f)
        wc = 0
        for item in annot.get("form", []):
            label = item.get("label", "other")
            words = item.get("words", [])
            wc += len(words)
            label_counter[label] += len(words)
        word_counts.append(wc)

    print("\nLabel distribution (word count):")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")

    # Word‑count histogram
    plt.figure(figsize=(8, 4))
    plt.hist(word_counts, bins=30, edgecolor='black')
    plt.title("Word Count per Form (FUNSD)")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.show()

#pick a random file and display it and its annotations
def random_funsd_visualization(data_dir: Path, num_samples: int = 1,
                               figsize: Tuple[int, int] = (12, 8)):
    """
    Pick random FUNSD document(s) and draw the ground‑truth bounding boxes
    with their entity labels.
    """
    funsd_dir = data_dir / "funsd"
    all_images = list(funsd_dir.rglob("*.png"))
    if not all_images:
        raise ProjectException("No images in FUNSD directory")
    selected = random.sample(all_images, min(num_samples, len(all_images)))

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Locate annotation
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            found = list(funsd_dir.rglob(f"{img_path.stem}.json"))
            if found:
                json_path = found[0]
            else:
                continue
        with open(json_path, 'r') as f:
            annot = json.load(f)

        # Flatten words with labels
        words_with_labels = []
        for item in annot.get("form", []):
            label = item.get("label", "other")
            for word in item.get("words", []):
                words_with_labels.append({
                    "text": word["text"],
                    "box": word["box"],
                    "label": label
                })

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        colors = {"question": "blue", "answer": "green",
                  "header": "orange", "other": "gray"}

        for w in words_with_labels:
            box = w["box"]
            color = colors.get(w["label"], "black")
            rect = patches.Rectangle((box[0], box[1]),
                                     box[2] - box[0],
                                     box[3] - box[1],
                                     linewidth=2, edgecolor=color,
                                     facecolor='none')
            ax.add_patch(rect)

        # Annotate a few words
        for w in words_with_labels[:20]:
            box = w["box"]
            ax.text(box[0], box[1] - 5, w["label"], fontsize=8, color='red',
                    bbox=dict(facecolor='white', alpha=0.7))

        ax.set_title(f"Sample: {img_path.name}")
        ax.axis('off')
        plt.show()

#dataframe display helper
def display_df(df, max_rows: int = 10):
    """Convenience wrapper to display a pandas DataFrame in a notebook."""
    from IPython.display import display
    display(df.head(max_rows))