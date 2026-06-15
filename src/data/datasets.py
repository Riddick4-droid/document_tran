import random
import json
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

from src.tokenizer_utils import TokenizerWrapper,load_tokenizer
from src.logger import get_logger
from src.exceptions import ProjectException
from src.utils import load_wikitext_articles


logger = get_logger(__name__)


class WikiTextLayoutDataset(Dataset):
    """
    Converts WikiText‑103 articles into pre‑training samples for training.
    Each sample is a chunk of tokens with synthetic 2D bounding boxes
    and MLM labels.
    """
    def __init__(self, data_dir, tokenizer_wrapper: TokenizerWrapper, max_length:int=512, mlm_prob: float=0.15,
                 stride: int=None, max_articles:int=None):
        self.tokenizer=tokenizer_wrapper
        self.max_length = max_length
        self.mlm_prob=mlm_prob
        self.stride=stride if stride is not None else max_lenght //2

        #load articles
        articles  = load_wikitext_articles(wiki_dir=data_dir)
        if max_articles is not None:
            articles = articles[:max_articles]
        logger.info(f"Processing {len(articles)} articles for pre-training")

        #build sample list
        self.samples: List[Dict]=[]
        for article in tqdm(articles, desc="chunking articles..."):
            if not article.strip():
                continue
            token_ids, boxes = self._article_to_tokens_and_boxes(article)
            if not token_ids:
                continue
            chunks = self._chunk_and_mask(token_ids, boxes)
            self.samples.extend(chunks)
        logger.info(f"created {len(self.samples)} pretraining samples")

    def _article_to_tokens_and_boxes(self, article:str)->Tuple[List[int], List[Tuple[float, float, float, float]]]:
        lines = article.split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        if not lines:
            return [],[]
        num_lines = len(lines)

        #tokenize the current article
        line_token_ids = []
        for line in lines:
            ids = self.tokenizer.tokenizer.encode(
                line,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length
            )
            line_token_ids.append(ids)
        max_tokens_line = max(len(ids) for ids in line_token_ids) if line_token_ids else 1

        all_token_ids = []
        all_boxes = []
        for line_idx, ids in enumerate(line_token_ids):
            for tok_idx, tid in enumerate(ids):
                x0 = tok_idx / max_tokens_line
                x1 = (tok_idx + 1) / max_tokens_line
                y0 = line_idx / num_lines
                y1 = (line_idx + 1) / num_lines
                all_token_ids.append(tid)
                all_boxes.append((x0, y0, x1, y1))
        return all_token_ids, all_boxes
    
    def _chunk_and_mask(self, token_ids: List[int], boxes: List[Tuple])->List[Dict]:
        chunks = []
        content_len = self.max_length-2
        start = 0
        while start < len(token_ids):
            end = min(start + content_len, len(token_ids))
            chunk_ids = token_ids[start:end]
            chunk_boxes = boxes[start:end]

            input_ids = [self.tokenizer.cls_token_id] + chunk_ids + [self.tokenizer.sep_token_id]
            cls_box = (0.0,0.0,1.0,1.0)
            sep_box = (0.0,0.0,1.0,1.0)
            bboxes = [cls_box] + chunk_boxes +[sep_box]
            attention_mask = [1] * len(input_ids)

            #pad id needed
            pad_len = self.max_length - len(input_ids) #if 0 no padding

            if pad_len > 0:
                input_ids += [self.tokenizer.pad_token_id] * pad_len #create a pad of lenght n of whatever the diff b/n max_len and input id
                bboxes += [(0.0,0.0,0.0,0.0)] * pad_len
                attention_mask += [0] * pad_len

            labels = self._create_mlm_labels(input_ids)
            chunks.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "bboxes": torch.tensor(bboxes, dtype=torch.float),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })
            start += self.stride
        return chunks
    
    def _create_mlm_labels(self, input_ids:List[int])->List[int]:
        labels = [-100] * len(input_ids)
        candidate_indices = [
            i for i , tid in enumerate(input_ids) if tid not in (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id)
        ]
        random.shuffle(candidate_indices)
        num_masks = max(1, int(len(candidate_indices) * self.mlm_prob))
        mask_indices = candidate_indices[:num_masks]

        for idx in mask_indices:
            labels[idx] = input_ids[idx]
            r = random.random()
            if r < 0.8:
                input_ids[idx] = self.tokenizer.mask_token_id
            elif r < 0.9:
                input_ids[idx] = random.randrange(self.tokenizer.vocab_size)
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    

class FUNSDDataset(Dataset):
    """
    Loads FUNSD documents for token classification (NER).
    Returns token IDs, normalised bounding boxes, attention mask, and BIO labels.
    """
    ENTITY_TO_BIO = {
        "question": 0,
        "answer": 2,
        "header":4,
        "other":6
    }
    IGNORE_INDEX = -100

    def __init__(self, data_dir:Path, tokenizer_wrapper:TokenizerWrapper, max_length:int=512, split:str="train"):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer_wrapper
        self.max_length = max_length
        self.split = split
        self.samples : List[Tuple[Path, Path]] = []
        self._find_samples()
    
    def _find_samples(self):
        """Locate image‑annotation pairs, filtering by split."""
        for subdir_name, target_split in [("training_data", "train"), ("testing_data", "test")]:
            if target_split != self.split:
                continue
            subdir = self.data_dir / subdir_name
            if not subdir.exists():
                subdir = self.data_dir / "dataset"/ subdir_name
            if not subdir.exists():
                continue
            img_dir = subdir / "images"
            annot_dir = subdir / "annotations"

            if not img_dir.exists() or not annot_dir.exists():
                img_dir = subdir
                annot_dir = subdir
            for img_path in sorted(img_dir.glob("*.png")):
                annot_path = annot_dir / f"{img_path.stem}.json"
                if not annot_path.exists():
                    found = list(subdir.rglob(f"{img_path.stem}.json"))
                    if found:
                        annot_path = found[0]
                    else:
                        continue
                self.samples.append((img_path,annot_path))
        if not self.samples:
            all_imgs = sorted(self.data_dir.rglob("*.png"))
            for img_path in all_imgs:
                annot_path = img_path.with_suffix(".json")
                if not annot_path.exists():
                    found = list(self.data_dir.rglob(f"{img_path.stem}.json"))
                if found:
                    annot_path = found[0]
                else:
                    continue
        logger.info(f"FUNSD {self.split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, annot_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ProjectException(f"Failed to load image {img_path}")
        h,w,_ = img.shape

        with open(annot_path, "r", encoding="utf-8") as f:
            annot = json.load(f)
        #build list of words with normalized boxes and BIO labels
        words = []
        for form_item in annot.get("form",[]):
            entity_label = form_item.get("label","other")
            word_list = form_item.get("words",[])
            for i, word_info in enumerate(word_list):
                box = word_info["box"] #pixel coords
                norm_box = [box[0]/w, box[1]/h, box[2]/w, box[3]/h]
                bio_prefix = "B" if i==0 else "I"
                label_str = f"{bio_prefix}-{entity_label.upper()}"
                words.append({
                    "text":word_info["text"],
                    "box":norm_box,
                    "label":label_str
                })
        return self._tokenize_and_align(words)
    
    def _tokenize_and_align(self, words:List[Dict])->Dict:
        input_ids = [self.tokenizer.cls_token_id]
        bboxes = [[0.0,0.0,1.0,1.0]]
        labels = [self.IGNORE_INDEX]

        for w in words:
            sub_ids = self.tokenizer.tokenizer.encode(w["text"], add_special_tokens=False)
            if not sub_ids:
                continue
            bio_label = w["label"]
            label_int = self._bio_label_to_int(bio_label)

            for i, tid in enumerate(sub_ids):
                input_ids.append(tid)
                bboxes.append(w["box"])
                if i == 0:
                    labels.append(label_int)
                else:
                    # Subword after first gets I- variant
                    if label_int % 2 == 0:  # B- -> I-
                        labels.append(label_int + 1)
                    else:
                        labels.append(label_int)

        # Add [SEP]
        input_ids.append(self.tokenizer.sep_token_id)
        bboxes.append([0.0, 0.0, 1.0, 1.0])
        labels.append(self.IGNORE_INDEX)

        # Truncate / pad
        seq_len = len(input_ids)
        if seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
            bboxes = bboxes[:self.max_length]
            labels = labels[:self.max_length]
        else:
            pad_len = self.max_length - seq_len
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            bboxes += [[0.0, 0.0, 0.0, 0.0]] * pad_len
            labels += [self.IGNORE_INDEX] * pad_len
        # Safety: force to exact max_length (in case of any off-by-one)
        input_ids = input_ids[:self.max_length]
        bboxes = bboxes[:self.max_length]
        labels = labels[:self.max_length]
        attention_mask = [1] * min(seq_len, self.max_length) + [0] * max(0, self.max_length - seq_len)

        # Final assert
        assert len(input_ids) == self.max_length, f"input_ids length {len(input_ids)} != {self.max_length}"
        assert len(bboxes) == self.max_length
        assert len(labels) == self.max_length
        assert len(attention_mask) == self.max_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "bboxes": torch.tensor(bboxes, dtype=torch.float),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _bio_label_to_int(self, bio_str: str) -> int:
        bio, entity = bio_str.split("-")
        entity = entity.lower()
        base = self.ENTITY_TO_BIO[entity]
        return base if bio == "B" else base + 1

    @classmethod
    def get_num_labels(cls) -> int:
        return 8

    @classmethod
    def get_label_list(cls) -> List[str]:
        entities = ["QUESTION", "ANSWER", "HEADER", "OTHER"]
        return [f"{b}-{e}" for e in entities for b in ["B", "I"]]



