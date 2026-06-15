# src/inference.py
import torch
import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.logger import get_logger
from src.exceptions import ProjectException
from src.models.rifd import RIFD
from src.tokenizer_utils import load_tokenizer
from src.data.datasets import FUNSDDataset
from src.evaluate import predict_entities_for_document
from src.utils import load_config

logger = get_logger(__name__)


class DocumentInference:
    """Handles single‑document entity extraction using a trained RIFD model."""

    def __init__(self, config_path: str = "configs/config.yaml",
                 checkpoint_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.tokenizer = load_tokenizer(self.config)
        self.label_list = FUNSDDataset.get_label_list()

        # Determine device
        device_cfg = self.config.get('inference', {}).get('device', 'cpu')
        self.device = 'cuda' if device_cfg == 'cuda' and torch.cuda.is_available() else 'cpu'
        logger.info(f"Inference device: {self.device}")

        # Build model
        model_cfg = self.config['model']
        self.model = RIFD(
            vocab_size=self.tokenizer.get_vocab_size(),
            d_model=model_cfg['d_model'],
            num_heads=model_cfg['num_heads'],
            num_layers=model_cfg['num_layers'],
            d_ff=model_cfg['d_ff'],
            max_len=model_cfg['max_len'],
            dropout=model_cfg['dropout'],
            num_ner_labels=FUNSDDataset.get_num_labels()
        )

        # Load weights
        if checkpoint_path is None:
            checkpoint_path = self.config['inference'].get('model_path',
                                                          self.config['finetuning']['save_path'])
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise ProjectException(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {checkpoint_path}")

    def predict_from_image(self, image_path: str, ocr_words: List[Dict]) -> List[Dict]:
        """
        Extract entities from a document image given its OCR words.

        Args:
            image_path: Path to the document image (used for normalising boxes).
            ocr_words: List of dicts with 'text' (str) and 'box' [x0,y0,x1,y1] (pixel coords).
        Returns:
            List of entity dicts:
              {'label': str, 'text': str, 'box': [x0,y0,x1,y1], 'words': [...]}
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ProjectException(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]
        return predict_entities_for_document(self.model, ocr_words, self.tokenizer,
                                             self.label_list, (h, w), self.device)

    def predict_from_funsd_annotation(self, image_path: str, annotation_path: str) -> List[Dict]:
        """
        Extract entities from a FUNSD image using its ground‑truth OCR annotation.
        """
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annot = json.load(f)
        words = []
        for item in annot.get('form', []):
            for w in item.get('words', []):
                words.append({'text': w['text'], 'box': w['box']})
        return self.predict_from_image(image_path, words)

    def extract_to_json(self, entities: List[Dict], indent: int = 2) -> str:
        """Convert entity list to pretty JSON string."""
        return json.dumps(entities, indent=indent, ensure_ascii=False)

# Standalone test (executed when file is run directly)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("Testing inference module...")
    # This test requires a trained model; if you don't have one, skip or use dummy.
    # For now, we'll just demonstrate the interface with a dummy run if a model exists.
    config_path = "configs/config.yaml"
    if not Path(config_path).exists():
        raise ProjectException(f"Config not found: {config_path}")

    # Try to load the model checkpoint from config
    config = load_config(config_path)
    checkpoint = config.get('finetuning', {}).get('save_path', 'checkpoints/best-ner-model.pth')

    if not Path(checkpoint).exists():
        logger.warning(f"No checkpoint at {checkpoint}, skipping inference test.")
        sys.exit(0)

    inferencer = DocumentInference(config_path, checkpoint_path=checkpoint)

    # Pick a random FUNSD test sample (if available)
    from src.data.datasets import FUNSDDataset
    data_dir = Path(config['data']['root']) / config['data']['datasets']['finetuning']['local_dir']
    test_dataset = FUNSDDataset(data_dir, inferencer.tokenizer, max_length=512, split='test')
    if len(test_dataset) > 0:
        img_path, ann_path = test_dataset.samples[0]
        entities = inferencer.predict_from_funsd_annotation(str(img_path), str(ann_path))
        print("\nExtracted entities:")
        print(inferencer.extract_to_json(entities))
    else:
        print("No test samples found.")