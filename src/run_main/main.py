import argparse
import sys
from pathlib import Path

from src.logger import get_logger
from src.utils import load_config
from src.exceptions import ProjectException

logger = get_logger(__name__)

def cmd_ingest(args):
    """Download datasets as specified in config file"""
    from src.data.data_ingestion import download_datasets
    config = load_config(args.config)
    download_datasets(config=config, force=args.force)

def cmd_pretrain(args):
    """Run pretraining on WikiText-103 (MLM configs)"""
    from src.models.rifd import RIFD
    from src.tokenizer_utils import load_tokenizer
    from src.data.datasets import WikiTextLayoutDataset
    from src.loss_utils import MLMLoss
    from src.trainer import Pretrainer
    import torch

    config = load_config(args.config)
    tokenizer = load_tokenizer(config=config)
    data_cfg  = config["data"]["datasets"]["pretraining"]
    data_root = Path(config["data"]["root"])
    wiki_dir = data_root / data_cfg["local_dir"]

    #Dataset
    dataset = WikiTextLayoutDataset(data_dir=wiki_dir, tokenizer_wrapper=tokenizer, max_length=data_cfg.get("max_length",512),
                                    mlm_prob=data_cfg.get("mlm_prob", 0.15), 
                                    stride=data_cfg.get("stride",256), 
                                    max_articles=data_cfg.get("max_articles"))
    
    #train/validation splitting
    val_split = config["pretraining"].get("val_split",0.1)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    #model setup
    model_cfg = config["model"]
    model = RIFD(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        d_ff=model_cfg["d_ff"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"],
        )
    loss_fn = MLMLoss()

    trainer = Pretrainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=loss_fn,
        tokenizer_wrapper=tokenizer
    )
    trainer.train()

def cmd_finetune(args):
    """Run finetuning on FUNSD (token classification)"""
    from src.models.rifd import RIFD
    from src.tokenizer_utils import load_tokenizer
    from src.data.datasets import FUNSDDataset
    from src.loss_utils import TokenClassificationLoss
    from src.trainer import FineTuner
    import torch

    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    data_cfg = config["data"]["datasets"]["finetuning"]
    data_dir = Path(config["data"]["root"]) / data_cfg["local_dir"]

    #dataset
    train_dataset = FUNSDDataset(
        data_dir=data_dir,
        tokenizer_wrapper=tokenizer,
        max_length=data_cfg.get("max_length", 512),
        split="train", #this is the separator that tells what subset of dataset we are using during what mode
    )

    #using tests set for validation (FUNSD has no public val)
    val_dataset = FUNSDDataset(
        data_dir=data_dir,
        tokenizer_wrapper=tokenizer,
        max_length=data_cfg.get("max_length", 512),
        split="test",
    )

    label_list = FUNSDDataset.get_label_list()

    #model (with NER head)
    model_cfg = config["model"]
    model = RIFD(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        d_ff=model_cfg["d_ff"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"],
        num_ner_labels=FUNSDDataset.get_num_labels(),
    )

    #load pretraining encoder weights if available
    if args.pretrained_checkpoint:
        ckpt = Path(args.pretrained_checkpoint)
        if ckpt.exists():
            logger.info(f"Loading pretrained encoder from {ckpt}")
            state = torch.load(ckpt, map_location="cpu",weights_only=True)
            #filter out the NER keys
            encoder_state = {k:v for k,v in state.items() if not k.startswith("ner_head")}
            missing, unxpected = model.load_state_dict(encoder_state, strict=False)
            logger.info(f"Missing keys: {len(missing)} (expected: NER head)")
        else:
            logger.warning(f"Pretrained checkpoint not found: {ckpt}")
    
    loss_fn = TokenClassificationLoss()

    trainer = FineTuner(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        label_list=label_list,
        tokenizer_wrapper=tokenizer,
    )
    trainer.train()

def cmd_evaluate(args):
    """Run evaluation on the test set and generate visualizations."""
    from src.evaluate import run_full_evaluation
    config = load_config(args.config)
    run_full_evaluation(
        config,
        model_path=args.checkpoint,
        visualize_samples=args.vis_samples,
        output_dir=args.output_dir,
        dummy=False,
    )

def cmd_infer(args):
    """Run inference on a single document image with OCR annotation."""
    from src.inference import DocumentInference
    from pathlib import Path

    config = load_config(args.config)
    inferencer = DocumentInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )

    if args.annotation:
        # Use FUNSD annotation JSON to supply OCR words
        entities = inferencer.predict_from_funsd_annotation(
            args.image, args.annotation
        )
    else:
        # Expect a JSON file with OCR words (list of {"text": ..., "box": ...})
        import json
        with open(args.ocr_json, 'r') as f:
            ocr_words = json.load(f)
        entities = inferencer.predict_from_image(args.image, ocr_words)

    print(inferencer.extract_to_json(entities))

def main():
    parser = argparse.ArgumentParser(description="RIFD – Document Layout Transformer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    #common args
    def add_common_args(p):
        p.add_argument("--config", default="config/configs.yaml", help="path to the config.yaml file")
    
    #ingest
    p_ingest = subparsers.add_parser("ingest", help="Download datasets")
    add_common_args(p=p_ingest)
    p_ingest.add_argument("--force", action="store_true", help="Force re-download")

    #pretrain
    p_pretrain = subparsers.add_parser("pretrain", help="Run MLM pre‑training")
    add_common_args(p_pretrain)

    # Fine‑tune
    p_finetune = subparsers.add_parser("finetune", help="Run NER fine‑tuning")
    add_common_args(p_finetune)
    p_finetune.add_argument("--pretrained_checkpoint", default=None,
                            help="Path to pre‑trained encoder checkpoint (e.g., best-dpt-model.pth)")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate on test set")
    add_common_args(p_eval)
    p_eval.add_argument("--checkpoint", default=None, help="NER checkpoint path (uses config if omitted)")
    p_eval.add_argument("--vis_samples", type=int, default=3, help="Number of visualization samples")
    p_eval.add_argument("--output_dir", default="evaluation", help="Output directory for visualizations")

    # Infer
    p_infer = subparsers.add_parser("infer", help="Inference on a single image")
    add_common_args(p_infer)
    p_infer.add_argument("--checkpoint", required=True, help="Path to NER checkpoint")
    p_infer.add_argument("--image", required=True, help="Path to document image")
    p_infer.add_argument("--annotation", default=None, help="FUNSD annotation JSON (alternative to --ocr_json)")
    p_infer.add_argument("--ocr_json", default=None, help="JSON file with OCR words (if not using FUNSD)")

    args = parser.parse_args()

    # Dispatch
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "pretrain":
        cmd_pretrain(args)
    elif args.command == "finetune":
        cmd_finetune(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "infer":
        cmd_infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()