import argparse
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.logger import get_logger
from src.utils import load_config
from src.exceptions import ProjectException
from src.data.data_ingestion import download_datasets
from src.tokenizer_utils import load_tokenizer
from src.models.rifd import RIFD
from src.loss_utils import MLMLoss, TokenClassificationLoss
from src.data.datasets import WikiTextLayoutDataset, FUNSDDataset
from src.trainer import PreTrainer, FineTuner

logger = get_logger(__name__)

def train(args):
    """Run pre‑training or fine‑tuning according to the provided arguments."""
    config = load_config(args.config)
    if args.data_dir:
        config["data"]["root"] = args.data_dir
    elif "SM_CHANNEL_TRAINING" in os.environ:
        #sagemaker copies training data to this directory#
        config["data"]["root"] = os.environ["SM_CHANNEL_TRAINING"]

    #overide model save path with sagemaker model dir
    if "SM_MODEL_DIR" in os.environ:
        model_dir = Path(os.environ["SM_MODEL_DIR"])
    elif args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = Path("checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)

# Override hyperparameters if provided
    if args.epochs is not None:
        config[args.task]["epochs"] = args.epochs
    if args.batch_size is not None:
        config[args.task]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config[args.task]["learning_rate"] = args.learning_rate

    # Ensure datasets are present (idempotent)
    download_datasets(config)

    # ---- Tokenizer ----
    tokenizer = load_tokenizer(config)

    # ---- Task-specific setup ----
    if args.task == "pretrain":
        data_cfg = config["data"]["datasets"]["pretraining"]
        wiki_dir = Path(config["data"]["root"]) / data_cfg["local_dir"]

        dataset = WikiTextLayoutDataset(
            data_dir=wiki_dir,
            tokenizer_wrapper=tokenizer,
            max_length=data_cfg.get("max_length", 512),
            mlm_prob=data_cfg.get("mlm_prob", 0.15),
            stride=data_cfg.get("stride", 256),
            max_articles=data_cfg.get("max_articles"),
        )

        # Train/validation split
        val_split = config["pretraining"].get("val_split", 0.1)
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Model without NER head
        model_cfg = config["model"]
        model = RIFD(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=model_cfg["d_model"],
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            d_ff=model_cfg["d_ff"],
            max_len=model_cfg["max_len"],
            dropout=model_cfg["dropout"],
            num_ner_labels=None,
        )

        loss_fn = MLMLoss()
        config["pretraining"]["save_path"] = str(model_dir / "best-dpt-model.pth")

        trainer = PreTrainer(
            model=model,
            config=config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            loss_fn=loss_fn,
            tokenizer_wrapper=tokenizer,
        )

    elif args.task == "finetune":
        data_cfg = config["data"]["datasets"]["finetuning"]
        data_dir = Path(config["data"]["root"]) / data_cfg["local_dir"]

        train_dataset = FUNSDDataset(
            data_dir=data_dir,
            tokenizer_wrapper=tokenizer,
            max_length=data_cfg.get("max_length", 512),
            split="train",
        )
        val_dataset = FUNSDDataset(
            data_dir=data_dir,
            tokenizer_wrapper=tokenizer,
            max_length=data_cfg.get("max_length", 512),
            split="test",
        )

        label_list = FUNSDDataset.get_label_list()

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

        # Load pre‑trained encoder if provided
        if args.pretrained_checkpoint:
            ckpt = Path(args.pretrained_checkpoint)
            if ckpt.exists():
                logger.info(f"Loading encoder from {ckpt}")
                state = torch.load(ckpt, map_location="cpu")
                encoder_state = {k: v for k, v in state.items() if not k.startswith("ner_head")}
                missing, unexpected = model.load_state_dict(encoder_state, strict=False)
                logger.info(f"Missing (NER) keys: {len(missing)}")
            else:
                logger.warning(f"Pre‑trained checkpoint not found: {ckpt}")

        loss_fn = TokenClassificationLoss()
        config["finetuning"]["save_path"] = str(model_dir / "best-ner-model.pth")

        trainer = FineTuner(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss_fn=loss_fn,
            label_list=label_list,
            tokenizer_wrapper=tokenizer,
        )
    else:
        raise ProjectException(f"Unknown task: {args.task}")

    # ---- Run training ----
    trainer.train()

    # ---- Save model for SageMaker inference ----
    # SageMaker expects the model at model_dir (SM_MODEL_DIR)
    # The best model is already saved during training by the trainer.
    # We can also copy the tokenizer configuration if needed, but for inference we'll rely on our code.
    logger.info(f"Training finished. Model artifacts saved to {model_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker training script for RIFD")

    # Essential paths
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to project config YAML")
    parser.add_argument("--task", type=str, required=True, choices=["pretrain", "finetune"],
                        help="Training phase")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data root directory (default: config or SM_CHANNEL_TRAINING)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Override model output directory (default: config or SM_MODEL_DIR)")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None,
                        help="Pre‑trained encoder checkpoint (for fine‑tuning)")

    # Hyperparameter overrides (optional)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)