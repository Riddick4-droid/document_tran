import time
import math
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any
import numpy as np
import random
import os

from src.logger import get_logger
from src.exceptions import ProjectException
from src.metrics import compute_perplexity, compute_token_accuracy, compute_entity_f1, integers_to_label_strings
from src.data.datasets import FUNSDDataset

logger = get_logger(__name__)

class BaseTrainer:
    """Common training boilerplate for RIFD pre‑training and fine‑tuning."""
    def __init__(self,model:torch.nn.Module, config:Dict[str,Any], task: str):
        self.model = model
        self.config = config
        self.task = task #one of two- either pretraining or finetuning

        #setup the training configs
        train_cfg = config[task] #this will retrieve the task-specific configs
        self.batch_size = train_cfg["batch_size"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        self.grad_clip = train_cfg["grad_clip"] #prevents exploding gradients during backprop
        self.num_epochs = train_cfg["epochs"]
        self.save_path = train_cfg["save_path"]
        self.log_dir = train_cfg["log_dir"]
        self.val_subset_fraction = train_cfg.get("val_subset_fraction", 1.0) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #logging and tracking
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=True)
        self.best_metric = None #val loss for pretrain, entity F1 for finetune
        self.best_metric_name = "val_loss" if self.task == "pretrain" else "entity_f1"

        logger.info(f" Training on Device: {self.device}")
        logger.info(f"Batch size of per step: {self.batch_size}, Epochs: {self.num_epochs}, LR: {self.lr}")

        # placeholder for dataloaders – set in subclasses
        self.train_loader = None
        self.val_loader = None
        self.loss_fn = None

    def _log_dataset_sizes(self, train_dataset, val_dataset):
        """Logs the size of datasets used during training"""
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

    def _prepare_dataloader(self, train_dataset, val_dataset):
        # Apply random subset to validation set if fraction < 1.0
        if self.val_subset_fraction < 1.0:
            val_size = max(1, int(len(val_dataset)*self.val_subset_fraction))
            indices = random.sample(range(len(val_dataset)),val_size)
            val_dataset = Subset(val_dataset, indices)
            logger.info(f"Using validation subset of size {val_size} (fraction={self.val_subset_fraction})")

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=False, # using cpu
            drop_last=True #drops the last incomplete or odd batch
        )

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=False, # using cpu
            drop_last=True #drops the last incomplete or odd batch
        )
        self._log_dataset_sizes(train_dataset=train_dataset, val_dataset=val_dataset)

    def _train_one_epoch(self, epoch:int)->float:
        """train only one epoch setup. usually expanded in training step by the argument: 'epoch:int'"""
        self.model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            bboxes = batch["bboxes"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            self.model.zero_grad(set_to_none=True) #set model gradients to zero after each epoch
            with torch.amp.autocast(device_type=self.device, enabled=True):
                logits = self.model(
                    input_ids = input_ids,
                    bboxes = bboxes,
                    attention_mask = attention_mask,
                    task = "mlm" if self.task == "pretrain" else "ner"
                ) #the rifd model takes these as input in the forward pass
                loss = self.loss_fn(logits, labels) #task-specific loss, see loss_utils.py for more details
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": total_loss/steps}) #step level loss logging-so while training we see each step's loss
        avg_loss = total_loss / steps if steps > 0 else 0.0
        return avg_loss
    
    #setup the validation phase
    @torch.no_grad()
    def _validate(self, epoch:int)->Tuple[float, Dict[str, float]]: #return the val loss as well as other metrics
        self.model.eval()
        total_loss = 0.0 #val loss
        steps  = 0

        # For NER metrics we may need to collect predictions
        all_int_labels = []
        all_int_preds = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            bboxes = batch['bboxes'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device , enabled=True):
                logits = self.model(
                    input_ids=input_ids,
                    bboxes=bboxes,
                    attention_mask=attention_mask,
                    task="mlm" if self.task == "pretrain" else "ner"
                )
                loss = self.loss_fn(logits, labels)
            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"val_loss": total_loss / steps})
            
            if self.task == "finetune":
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                all_int_preds.extend(preds)
                all_int_labels.extend(labels.cpu().tolist())
        avg_loss = total_loss / steps if steps > 0 else 0.0
        metrics = {"val_loss":avg_loss}

        if self.task == "pretrain":
            metrics["val_perplexity"] = compute_perplexity(avg_loss)
        else:
            #compute token accuracy and entity f1
            label_list = FUNSDDataset.get_label_list()
            true_str = integers_to_label_strings(all_int_labels, label_list, ignore_index=-100)
            pred_str = integers_to_label_strings(all_int_preds, label_list, ignore_index=-100)
            token_acc = compute_token_accuracy(all_int_labels, all_int_preds)
            entity_f1 = compute_entity_f1(true_str, pred_str)
            metrics["token_accuracy"] = token_acc
            metrics["entity_f1"] = entity_f1
        return avg_loss, metrics
    
    def _check_overfitting(self, train_loss: float, val_loss:float)->bool:
        """hueristic to check overfitting if 
        val_loss > 1.5(can be set depending on preference-kept low toensure strictness) * train loss and both are low"""
        if train_loss < 1.0 and val_loss > 1.5 * train_loss:
            logger.warning(f"Possible overfitting detected! (val_loss >> train_loss)")
            return True
        return False
    
    def _check_underfitting(self, train_loss: float, val_loss:float, epoch:int)->bool:
        """Heuristic: underfitting if losses are high and not improving much over epochs."""
        if epoch > 2 and train_loss > 2.0 and val_loss > 2.0: # note that at a low epoch value the model has not learned anything substantial yet
            if hasattr(self, "_last_train_loss"):
                if abs(train_loss - self._last_train_loss) / self._last_train_loss < 0.05:
                    logger.warning("Possible underfitting detected! (high loss, slow decrease)")
                    return True
        self._last_train_loss = train_loss
        return False
    
    def train(self):
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs #max number of iterations
        )
        logger.info("Starting training...")

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_one_epoch(epoch=epoch)
            val_loss, val_metrics = self._validate(epoch=epoch)

            self.scheduler.step()

            #log metircs to tb
            global_step = epoch * len(self.train_loader) #total number of steps in general
            self.writer.add_scalar(tag="Loss/train", scalar_value=train_loss, global_step=global_step)
            self.writer.add_scalar(tag="Loss/Val", scalar_value=val_loss, global_step=global_step)
            for k, v in val_metrics.items():
                self.writer.add_scalar(k,v, global_step)
            
            #print epoch sumaary
            logger.info(f"Epoch {epoch}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                + (f" | Val PPL: {val_metrics.get('val_perplexity', 0.0):.2f}" if self.task == "pretrain" else
                   f" | Token Acc: {val_metrics.get('token_accuracy', 0.0):.4f} | Entity F1: {val_metrics.get('entity_f1', 0.0):.4f}")
                   )
            
            #over and underfitting check
            self._check_overfitting(train_loss=train_loss, val_loss=val_loss)
            self._check_underfitting(train_loss=train_loss, val_loss=val_loss, epoch=epoch)

            #save best model based on chosen metric
            current_metric = val_metrics.get(self.best_metric_name, val_loss)
            if self.best_metric is None or (
                (self.task == "pretrain" and current_metric < self.best_metric) or (self.task == "finetune" and current_metric > self.best_metric)
            ):
                self.best_metric = current_metric
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f"->Best model saved ({self.best_metric_name}={current_metric:.4f})")
        self.writer.close()
        logger.info("Training Complete!!!!!")

#define task-specific subclasses that inherit from the basetrainer
class Pretrainer(BaseTrainer):
     """MLM pre‑training on WikiText‑103 with synthetic layout."""
     def __init__(self, model, config: Dict[str, Any], train_dataset, val_dataset, loss_fn, tokenizer_wrapper):
         super().__init__(model, config, "pretraining")
         self.loss_fn = loss_fn
         self.tokenizer = tokenizer_wrapper
         self._prepare_dataloader(train_dataset, val_dataset)



class FineTuner(BaseTrainer):
     """MLM pre‑training on WikiText‑103 with synthetic layout."""
     def __init__(self, model, config: Dict[str, Any], train_dataset, val_dataset, loss_fn,label_list, tokenizer_wrapper):
         super().__init__(model, config, "finetuning")
         self.loss_fn = loss_fn
         self.tokenizer = tokenizer_wrapper
         self._prepare_dataloader(train_dataset, val_dataset)

         #ensuring NER head exists and used
         if model.ner_head is None:
             raise ProjectException(f"Model does not have an NER head; set num_ner_labels when creating RIFD.")
         if model.ner_head.classifier.out_features != len(label_list):
             raise ProjectException(
                 f"NER head expects {model.ner_head.classifier.out_features} labels, "
                f"but dataset provides {len(label_list)}"
             )
