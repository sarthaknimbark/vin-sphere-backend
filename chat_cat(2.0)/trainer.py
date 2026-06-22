import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MultiTaskTrainer:
    """
    Trainer class for training and evaluating MultiTaskVINTransformer.
    
    Supports:
    - GPU acceleration
    - FP16 Mixed Precision training (on CUDA)
    - Cosine learning rate scheduling
    - Early stopping on validation loss
    - Saving and resuming checkpoint states
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterions: Dict[str, nn.Module],
        device: torch.device,
        save_dir: str = "models",
        patience: int = 5,
        model_name: str = "vin_transformer_best.pt"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterions = criterions
        self.device = device
        self.save_dir = save_dir
        self.patience = patience
        self.model_name = model_name
        
        # Checkpoint path
        self.checkpoint_path = os.path.join(save_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Mixed Precision Scaler (only for CUDA device)
        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        
        # Track training metrics
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 1

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            x_batch = batch['x'].to(self.device)
            
            # Enable Autocasting for mixed precision if CUDA is used
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits_dict = self.model(x_batch)
                    loss = 0.0
                    for target, criterion in self.criterions.items():
                        y_true = batch[target].to(self.device)
                        loss += criterion(logits_dict[target], y_true)
                
                # Scale loss and backpropagate
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits_dict = self.model(x_batch)
                loss = 0.0
                for target, criterion in self.criterions.items():
                    y_true = batch[target].to(self.device)
                    loss += criterion(logits_dict[target], y_true)
                
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item() * x_batch.size(0)
            
        epoch_loss = total_loss / len(self.train_loader.dataset)
        return epoch_loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_val_loss = 0.0
        correct_counts = {target: 0 for target in self.criterions.keys()}
        
        for batch in self.val_loader:
            x_batch = batch['x'].to(self.device)
            
            logits_dict = self.model(x_batch)
            loss = 0.0
            for target, criterion in self.criterions.items():
                y_true = batch[target].to(self.device)
                loss += criterion(logits_dict[target], y_true)
                
                # Accuracy tracking
                preds = torch.argmax(logits_dict[target], dim=1)
                correct_counts[target] += torch.sum(preds == y_true).item()
                
            total_val_loss += loss.item() * x_batch.size(0)
            
        val_loss = total_val_loss / len(self.val_loader.dataset)
        
        eval_metrics = {"val_loss": val_loss}
        for target in self.criterions.keys():
            acc = correct_counts[target] / len(self.val_loader.dataset)
            eval_metrics[f"{target}_acc"] = acc
            
        return eval_metrics

    def fit(self, max_epochs: int) -> None:
        logger.info(f"Starting model training (Device: {self.device}, FP16: {self.use_amp})")
        
        for epoch in range(self.start_epoch, max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            val_loss = metrics["val_loss"]
            
            # Step scheduler if it is a standard PyTorch scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                
            lr = self.optimizer.param_groups[0]['lr']
            
            acc_str = ", ".join([f"{t} Acc: {metrics[f'{t}_acc']:.2%}" for t in self.criterions.keys()])
            logger.info(f"Epoch {epoch}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f} | {acc_str}")
            
            # Early Stopping and Checkpointing Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
                logger.info(f"--> Saved best model checkpoint at {self.checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best validation loss: {self.best_val_loss:.4f}")
                    break
                    
    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_loss": val_loss,
            "num_classes_dict": self.model.num_classes_dict,
            "config": getattr(self.model, "config", None)
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, path: Optional[str] = None) -> None:
        load_path = path if path is not None else self.checkpoint_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint file not found at {load_path}")
            
        logger.info(f"Resuming model training from checkpoint: {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["val_loss"]
        self.start_epoch = checkpoint["epoch"] + 1
