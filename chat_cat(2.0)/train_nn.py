import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from feature_engineering import prepare_data
from dataset import MultiTaskVINDataset
from model import MultiTaskVINTransformer
from trainer import MultiTaskTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="UAE VIN Intelligence System - Multi-Task Transformer Training")
    parser.add_argument("--data_path", type=str, default="../Data/data_methaq(2.0).csv",
                        help="Path to the raw CSV or Parquet dataset")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Output directory for saved models and encoders")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint")
    parser.add_argument("--downsample", type=int, default=200000,
                        help="Downsample training dataset to speed up training (negative for no downsampling)")
    
    args = parser.parse_args()
    
    # 1. Check data file path
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset not found at: {args.data_path}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 2. Data Preparation and Encoding
    # prepare_data encodes and returns features X, encoded targets y, and fitted encoders dict
    X, y, encoders = prepare_data(args.data_path, args.model_dir)
    
    # Reconstruct the 17-char VINs from X POS columns
    logger.info("Reconstructing raw VIN characters from feature columns...")
    pos_cols = [f"POS_{i}" for i in range(1, 18)]
    vins = X[pos_cols].agg("".join, axis=1).values
    
    # Target keys matching the required multi-task heads (excluding engine and fuel_type)
    target_keys = ["trim", "regional_spec", "body_type"]
    y_dict = {
        "trim": y["TRIM_enc"].values,
        "regional_spec": y["REGIONAL_SPEC_enc"].values,
        "body_type": y["BODY_TYPE_enc"].values,
    }
    
    # Downsample if requested
    if args.downsample > 0 and len(vins) > args.downsample:
        logger.info(f"Downsampling dataset from {len(vins):,} to {args.downsample:,}...")
        np.random.seed(42)
        sampled_indices = np.random.choice(len(vins), size=args.downsample, replace=False)
        vins = vins[sampled_indices]
        y_dict = {k: v[sampled_indices] for k, v in y_dict.items()}
        
    # 3. Train / Validation Split
    logger.info("Splitting dataset into train and validation sets...")
    indices = np.arange(len(vins))
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42)
    logger.info(f"Train split size: {len(train_idx):,}, Val split size: {len(val_idx):,}")
    
    # 4. Class Weights and Criterions
    criterions = {}
    for target in target_keys:
        y_target_train = y_dict[target][train_idx]
        class_counts = np.bincount(y_target_train)
        # Ensure zero-class safety
        class_counts = np.maximum(class_counts, 1)
        total_samples = len(y_target_train)
        num_classes = len(encoders[target.upper()].classes_)
        
        # In case the bincount size is less than total classes, pad with 1
        if len(class_counts) < num_classes:
            padded_counts = np.ones(num_classes)
            padded_counts[:len(class_counts)] = class_counts
            class_counts = padded_counts
            
        # Balanced class weights: total_samples / (num_classes * count)
        weights = total_samples / (num_classes * class_counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        
        criterions[target] = nn.CrossEntropyLoss(weight=weights_tensor)
        logger.info(f"Calculated class weights for target '{target}' with {num_classes} classes.")
        
    # 5. Create Datasets and DataLoaders
    train_dataset = MultiTaskVINDataset(vins[train_idx], {k: v[train_idx] for k, v in y_dict.items()})
    val_dataset = MultiTaskVINDataset(vins[val_idx], {k: v[val_idx] for k, v in y_dict.items()})
    
    # Using pin_memory if training on GPU
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_mem, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_mem, num_workers=0)
    
    # 6. Instantiate Model, Optimizer and Cosine Scheduler
    num_classes_dict = {target: len(encoders[target.upper()].classes_) for target in target_keys}
    model = MultiTaskVINTransformer(
        num_classes_dict=num_classes_dict,
        vocab_size=39,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        hidden_dim=64
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 7. Training Execution
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterions=criterions,
        device=device,
        save_dir=args.model_dir,
        patience=args.patience,
        model_name="vin_transformer_best.pt"
    )
    
    # Resume checkpoint if requested
    if args.resume:
        try:
            trainer.load_checkpoint()
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting training from scratch.")
            
    trainer.fit(args.epochs)
    logger.info("Transformer model training sequence finished!")

if __name__ == "__main__":
    main()
