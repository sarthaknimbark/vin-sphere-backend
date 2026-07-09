import os
import gc
import json
import argparse
import logging
import pickle
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
from model_utils import save_fallback_model
from evaluate import evaluate_predictions
from train import split_and_align_classes, predict_from_target_model, BACKOFF_HIERARCHY

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_gcc_rules(data_path: str, save_dir: str):
    logger.info("Extracting GCC vs. Non-GCC pattern-matching rules...")
    if data_path.lower() in ("postgres", "postgresql", "db", "database"):
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from database.operations import fetch_training_data_by_length
        df = fetch_training_data_by_length(17)
    else:
        df = pd.read_csv(data_path)
    df['VIN'] = df['chassisNumber'].astype(str).str.upper().str.strip()
    df = df[df['VIN'].str.len() == 17].copy()
    
    df['regionalSpec'] = df['regionalSpec'].astype(str).str.upper().str.strip()
    df = df[df['regionalSpec'].isin(['GCC', 'NON-GCC'])].copy()
    df['make'] = df['make'].astype(str).str.upper().str.strip()
    
    # 1. Global WMI rules
    df['WMI'] = df['VIN'].str[0:3]
    wmi_groups = df.groupby(['WMI', 'regionalSpec']).size().unstack(fill_value=0)
    wmi_groups['Total'] = wmi_groups['GCC'] + wmi_groups['NON-GCC']
    
    # WMI is GCC-exclusive if all occurrences are GCC and total count > 5
    gcc_wmis = wmi_groups[(wmi_groups['NON-GCC'] == 0) & (wmi_groups['GCC'] > 5)].index.tolist()
    # WMI is Non-GCC-exclusive if all occurrences are Non-GCC and total count > 5
    ngcc_wmis = wmi_groups[(wmi_groups['GCC'] == 0) & (wmi_groups['NON-GCC'] > 5)].index.tolist()

    # 2. Make-specific rules for character positions
    make_rules = {}
    makes = df['make'].unique()
    for m in makes:
        df_m = df[df['make'] == m]
        gcc_m = df_m[df_m['regionalSpec'] == 'GCC']
        ngcc_m = df_m[df_m['regionalSpec'] == 'NON-GCC']
        
        # Only extract rules if we have sufficient samples to be statistically sound
        if len(gcc_m) < 5 or len(ngcc_m) < 5:
            continue
            
        make_rules[m] = {}
        for pos in range(17):
            chars_gcc = set(gcc_m['VIN'].str[pos].unique())
            chars_ngcc = set(ngcc_m['VIN'].str[pos].unique())
            
            overlap = chars_gcc.intersection(chars_ngcc)
            only_gcc = chars_gcc - overlap
            only_ngcc = chars_ngcc - overlap
            
            if only_gcc or only_ngcc:
                make_rules[m][f"POS_{pos+1}"] = {
                    "gcc_only": sorted(list(only_gcc)),
                    "non_gcc_only": sorted(list(only_ngcc))
                }
                
    rules = {
        "manual_overrides": {
            "WDDLJ6FB5HA187267": "NON-GCC"
        },
        "gcc_exclusive_wmis": gcc_wmis,
        "non_gcc_exclusive_wmis": ngcc_wmis,
        "make_specific_position_rules": make_rules
    }
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "gcc_non_gcc_rules.json")
    with open(filepath, 'w') as f:
        json.dump(rules, f, indent=2)
    
    logger.info(f"Successfully saved GCC vs Non-GCC rules to {filepath}")
    return rules

def main():
    # Resolve default model directory based on execution context
    default_model_dir = "models"
    if os.path.basename(os.getcwd()) != "chat_cat(2.0)" and os.path.exists("chat_cat(2.0)"):
        default_model_dir = "chat_cat(2.0)/models"

    parser = argparse.ArgumentParser(description="UAE VIN Intelligence System - Unified Pinpoint Training Pipeline")
    parser.add_argument("--data_path", type=str, default="../Data/data_methaq(2.0).csv",
                        help="Path to the raw CSV or Parquet dataset")
    parser.add_argument("--model_dir", type=str, default=default_model_dir,
                        help="Output directory for saved models and encoders")
    parser.add_argument("--downsample", type=int, default=200000,
                        help="Downsample training set for the NN (negative for no downsampling)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Maximum training epochs for the NN")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Training batch size for the NN")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for the NN")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for the NN optimizer")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience for the NN")
    parser.add_argument("--resume", action="store_true",
                        help="Resume Transformer training from existing checkpoint")
    parser.add_argument("--skip_nn", action="store_true",
                        help="Skip training the neural network (Transformer)")
    parser.add_argument("--skip_fallback", action="store_true",
                        help="Skip training the fallback lookup model")
    parser.add_argument("--skip_rules", action="store_true",
                        help="Skip extracting GCC vs. Non-GCC pattern matching rules")
    
    args = parser.parse_args()
    
    is_db = args.data_path.lower() in ("postgres", "postgresql", "db", "database")
    if not is_db and not os.path.exists(args.data_path):
        logger.error(f"Dataset not found at: {args.data_path}")
        return
        
    # Phase 1: Rule extraction
    if not args.skip_rules:
        logger.info("\n--- Phase 1: Pattern Matching Rule Extraction ---")
        extract_gcc_rules(args.data_path, args.model_dir)
        
    # Phase 2: Fallback lookup model training
    if not args.skip_fallback:
        logger.info("\n--- Phase 2: Fallback Lookup Model Training ---")
        X, y, encoders = prepare_data(args.data_path, args.model_dir)
        targets = ["MAKE", "MODEL", "TRIM", "BODY_TYPE", "YEAR", "CYLINDERS", "ORIGIN", "NO_OF_PASSENGERS", "WEIGHT", "REGIONAL_SPEC", "COLOR"]
        fallback_model = {}
        
        for target in targets:
            logger.info(f"Building fallback model for: {target}")
            y_target = y[f"{target}_enc"]
            le = encoders[target]
            feature_levels = BACKOFF_HIERARCHY[target]
            
            # Use stratified split with 20% test size
            idx_train, idx_val = split_and_align_classes(X, y_target, 0.20)
            
            X_train = X.loc[idx_train]
            y_train = y_target.loc[idx_train].values
            
            target_model = {}
            
            # Global default calculation
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            if len(class_counts) > 0:
                best_idx = np.argmax(class_counts)
                global_default_enc = unique_classes[best_idx]
                global_prob = float(class_counts[best_idx] / len(y_train))
                global_default = le.inverse_transform([global_default_enc])[0]
            else:
                global_default = "UNKNOWN"
                global_prob = 1.0
                
            target_model["_default"] = (global_default, global_prob)
            
            train_df = X_train.copy()
            train_df["target_enc"] = y_train
            
            for level_idx, features in enumerate(feature_levels):
                groupby_cols = features.copy()
                counts_df = train_df.groupby(groupby_cols + ["target_enc"]).size().reset_index(name="count")
                best_df = counts_df.sort_values(by=groupby_cols + ["count"], ascending=False).drop_duplicates(subset=groupby_cols)
                total_counts = train_df.groupby(groupby_cols).size().reset_index(name="total_count")
                best_df = pd.merge(best_df, total_counts, on=groupby_cols)
                best_df["prob"] = best_df["count"] / best_df["total_count"]
                best_df["class_str"] = le.inverse_transform(best_df["target_enc"].astype(int))
                
                lookup_dict = {}
                for _, row in best_df.iterrows():
                    key = tuple(row[f] for f in groupby_cols)
                    if len(key) == 1:
                        key = key[0]
                    lookup_dict[key] = (row["class_str"], float(row["prob"]))
                target_model[level_idx] = lookup_dict
                
            fallback_model[target] = target_model
            
            # Quick evaluation
            X_val = X.loc[idx_val]
            y_val_enc = y_target.loc[idx_val].values
            y_val_true = le.inverse_transform(y_val_enc)
            all_features_used = list(set(f for lvl in feature_levels for f in lvl))
            val_records = X_val[all_features_used].to_dict(orient="records")
            
            val_preds = []
            for val_row in val_records:
                pred_val, _ = predict_from_target_model(val_row, target_model, feature_levels)
                val_preds.append(pred_val)
                
            evaluate_predictions(y_val_true, val_preds, target)
            gc.collect()
            
        save_fallback_model(fallback_model, args.model_dir)
        logger.info("Fallback lookup training complete!")

    # Phase 3: Neural network training
    if not args.skip_nn:
        logger.info("\n--- Phase 3: Neural Network Transformer Training ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Reload features/labels and encoders
        X, y, encoders = prepare_data(args.data_path, args.model_dir)
        
        pos_cols = [f"POS_{i}" for i in range(1, 18)]
        vins = X[pos_cols].agg("".join, axis=1).values
        
        target_keys = ["trim", "regional_spec", "body_type"]
        y_dict = {
            "trim": y["TRIM_enc"].values,
            "regional_spec": y["REGIONAL_SPEC_enc"].values,
            "body_type": y["BODY_TYPE_enc"].values,
        }
        
        if args.downsample > 0 and len(vins) > args.downsample:
            logger.info(f"Downsampling neural network training dataset to {args.downsample:,}...")
            np.random.seed(42)
            sampled_indices = np.random.choice(len(vins), size=args.downsample, replace=False)
            vins = vins[sampled_indices]
            y_dict = {k: v[sampled_indices] for k, v in y_dict.items()}
            
        indices = np.arange(len(vins))
        train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42)
        logger.info(f"Train split size: {len(train_idx):,}, Val split size: {len(val_idx):,}")
        
        criterions = {}
        for target in target_keys:
            y_target_train = y_dict[target][train_idx]
            class_counts = np.bincount(y_target_train)
            class_counts = np.maximum(class_counts, 1)
            total_samples = len(y_target_train)
            num_classes = len(encoders[target.upper()].classes_)
            
            if len(class_counts) < num_classes:
                padded_counts = np.ones(num_classes)
                padded_counts[:len(class_counts)] = class_counts
                class_counts = padded_counts
                
            weights = total_samples / (num_classes * class_counts)
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            criterions[target] = nn.CrossEntropyLoss(weight=weights_tensor)
            
        train_dataset = MultiTaskVINDataset(vins[train_idx], {k: v[train_idx] for k, v in y_dict.items()})
        val_dataset = MultiTaskVINDataset(vins[val_idx], {k: v[val_idx] for k, v in y_dict.items()})
        
        pin_mem = (device.type == "cuda")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_mem, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_mem, num_workers=0)
        
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
        
        if args.resume:
            try:
                trainer.load_checkpoint()
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}. Starting from scratch.")
                
        trainer.fit(args.epochs)
        logger.info("Neural Network Transformer training complete!")
        
        # Save validation metrics to metrics.json
        final_metrics = trainer.evaluate()
        metrics_file = os.path.join(args.model_dir, "metrics.json")
        simple_metrics = {}
        for k, v in final_metrics.items():
            simple_metrics[k] = float(v)
        with open(metrics_file, "w") as f:
            json.dump(simple_metrics, f, indent=2)
        logger.info(f"Saved NN metrics to {metrics_file}")

if __name__ == "__main__":
    main()
