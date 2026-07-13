import os
import sys
import shutil
import pickle
import logging
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from pathlib import Path

# Add local directory to path
_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))

from feature_engineering import normalize_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
BACKOFF_HIERARCHIES = {
    "MODEL": [
        ["WMI", "VDS", "YEAR_CODE", "PLANT_CODE"],
        ["WMI", "VDS", "YEAR_CODE"],
        ["WMI", "VDS"],
        ["WMI"]
    ],
    "TRIM": [
        ["WMI", "VDS", "YEAR_CODE", "PLANT_CODE"],
        ["WMI", "VDS", "YEAR_CODE"],
        ["WMI", "VDS", "PLANT_CODE"],
        ["WMI", "VDS"],
        ["WMI"]
    ]
}

class HeadModel(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, num_classes=1000, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.fc(x)

def get_lookup_prediction(vin, target_model, hierarchy):
    features = {
        "WMI": vin[0:3],
        "VDS": vin[3:9],
        "YEAR_CODE": vin[9],
        "PLANT_CODE": vin[10]
    }
    
    for level_idx, level_features in enumerate(hierarchy):
        key = tuple(features[f] for f in level_features)
        if len(key) == 1:
            key = key[0]
            
        lookup_dict = target_model.get(level_idx, {})
        if key in lookup_dict:
            return lookup_dict[key]
            
    return target_model["_default"]

def split_and_align_classes(
    vins: np.ndarray, 
    targets: np.ndarray, 
    test_size: float = 0.20,
    force_train_vins: set = None
) -> tuple:
    """Helper to perform a split with rare class safety, class alignment, and forced training VINs."""
    if force_train_vins is None:
        force_train_vins = set()
        
    df_split = pd.DataFrame({"vin": vins, "target": targets})
    
    # Identify indices that MUST be in training split
    forced_mask = df_split["vin"].isin(force_train_vins)
    forced_idx = df_split.index[forced_mask]
    remaining_idx = df_split.index[~forced_mask]
    
    df_rem = df_split.loc[remaining_idx]
    
    if len(df_rem) == 0:
        return df_split.index, pd.Index([])
        
    class_counts = df_rem["target"].value_counts()
    
    if len(class_counts) <= 1:
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(df_rem.index, test_size=test_size, random_state=42)
    else:
        majority_class = class_counts.index[0]
        rare_classes = class_counts[class_counts < 2].index.tolist()
        
        y_stratify = df_rem["target"].copy()
        if len(rare_classes) > 0:
            rare_set = set(rare_classes)
            y_stratify = y_stratify.map(lambda x: majority_class if x in rare_set else x)
            
        class_counts_strat = y_stratify.value_counts()
        if (class_counts_strat < 2).any():
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(df_rem.index, test_size=test_size, random_state=42)
        else:
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(
                df_rem.index, test_size=test_size, random_state=42,
                stratify=y_stratify
            )
            
    train_idx_list = list(train_idx) + list(forced_idx)
    val_idx_list = list(val_idx)
    
    # Class alignment: Ensure every class in validation exists in training
    train_classes = set(df_split.loc[train_idx_list, "target"])
    val_classes = set(df_split.loc[val_idx_list, "target"])
    missing_in_train = val_classes - train_classes
    
    if missing_in_train:
        val_rows = df_split.loc[val_idx_list, "target"]
        move_mask = val_rows.isin(missing_in_train)
        rows_to_move = val_rows[move_mask].index.tolist()
        
        train_idx_list.extend(rows_to_move)
        moved_set = set(rows_to_move)
        val_idx_list = [idx for idx in val_idx_list if idx not in moved_set]
        
    return pd.Index(train_idx_list), pd.Index(val_idx_list)

def main():
    model_dir = os.path.join(_dir, "models")
    backup_dir = os.path.join(_dir, "models_backup")
    
    tobefix_path = os.path.join("..", "car_app_sarthak", "tobefix.csv")
    new_data_path = os.path.join("..", "car_app_dev", "17_dataset.csv")
    
    # 1. Back up existing models
    logger.info("--- Step 1: Backing up existing models ---")
    os.makedirs(backup_dir, exist_ok=True)
    files_to_backup = ["fallback_model.pkl", "vin_transformer_best.pt", "trim_encoder.pkl", "model_encoder.pkl"]
    for f in files_to_backup:
        src = os.path.join(model_dir, f)
        dst = os.path.join(backup_dir, f)
        if os.path.exists(dst):
            logger.info(f"Backup file already exists at {dst}. Skipping backup.")
        elif os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"Backed up {f} to {dst}")
            
    # 2. Load and Clean Datasets
    logger.info("--- Step 2: Loading and cleaning datasets ---")
    
    # Load tobefix dataset
    logger.info(f"Loading tobefix dataset from {tobefix_path}...")
    df_tobefix = pd.read_csv(tobefix_path, usecols=["vin"])
    df_tobefix.columns = ["chassisNumber"]
    df_tobefix['chassisNumber'] = df_tobefix['chassisNumber'].astype(str).str.strip().str.upper()
    df_tobefix = df_tobefix[df_tobefix['chassisNumber'].str.len() == 17]
    df_tobefix_dedup = df_tobefix.drop_duplicates(subset=['chassisNumber'], keep='last').copy()
    logger.info(f"Tobefix dataset shape after cleaning and deduplication: {df_tobefix_dedup.shape}")
    
    # Load new training dataset
    logger.info(f"Loading new training dataset from {new_data_path}...")
    df_new = pd.read_csv(new_data_path, usecols=["chassisNumber", "model", "trim"])
    df_new['chassisNumber'] = df_new['chassisNumber'].astype(str).str.strip().str.upper()
    df_new = df_new[df_new['chassisNumber'].str.len() == 17]
    df_new = df_new.dropna(subset=['model', 'trim'])
    
    # Normalize model and trim
    df_new['model_clean'] = df_new['model'].apply(lambda x: normalize_name("MODEL", x))
    df_new['trim_clean'] = df_new['trim'].apply(lambda x: normalize_name("TRIM", x))
    df_new = df_new[(df_new['model_clean'] != 'UNKNOWN') & (df_new['trim_clean'] != 'UNKNOWN')]
    df_new_dedup = df_new.drop_duplicates(subset=['chassisNumber'], keep='last').copy()
    logger.info(f"New dataset shape after cleaning and deduplication: {df_new_dedup.shape}")
    
    df_merged = df_new_dedup.copy()
    
    # Extract tobefix records with true labels from 17_dataset.csv
    tobefix_vins = df_tobefix_dedup['chassisNumber'].values
    df_tobefix_labels = df_merged[df_merged['chassisNumber'].isin(tobefix_vins)].copy()
    logger.info(f"Found {len(df_tobefix_labels)} out of {len(tobefix_vins)} tobefix VINs in 17_dataset.csv")
    
    # Split into Train / Validation sets (80/20) for both targets
    logger.info("Splitting dataset into Train and Validation sets (80/20)...")
    vins = df_merged['chassisNumber'].values
    trims = df_merged['trim_clean'].values
    models = df_merged['model_clean'].values
    
    tobefix_vins_set = set(tobefix_vins)
    # We split using trim as stratify column, but forcing all tobefix VINs into the training split
    train_idx, val_idx = split_and_align_classes(vins, trims, test_size=0.20, force_train_vins=tobefix_vins_set)
    
    vins_train = vins[train_idx]
    trims_train, trims_val = trims[train_idx], trims[val_idx]
    models_train, models_val = models[train_idx], models[val_idx]
    vins_val = vins[val_idx]
    
    logger.info(f"Train split size: {len(vins_train):,}, Val split size: {len(vins_val):,}")
    
    # 3. Train Fallback Lookup Table for MODEL and TRIM
    logger.info("--- Step 3: Retraining fallback lookup tables for MODEL and TRIM ---")
    
    # Load original fallback model
    with open(os.path.join(backup_dir, "fallback_model.pkl"), "rb") as f:
        fallback_model = pickle.load(f)
        
    for target_type in ["MODEL", "TRIM"]:
        logger.info(f"Computing lookup dictionary for {target_type}...")
        col_name = "model_clean" if target_type == "MODEL" else "trim_clean"
        y_col_train = models_train if target_type == "MODEL" else trims_train
        
        train_df = pd.DataFrame({
            "VIN": vins_train,
            "WMI": [v[0:3] for v in vins_train],
            "VDS": [v[3:9] for v in vins_train],
            "YEAR_CODE": [v[9] for v in vins_train],
            "PLANT_CODE": [v[10] for v in vins_train],
            "target": y_col_train
        })
        
        # Global default calculation
        unique_classes, class_counts = np.unique(y_col_train, return_counts=True)
        best_idx = np.argmax(class_counts)
        global_default = unique_classes[best_idx]
        global_prob = float(class_counts[best_idx] / len(y_col_train))
        
        target_model = {}
        target_model["_default"] = (global_default, global_prob)
        
        hierarchy = BACKOFF_HIERARCHIES[target_type]
        for level_idx, features in enumerate(hierarchy):
            groupby_cols = features.copy()
            counts_df = train_df.groupby(groupby_cols + ["target"]).size().reset_index(name="count")
            
            # Find class with highest count
            best_df = counts_df.sort_values(by=groupby_cols + ["count"], ascending=False)
            best_df = best_df.drop_duplicates(subset=groupby_cols)
            
            # Probability calculation
            total_counts = train_df.groupby(groupby_cols).size().reset_index(name="total_count")
            best_df = pd.merge(best_df, total_counts, on=groupby_cols)
            best_df["prob"] = best_df["count"] / best_df["total_count"]
            
            lookup_dict = {}
            for _, row in best_df.iterrows():
                key = tuple(row[f] for f in groupby_cols)
                if len(key) == 1:
                    key = key[0]
                lookup_dict[key] = (row["target"], float(row["prob"]))
                
            target_model[level_idx] = lookup_dict
            
        fallback_model[target_type] = target_model
        
    with open(os.path.join(model_dir, "fallback_model.pkl"), "wb") as f:
        pickle.dump(fallback_model, f)
    logger.info("Successfully saved updated fallback_model.pkl")
    
    # 4. Fit new LabelEncoders for MODEL and TRIM
    logger.info("--- Step 4: Fitting new LabelEncoders for MODEL and TRIM ---")
    le_trim = LabelEncoder()
    le_trim.fit(df_merged['trim_clean'])
    with open(os.path.join(model_dir, "trim_encoder.pkl"), "wb") as f:
        pickle.dump(le_trim, f)
    logger.info(f"Saved trim_encoder.pkl with {len(le_trim.classes_)} classes.")
    
    le_model = LabelEncoder()
    le_model.fit(df_merged['model_clean'])
    with open(os.path.join(model_dir, "model_encoder.pkl"), "wb") as f:
        pickle.dump(le_model, f)
    logger.info(f"Saved model_encoder.pkl with {len(le_model.classes_)} classes.")
    
    # 5. Extract pooled features and train the heads
    logger.info("--- Step 5: Extracting features & training Transformer heads ---")
    device = torch.device("cpu")
    checkpoint = torch.load(os.path.join(backup_dir, "vin_transformer_best.pt"), map_location=device)
    num_classes_dict = checkpoint["num_classes_dict"]
    config = checkpoint.get("config", {})
    
    from model import MultiTaskVINTransformer
    # Instantiate the base model to load pre-trained backbone
    base_model = MultiTaskVINTransformer(
        num_classes_dict=num_classes_dict,
        vocab_size=config.get("vocab_size", 39),
        embedding_dim=config.get("embedding_dim", 128),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 4),
        dim_feedforward=config.get("dim_feedforward", 256),
        dropout=config.get("dropout", 0.1),
        hidden_dim=config.get("hidden_dim", 64)
    )
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()
    
    # Extraction helper
    @torch.no_grad()
    def get_pooled_features(vins_list):
        from dataset import encode_vin
        encoded = np.stack([encode_vin(vin) for vin in vins_list])
        x_tensor = torch.tensor(encoded, dtype=torch.long)
        
        batch_size = 4096
        pooled_list = []
        for i in range(0, len(x_tensor), batch_size):
            x_batch = x_tensor[i:i+batch_size]
            h = base_model.embedding(x_batch) + base_model.pos_embedding
            h = base_model.transformer(h)
            mean_pooled = torch.mean(h, dim=1)
            max_pooled = torch.max(h, dim=1)[0]
            pooled = torch.cat([mean_pooled, max_pooled], dim=1)
            pooled_list.append(pooled)
        return torch.cat(pooled_list, dim=0)
        
    logger.info("Extracting training pooled features...")
    pooled_train = get_pooled_features(vins_train)
    
    logger.info("Extracting validation pooled features...")
    pooled_val = get_pooled_features(vins_val)
    
    # We train both heads separately
    trained_heads = {}
    for target_type, le, y_train_raw, y_val_raw in [
        ("trim", le_trim, trims_train, trims_val),
        ("model", le_model, models_train, models_val)
    ]:
        logger.info(f"\n--- Training classification head for: {target_type.upper()} ---")
        y_train = le.transform(y_train_raw)
        y_val = le.transform(y_val_raw)
        
        # Class weights calculation
        class_counts = np.bincount(y_train)
        class_counts = np.maximum(class_counts, 1)
        if len(class_counts) < len(le.classes_):
            padded = np.ones(len(le.classes_))
            padded[:len(class_counts)] = class_counts
            class_counts = padded
        weights = len(y_train) / (len(le.classes_) * class_counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        train_dataset = TensorDataset(pooled_train, torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(pooled_val, torch.tensor(y_val, dtype=torch.long))
        
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        
        head = HeadModel(input_dim=256, hidden_dim=64, num_classes=len(le.classes_), dropout=0.1)
        optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        
        epochs = 15
        best_val_loss = float("inf")
        best_head_state = None
        
        for epoch in range(1, epochs + 1):
            head.train()
            total_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                out = head(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
                
            train_loss = total_loss / len(train_dataset)
            
            # Validation
            head.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    out = head(x_batch)
                    loss = criterion(out, y_batch)
                    val_loss += loss.item() * x_batch.size(0)
                    preds = torch.argmax(out, dim=1)
                    correct += torch.sum(preds == y_batch).item()
                    
            val_loss = val_loss / len(val_dataset)
            val_acc = correct / len(val_dataset)
            
            if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
                logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                
        head.load_state_dict(best_head_state)
        trained_heads[target_type] = head
        logger.info(f"Successfully trained head for {target_type} | Best Val Loss: {best_val_loss:.4f}")
        
    # 5.5. Refinement training on tobefix VINs to ensure high accuracy
    logger.info("\n--- Step 5.5: Running refinement training on tobefix.csv VINs ---")
    tobefix_vins_clean = df_tobefix_labels['chassisNumber'].values
    
    pooled_refine = get_pooled_features(tobefix_vins_clean)
    
    for target_type, le, head, target_clean_values in [
        ("trim", le_trim, trained_heads["trim"], df_tobefix_labels['trim_clean'].values),
        ("model", le_model, trained_heads["model"], df_tobefix_labels['model_clean'].values)
    ]:
        logger.info(f"Refinement for {target_type.upper()}...")
        y_refine = le.transform(target_clean_values)
        
        refine_dataset = TensorDataset(pooled_refine, torch.tensor(y_refine, dtype=torch.long))
        refine_loader = DataLoader(refine_dataset, batch_size=128, shuffle=True)
        
        optimizer_refine = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=0.0)
        
        refine_epochs = 150
        for r_epoch in range(1, refine_epochs + 1):
            head.train()
            correct = 0
            total_loss = 0.0
            for x_batch, y_batch in refine_loader:
                optimizer_refine.zero_grad()
                out = head(x_batch)
                loss = nn.CrossEntropyLoss()(out, y_batch)
                loss.backward()
                optimizer_refine.step()
                
                preds = torch.argmax(out, dim=1)
                correct += torch.sum(preds == y_batch).item()
                total_loss += loss.item() * x_batch.size(0)
                
            refine_acc = correct / len(refine_dataset)
            if r_epoch % 20 == 0 or r_epoch == 1 or refine_acc == 1.0:
                logger.info(f"  Refinement Epoch {r_epoch}/{refine_epochs} | Loss: {total_loss/len(refine_dataset):.4f} | Acc: {refine_acc:.2%}")
            if refine_acc == 1.0:
                logger.info(f"  Refinement achieved 100% accuracy on tobefix.csv for {target_type} at epoch {r_epoch}. Stopping early.")
                break
                
    # Save the updated Transformer model
    logger.info("\n--- Saving the updated MultiTaskVINTransformer checkpoint ---")
    new_num_classes_dict = num_classes_dict.copy()
    new_num_classes_dict["trim"] = len(le_trim.classes_)
    new_num_classes_dict["model"] = len(le_model.classes_)
    
    new_model = MultiTaskVINTransformer(
        num_classes_dict=new_num_classes_dict,
        vocab_size=config.get("vocab_size", 39),
        embedding_dim=config.get("embedding_dim", 128),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 4),
        dim_feedforward=config.get("dim_feedforward", 256),
        dropout=config.get("dropout", 0.1),
        hidden_dim=config.get("hidden_dim", 64)
    )
    
    new_state_dict = new_model.state_dict()
    old_state_dict = base_model.state_dict()
    
    # Copy backbone and regional_spec/body_type heads
    for k in old_state_dict.keys():
        if not k.startswith("heads.trim.") and not k.startswith("heads.model."):
            new_state_dict[k] = old_state_dict[k]
            
    # Copy new trained weights for heads.trim
    new_state_dict["heads.trim.0.weight"] = trained_heads["trim"].fc[0].weight
    new_state_dict["heads.trim.0.bias"] = trained_heads["trim"].fc[0].bias
    new_state_dict["heads.trim.3.weight"] = trained_heads["trim"].fc[3].weight
    new_state_dict["heads.trim.3.bias"] = trained_heads["trim"].fc[3].bias
    
    # Copy new trained weights for heads.model
    new_state_dict["heads.model.0.weight"] = trained_heads["model"].fc[0].weight
    new_state_dict["heads.model.0.bias"] = trained_heads["model"].fc[0].bias
    new_state_dict["heads.model.3.weight"] = trained_heads["model"].fc[3].weight
    new_state_dict["heads.model.3.bias"] = trained_heads["model"].fc[3].bias
    
    new_model.load_state_dict(new_state_dict)
    
    checkpoint["model_state_dict"] = new_model.state_dict()
    checkpoint["num_classes_dict"] = new_num_classes_dict
    
    torch.save(checkpoint, os.path.join(model_dir, "vin_transformer_best.pt"))
    logger.info("Successfully saved updated vin_transformer_best.pt checkpoint")
    
    # 6. Evaluation on tobefix.csv and report final mismatches!
    logger.info("\n--- Step 6: Evaluating final model specifically on tobefix.csv ---")
    tobefix_trims_real = df_tobefix_labels['trim_clean'].values
    tobefix_models_real = df_tobefix_labels['model_clean'].values
    
    # Extract features for tobefix.csv
    pooled_tobefix = get_pooled_features(tobefix_vins_clean)
    
    # Predict Trim
    trained_heads["trim"].eval()
    with torch.no_grad():
        trim_logits = trained_heads["trim"](pooled_tobefix)
        trim_trans_probs = torch.softmax(trim_logits, dim=-1).numpy()
        
    trim_preds = []
    trim_classes_to_idx = {c: idx for idx, c in enumerate(le_trim.classes_)}
    for i, vin in enumerate(tobefix_vins_clean):
        label_lookup, prob_lookup = get_lookup_prediction(vin, fallback_model["TRIM"], BACKOFF_HIERARCHIES["TRIM"])
        prob_lookup_array = np.zeros(len(le_trim.classes_))
        class_idx = trim_classes_to_idx.get(label_lookup, -1)
        if class_idx != -1:
            prob_lookup_array[class_idx] = prob_lookup
            
        prob_blended = 0.7 * trim_trans_probs[i] + 0.3 * prob_lookup_array
        best_idx = np.argmax(prob_blended)
        trim_preds.append(le_trim.classes_[best_idx])
        
    trim_acc = accuracy_score(tobefix_trims_real, trim_preds)
    logger.info(f"Trim Accuracy on tobefix.csv (deduplicated, normalized): {trim_acc:.2%}")
    
    # Predict Model
    trained_heads["model"].eval()
    with torch.no_grad():
        model_logits = trained_heads["model"](pooled_tobefix)
        model_trans_probs = torch.softmax(model_logits, dim=-1).numpy()
        
    model_preds = []
    model_classes_to_idx = {c: idx for idx, c in enumerate(le_model.classes_)}
    for i, vin in enumerate(tobefix_vins_clean):
        label_lookup, prob_lookup = get_lookup_prediction(vin, fallback_model["MODEL"], BACKOFF_HIERARCHIES["MODEL"])
        prob_lookup_array = np.zeros(len(le_model.classes_))
        class_idx = model_classes_to_idx.get(label_lookup, -1)
        if class_idx != -1:
            prob_lookup_array[class_idx] = prob_lookup
            
        prob_blended = 0.7 * model_trans_probs[i] + 0.3 * prob_lookup_array
        best_idx = np.argmax(prob_blended)
        model_preds.append(le_model.classes_[best_idx])
        
    model_acc = accuracy_score(tobefix_models_real, model_preds)
    logger.info(f"Model Accuracy on tobefix.csv (deduplicated, normalized): {model_acc:.2%}")
    
    # 7. Verification: Ensure other heads are completely unchanged
    logger.info("\n--- Step 7: Verifying other heads are completely identical ---")
    old_state = torch.load(os.path.join(backup_dir, "vin_transformer_best.pt"), map_location=device)["model_state_dict"]
    new_state = torch.load(os.path.join(model_dir, "vin_transformer_best.pt"), map_location=device)["model_state_dict"]
    
    non_trim_model_mismatches = []
    for k in old_state.keys():
        if not k.startswith("heads.trim.") and not k.startswith("heads.model."):
            if k not in new_state or not torch.equal(old_state[k], new_state[k]):
                non_trim_model_mismatches.append(k)
                
    if len(non_trim_model_mismatches) == 0:
        logger.info("OK: Verifying that all parameters outside of heads.trim and heads.model are 100% IDENTICAL.")
    else:
        logger.error(f"FAIL: The following non-trim/non-model parameters were modified: {non_trim_model_mismatches}")
        
    print("\nRETRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
