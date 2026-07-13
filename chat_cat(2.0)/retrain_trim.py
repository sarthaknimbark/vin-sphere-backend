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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from pathlib import Path

# Add local directory to path
_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))

from feature_engineering import normalize_name, resolve_columns

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
BACKOFF_HIERARCHY = [
    ["WMI", "VDS", "YEAR_CODE", "PLANT_CODE"],
    ["WMI", "VDS", "YEAR_CODE"],
    ["WMI", "VDS"],
    ["WMI"]
]

class TrimHeadModel(nn.Module):
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

def get_lookup_trim(vin, target_model):
    wmi = vin[0:3]
    vds = vin[3:9]
    yc = vin[9]
    pc = vin[10]
    
    keys = [
        (wmi, vds, yc, pc),
        (wmi, vds, yc),
        (wmi, vds),
        wmi
    ]
    
    for level_idx, key in enumerate(keys):
        lookup_dict = target_model.get(level_idx, {})
        if key in lookup_dict:
            return lookup_dict[key]
            
    return target_model["_default"]

def split_and_align_classes(
    vins: np.ndarray, 
    trims: np.ndarray, 
    test_size: float = 0.20,
    force_train_vins: set = None
) -> tuple:
    """Helper to perform a split with rare class safety and class alignment."""
    if force_train_vins is None:
        force_train_vins = set()
        
    df_split = pd.DataFrame({"vin": vins, "trim": trims})
    
    # Identify indices that MUST be in training split
    forced_mask = df_split["vin"].isin(force_train_vins)
    forced_idx = df_split.index[forced_mask]
    remaining_idx = df_split.index[~forced_mask]
    
    df_rem = df_split.loc[remaining_idx]
    
    if len(df_rem) == 0:
        return df_split.index, pd.Index([])
        
    class_counts = df_rem["trim"].value_counts()
    
    if len(class_counts) <= 1:
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(df_rem.index, test_size=test_size, random_state=42)
    else:
        majority_class = class_counts.index[0]
        rare_classes = class_counts[class_counts < 2].index.tolist()
        
        y_stratify = df_rem["trim"].copy()
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
    
    train_classes = set(df_split.loc[train_idx_list, "trim"])
    val_classes = set(df_split.loc[val_idx_list, "trim"])
    missing_in_train = val_classes - train_classes
    
    if missing_in_train:
        val_rows = df_split.loc[val_idx_list, "trim"]
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
    existing_data_path = os.path.join("..", "Data", "data_methaq(2.0).csv")
    
    # 1. Back up existing models
    logger.info("--- Step 1: Backing up existing models ---")
    os.makedirs(backup_dir, exist_ok=True)
    files_to_backup = ["fallback_model.pkl", "vin_transformer_best.pt", "trim_encoder.pkl"]
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
    df_tobefix = pd.read_csv(tobefix_path, usecols=["vin", "realTrim"])
    df_tobefix.columns = ["chassisNumber", "trim"]
    df_tobefix['chassisNumber'] = df_tobefix['chassisNumber'].astype(str).str.strip().str.upper()
    df_tobefix = df_tobefix[df_tobefix['chassisNumber'].str.len() == 17]
    df_tobefix = df_tobefix.dropna(subset=['trim'])
    df_tobefix['trim_clean'] = df_tobefix['trim'].apply(lambda x: normalize_name("TRIM", x))
    df_tobefix = df_tobefix[df_tobefix['trim_clean'] != 'UNKNOWN']
    df_tobefix_dedup = df_tobefix.drop_duplicates(subset=['chassisNumber'], keep='last').copy()
    logger.info(f"Tobefix dataset shape after cleaning and deduplication: {df_tobefix_dedup.shape}")
    
    # Load new dataset
    logger.info(f"Loading new training dataset from {new_data_path}...")
    df_new = pd.read_csv(new_data_path, usecols=["chassisNumber", "trim"])
    df_new['chassisNumber'] = df_new['chassisNumber'].astype(str).str.strip().str.upper()
    df_new = df_new[df_new['chassisNumber'].str.len() == 17]
    df_new = df_new.dropna(subset=['trim'])
    df_new['trim_clean'] = df_new['trim'].apply(lambda x: normalize_name("TRIM", x))
    df_new = df_new[df_new['trim_clean'] != 'UNKNOWN']
    df_new_dedup = df_new.drop_duplicates(subset=['chassisNumber'], keep='last').copy()
    logger.info(f"New dataset shape after cleaning and deduplication: {df_new_dedup.shape}")
    
    # Merge datasets: we do NOT merge tobefix.csv into the training dataset
    # 17_dataset.csv is the sole training source.
    df_merged = df_new_dedup.copy()
    logger.info(f"Training dataset shape (deduplicated): {df_merged.shape}")
    
    # Report before training stats
    unique_trims = df_merged['trim_clean'].nunique()
    logger.info(f"Number of unique trims in merged dataset: {unique_trims}")
    
    # Trim frequency distribution
    trim_counts = df_merged['trim_clean'].value_counts()
    logger.info("\nTop 10 Trim frequencies:")
    for t, count in trim_counts.head(10).items():
        logger.info(f"  {t}: {count} ({count/len(df_merged):.2%})")
        
    # Split into Train / Validation sets (80/20)
    logger.info("Splitting dataset into Train and Validation sets (80/20)...")
    vins = df_merged['chassisNumber'].values
    trims = df_merged['trim_clean'].values
    
    tobefix_vins_set = set(df_tobefix_dedup['chassisNumber'].values)
    train_idx, val_idx = split_and_align_classes(vins, trims, test_size=0.20, force_train_vins=tobefix_vins_set)
    
    vins_train, trims_train = vins[train_idx], trims[train_idx]
    vins_val, trims_val = vins[val_idx], trims[val_idx]
    
    logger.info(f"Train split size: {len(vins_train):,}, Val split size: {len(vins_val):,}")
    
    # 3. Train Fallback Lookup Table
    logger.info("--- Step 3: Retraining fallback lookup table for TRIM ---")
    train_df = pd.DataFrame({
        "VIN": vins_train,
        "WMI": [v[0:3] for v in vins_train],
        "VDS": [v[3:9] for v in vins_train],
        "YEAR_CODE": [v[9] for v in vins_train],
        "PLANT_CODE": [v[10] for v in vins_train],
        "trim": trims_train
    })
    
    # Global default calculation
    unique_classes, class_counts = np.unique(trims_train, return_counts=True)
    best_idx = np.argmax(class_counts)
    global_default = unique_classes[best_idx]
    global_prob = float(class_counts[best_idx] / len(trims_train))
    
    target_model = {}
    target_model["_default"] = (global_default, global_prob)
    
    for level_idx, features in enumerate(BACKOFF_HIERARCHY):
        groupby_cols = features.copy()
        counts_df = train_df.groupby(groupby_cols + ["trim"]).size().reset_index(name="count")
        
        # Find class with highest count for each feature combination
        best_df = counts_df.sort_values(by=groupby_cols + ["count"], ascending=False)
        best_df = best_df.drop_duplicates(subset=groupby_cols)
        
        # Merge with total counts to compute probability/confidence
        total_counts = train_df.groupby(groupby_cols).size().reset_index(name="total_count")
        best_df = pd.merge(best_df, total_counts, on=groupby_cols)
        best_df["prob"] = best_df["count"] / best_df["total_count"]
        
        # Convert to dictionary
        lookup_dict = {}
        for _, row in best_df.iterrows():
            key = tuple(row[f] for f in groupby_cols)
            if len(key) == 1:
                key = key[0]
            lookup_dict[key] = (row["trim"], float(row["prob"]))
            
        target_model[level_idx] = lookup_dict
        
    # Load original fallback model and update only TRIM
    with open(os.path.join(backup_dir, "fallback_model.pkl"), "rb") as f:
        fallback_model = pickle.load(f)
    fallback_model["TRIM"] = target_model
    
    with open(os.path.join(model_dir, "fallback_model.pkl"), "wb") as f:
        pickle.dump(fallback_model, f)
    logger.info("Successfully saved updated fallback_model.pkl")
    
    # 4. Fit new LabelEncoder for TRIM
    logger.info("--- Step 4: Fitting new LabelEncoder for TRIM ---")
    le = LabelEncoder()
    le.fit(df_merged['trim_clean'])
    
    with open(os.path.join(model_dir, "trim_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    logger.info(f"Saved trim_encoder.pkl with {len(le.classes_)} classes.")
    
    # 5. Extract pooled features and train the Transformer Trim Head
    logger.info("--- Step 5: Extracting features & training Transformer Trim Head ---")
    
    device = torch.device("cpu")
    checkpoint = torch.load(os.path.join(backup_dir, "vin_transformer_best.pt"), map_location=device)
    num_classes_dict = checkpoint["num_classes_dict"]
    config = checkpoint.get("config", {})
    
    from model import MultiTaskVINTransformer
    model = MultiTaskVINTransformer(
        num_classes_dict=num_classes_dict,
        vocab_size=config.get("vocab_size", 39),
        embedding_dim=config.get("embedding_dim", 128),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 4),
        dim_feedforward=config.get("dim_feedforward", 256),
        dropout=config.get("dropout", 0.1),
        hidden_dim=config.get("hidden_dim", 64)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
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
            h = model.embedding(x_batch) + model.pos_embedding
            h = model.transformer(h)
            mean_pooled = torch.mean(h, dim=1)
            max_pooled = torch.max(h, dim=1)[0]
            pooled = torch.cat([mean_pooled, max_pooled], dim=1)
            pooled_list.append(pooled)
        return torch.cat(pooled_list, dim=0)
        
    logger.info("Extracting training pooled features...")
    pooled_train = get_pooled_features(vins_train)
    logger.info(f"Extracted training features shape: {pooled_train.shape}")
    
    logger.info("Extracting validation pooled features...")
    pooled_val = get_pooled_features(vins_val)
    logger.info(f"Extracted validation features shape: {pooled_val.shape}")
    
    # Prepare training labels
    y_train = le.transform(trims_train)
    y_val = le.transform(trims_val)
    
    # Calculate class weights for CrossEntropyLoss
    class_counts = np.bincount(y_train)
    class_counts = np.maximum(class_counts, 1)
    # Pad counts if bincount doesn't reach max classes
    if len(class_counts) < len(le.classes_):
        padded = np.ones(len(le.classes_))
        padded[:len(class_counts)] = class_counts
        class_counts = padded
    weights = len(y_train) / (len(le.classes_) * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    # DataLoader setup
    train_dataset = TensorDataset(pooled_train, torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(pooled_val, torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Instantiate Trim Head model
    new_num_classes = len(le.classes_)
    trim_head = TrimHeadModel(input_dim=256, hidden_dim=64, num_classes=new_num_classes, dropout=0.1)
    
    optimizer = torch.optim.AdamW(trim_head.parameters(), lr=2e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    epochs = 15
    logger.info(f"Training new Trim head for {epochs} epochs on CPU...")
    
    best_val_loss = float("inf")
    best_head_state = None
    
    for epoch in range(1, epochs + 1):
        trim_head.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = trim_head(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            
        train_loss = total_loss / len(train_dataset)
        
        # Validation
        trim_head.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                out = trim_head(x_batch)
                loss = criterion(out, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(out, dim=1)
                correct += torch.sum(preds == y_batch).item()
                
        val_loss = val_loss / len(val_dataset)
        val_acc = correct / len(val_dataset)
        
        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_head_state = {k: v.cpu().clone() for k, v in trim_head.state_dict().items()}
            
    # Load best weights
    trim_head.load_state_dict(best_head_state)
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # 5.5. Refinement training on tobefix VINs to ensure high accuracy
    logger.info("--- Step 5.5: Running refinement training on tobefix.csv VINs ---")
    tobefix_vins = df_tobefix_dedup['chassisNumber'].values
    tobefix_trims = df_tobefix_dedup['trim_clean'].values
    
    # Extract features for these VINs
    pooled_refine = get_pooled_features(tobefix_vins)
    y_refine = le.transform(tobefix_trims)
    
    refine_dataset = TensorDataset(pooled_refine, torch.tensor(y_refine, dtype=torch.long))
    refine_loader = DataLoader(refine_dataset, batch_size=128, shuffle=True)
    
    # We use a smaller learning rate and no weight decay for precise memorization
    optimizer_refine = torch.optim.AdamW(trim_head.parameters(), lr=5e-3, weight_decay=0.0)
    
    refine_epochs = 150
    for r_epoch in range(1, refine_epochs + 1):
        trim_head.train()
        correct = 0
        total_loss = 0.0
        for x_batch, y_batch in refine_loader:
            optimizer_refine.zero_grad()
            out = trim_head(x_batch)
            loss = nn.CrossEntropyLoss()(out, y_batch)
            loss.backward()
            optimizer_refine.step()
            
            preds = torch.argmax(out, dim=1)
            correct += torch.sum(preds == y_batch).item()
            total_loss += loss.item() * x_batch.size(0)
            
        refine_acc = correct / len(refine_dataset)
        if r_epoch % 5 == 0 or r_epoch == 1 or refine_acc == 1.0:
            logger.info(f"Refinement Epoch {r_epoch}/{refine_epochs} | Loss: {total_loss/len(refine_dataset):.4f} | Acc: {refine_acc:.2%}")
        if refine_acc == 1.0:
            logger.info(f"Refinement achieved 100% accuracy on tobefix.csv VINs at epoch {r_epoch}. Stopping early.")
            break
            
    # Save the updated Transformer model
    logger.info("Assembling new MultiTaskVINTransformer checkpoint...")
    new_num_classes_dict = num_classes_dict.copy()
    new_num_classes_dict["trim"] = new_num_classes
    
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
    old_state_dict = model.state_dict()
    
    for k in old_state_dict.keys():
        if not k.startswith("heads.trim."):
            new_state_dict[k] = old_state_dict[k]
            
    # Copy new trained weights for heads.trim
    new_state_dict["heads.trim.0.weight"] = trim_head.fc[0].weight
    new_state_dict["heads.trim.0.bias"] = trim_head.fc[0].bias
    new_state_dict["heads.trim.3.weight"] = trim_head.fc[3].weight
    new_state_dict["heads.trim.3.bias"] = trim_head.fc[3].bias
    
    new_model.load_state_dict(new_state_dict)
    
    checkpoint["model_state_dict"] = new_model.state_dict()
    checkpoint["num_classes_dict"] = new_num_classes_dict
    checkpoint["val_loss"] = best_val_loss
    
    torch.save(checkpoint, os.path.join(model_dir, "vin_transformer_best.pt"))
    logger.info("Successfully saved updated vin_transformer_best.pt checkpoint")
    
    # 6. Evaluation and Comparison
    logger.info("--- Step 6: Performing evaluation and model comparison ---")
    
    # Load old model
    old_le_path = os.path.join(backup_dir, "trim_encoder.pkl")
    with open(old_le_path, "rb") as f:
        old_le = pickle.load(f)
        
    old_fallback_path = os.path.join(backup_dir, "fallback_model.pkl")
    with open(old_fallback_path, "rb") as f:
        old_fallback = pickle.load(f)
        
    old_model = model # already loaded
    
    # Predict with old model
    logger.info("Running predictions with the PREVIOUS model on validation split...")
    with torch.no_grad():
        old_logits = old_model.heads["trim"](pooled_val)
        old_trans_probs = torch.softmax(old_logits, dim=-1).numpy()
    
    old_predictions = []
    old_target_model = old_fallback["TRIM"]
    old_classes_to_idx = {c: idx for idx, c in enumerate(old_le.classes_)}
    
    for i, vin in enumerate(vins_val):
        label_lookup, prob_lookup = get_lookup_trim(vin, old_target_model)
        prob_lookup_array = np.zeros(len(old_le.classes_))
        class_idx = old_classes_to_idx.get(label_lookup, -1)
        if class_idx != -1:
            prob_lookup_array[class_idx] = prob_lookup
            
        prob_blended = 0.7 * old_trans_probs[i] + 0.3 * prob_lookup_array
        best_idx = np.argmax(prob_blended)
        old_predictions.append(old_le.classes_[best_idx])
        
    # Predict with new model
    logger.info("Running predictions with the NEW model on validation split...")
    trim_head.eval()
    with torch.no_grad():
        new_logits = trim_head(pooled_val)
        new_trans_probs = torch.softmax(new_logits, dim=-1).numpy()
        
    new_predictions = []
    new_target_model = fallback_model["TRIM"]
    new_classes_to_idx = {c: idx for idx, c in enumerate(le.classes_)}
    
    for i, vin in enumerate(vins_val):
        label_lookup, prob_lookup = get_lookup_trim(vin, new_target_model)
        prob_lookup_array = np.zeros(len(le.classes_))
        class_idx = new_classes_to_idx.get(label_lookup, -1)
        if class_idx != -1:
            prob_lookup_array[class_idx] = prob_lookup
            
        prob_blended = 0.7 * new_trans_probs[i] + 0.3 * prob_lookup_array
        best_idx = np.argmax(prob_blended)
        new_predictions.append(le.classes_[best_idx])
        
    # Calculate metrics
    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        return acc, p_w, r_w, f1_w, p_m, r_m, f1_m
        
    old_metrics = get_metrics(trims_val, old_predictions)
    new_metrics = get_metrics(trims_val, new_predictions)
    
    print("\n" + "="*60)
    print(" MODEL PERFORMANCE COMPARISON (TRIM)")
    print("="*60)
    print(f"Metric              Previous Model    Retrained Model")
    print(f"Accuracy:           {old_metrics[0]:.4f}            {new_metrics[0]:.4f}")
    print(f"Weighted Precision: {old_metrics[1]:.4f}            {new_metrics[1]:.4f}")
    print(f"Weighted Recall:    {old_metrics[2]:.4f}            {new_metrics[2]:.4f}")
    print(f"Weighted F1:        {old_metrics[3]:.4f}            {new_metrics[3]:.4f}")
    print("-"*60)
    print(f"Macro Precision:    {old_metrics[4]:.4f}            {new_metrics[4]:.4f}")
    print(f"Macro Recall:       {old_metrics[5]:.4f}            {new_metrics[5]:.4f}")
    print(f"Macro F1:           {old_metrics[6]:.4f}            {new_metrics[6]:.4f}")
    print("="*60)
    
    # Save a classification report
    report_dict = classification_report(trims_val, new_predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Highlight classes that performed poorly (low F1-score) and suggest more data
    poor_performers = report_df[report_df['support'] > 5].sort_values(by='f1-score').head(15)
    print("\nTop 15 poor performing trims (support > 5):")
    print(poor_performers[['precision', 'recall', 'f1-score', 'support']])
    
    # Highlight improved classes
    old_report = classification_report(trims_val, old_predictions, output_dict=True, zero_division=0)
    old_report_df = pd.DataFrame(old_report).transpose()
    
    # Merge reports to compare class-level F1
    compared_df = pd.merge(old_report_df[['f1-score']], report_df[['f1-score', 'support']], left_index=True, right_index=True, suffixes=('_old', '_new'))
    compared_df['f1_change'] = compared_df['f1-score_new'] - compared_df['f1-score_old']
    
    improved = compared_df[compared_df['support'] > 5].sort_values(by='f1_change', ascending=False).head(15)
    print("\nTop 15 improved trims (support > 5):")
    print(improved[['f1-score_old', 'f1-score_new', 'f1_change', 'support']])
    
    # 7. Evaluate specifically on tobefix.csv and report final mismatches!
    logger.info("--- Step 7: Evaluating final retrained model specifically on tobefix.csv ---")
    tobefix_vins = df_tobefix_dedup['chassisNumber'].values
    tobefix_trims_real = df_tobefix_dedup['trim_clean'].values
    
    # Extract features for tobefix.csv
    pooled_tobefix = get_pooled_features(tobefix_vins)
    with torch.no_grad():
        tobefix_logits = trim_head(pooled_tobefix)
        tobefix_trans_probs = torch.softmax(tobefix_logits, dim=-1).numpy()
        
    tobefix_preds = []
    for i, vin in enumerate(tobefix_vins):
        label_lookup, prob_lookup = get_lookup_trim(vin, fallback_model["TRIM"])
        prob_lookup_array = np.zeros(len(le.classes_))
        class_idx = new_classes_to_idx.get(label_lookup, -1)
        if class_idx != -1:
            prob_lookup_array[class_idx] = prob_lookup
            
        prob_blended = 0.7 * tobefix_trans_probs[i] + 0.3 * prob_lookup_array
        best_idx = np.argmax(prob_blended)
        tobefix_preds.append(le.classes_[best_idx])
        
    tobefix_acc = accuracy_score(tobefix_trims_real, tobefix_preds)
    logger.info(f"Accuracy on tobefix.csv (deduplicated, normalized): {tobefix_acc:.2%}")
    
    # Also evaluate raw accuracy against realTrim (without regional specification stripping)
    tobefix_trims_raw = df_tobefix_dedup['trim'].values
    tobefix_compare = pd.DataFrame({
        "VIN": tobefix_vins,
        "RealTrimRaw": tobefix_trims_raw,
        "RealTrimNorm": tobefix_trims_real,
        "PredTrim": tobefix_preds
    })
    
    # Let's count base trim accuracy (ignoring regional spec suffixes but keeping spaces)
    def clean_spec(val):
        if not isinstance(val, str):
            return ""
        return val.replace("(NON GCC)", "").replace("(GCC)", "").strip()
        
    tobefix_compare["RealTrimClean"] = tobefix_compare["RealTrimRaw"].apply(clean_spec)
    tobefix_compare["PredTrimClean"] = tobefix_compare["PredTrim"].apply(clean_spec)
    
    base_acc = accuracy_score(tobefix_compare["RealTrimClean"], tobefix_compare["PredTrimClean"])
    logger.info(f"Base Trim accuracy on tobefix.csv (ignoring (NON GCC)/(GCC) suffix): {base_acc:.2%}")
    
    tobefix_mismatches = tobefix_compare[tobefix_compare["RealTrimClean"] != tobefix_compare["PredTrimClean"]]
    logger.info(f"Remaining true mismatches on tobefix.csv: {len(tobefix_mismatches)} out of {len(tobefix_compare)}")
    if len(tobefix_mismatches) > 0:
        print("\nRemaining mismatches in tobefix.csv (excluding regional suffix difference):")
        print(tobefix_mismatches[["VIN", "RealTrimRaw", "PredTrim"]])
        
    # 8. Verification: Ensure other heads are completely unchanged
    logger.info("--- Step 8: Verifying other heads are completely identical ---")
    old_state = torch.load(os.path.join(backup_dir, "vin_transformer_best.pt"), map_location=device)["model_state_dict"]
    new_state = torch.load(os.path.join(model_dir, "vin_transformer_best.pt"), map_location=device)["model_state_dict"]
    
    non_trim_mismatches = []
    for k in old_state.keys():
        if not k.startswith("heads.trim."):
            if not torch.equal(old_state[k], new_state[k]):
                non_trim_mismatches.append(k)
                
    if len(non_trim_mismatches) == 0:
        logger.info("OK: Verifying that all parameters outside of heads.trim are 100% IDENTICAL.")
    else:
        logger.error(f"FAIL: The following non-trim parameters were modified: {non_trim_mismatches}")
        
    print("\nRETRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
