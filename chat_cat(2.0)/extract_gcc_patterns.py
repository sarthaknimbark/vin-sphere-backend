import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    data_path = "../Data/data_methaq(2.0).csv"
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        return

    logger.info("Loading dataset for pattern analysis...")
    df = pd.read_csv(data_path)
    
    # 1. Clean VINs and regional spec
    df['VIN'] = df['chassisNumber'].astype(str).str.upper().str.strip()
    df = df[df['VIN'].str.len() == 17].copy()
    
    df['regionalSpec'] = df['regionalSpec'].astype(str).str.upper().str.strip()
    df = df[df['regionalSpec'].isin(['GCC', 'NON-GCC'])].copy()
    
    logger.info(f"Loaded {len(df):,} valid rows after filtering regionalSpec.")
    
    # 2. Extract character features
    logger.info("Extracting character features...")
    X_chars = pd.DataFrame(index=df.index)
    for pos in range(1, 18):
        X_chars[f"POS_{pos}"] = df['VIN'].str[pos-1]
        
    # 3. Label encode character positions for Decision Tree
    logger.info("Label encoding character features...")
    X_encoded = pd.DataFrame(index=df.index)
    char_encoders = {}
    for col in X_chars.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_chars[col])
        char_encoders[col] = le
        
    y = (df['regionalSpec'] == 'GCC').astype(int).values
    class_names = ['NON-GCC', 'GCC']
    
    # 4. Train an interpretable Decision Tree to extract rules
    logger.info("Training decision tree for rule extraction...")
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
    dt.fit(X_encoded, y)
    
    # 5. Extract rules in text format
    feature_names = list(X_chars.columns)
    text_rules = export_text(dt, feature_names=feature_names)
    
    # Decode the text rules to make them human readable
    decoded_lines = []
    for line in text_rules.split('\n'):
        if not line.strip():
            continue
        
        # Check if it is a decision node or a leaf node
        if 'class:' in line:
            class_val = int(float(line.split('class:')[1].strip()))
            decoded_lines.append(line.replace(f"class: {class_val}.0", f"DECISION -> {class_names[class_val]}"))
        elif '<=' in line or '>' in line:
            # Parse the line (e.g. "POS_1 <= 12.50")
            parts = line.split()
            feature = next(p for p in parts if p.startswith('POS_'))
            operator = next(p for p in parts if p in ['<=', '>'])
            value_float = float(next(p for p in parts if p.replace('.', '', 1).isdigit() or p.split('.')[0].isdigit()))
            
            # Map float value back to character categories
            encoder = char_encoders[feature]
            classes = list(encoder.classes_)
            
            if operator == '<=':
                matching_chars = [c for idx, c in enumerate(classes) if idx <= value_float]
            else:
                matching_chars = [c for idx, c in enumerate(classes) if idx > value_float]
                
            char_list_str = "".join(matching_chars)
            # Simplify display of character ranges
            if len(char_list_str) > 10:
                char_list_str = char_list_str[:5] + "..." + char_list_str[-5:]
                
            decoded_lines.append(line.replace(f"{feature} {operator} {value_float:.2f}", f"{feature} IN [{char_list_str}]"))
        else:
            decoded_lines.append(line)
            
    decoded_text_rules = "\n".join(decoded_lines)
    
    # 6. Analyze exact prefix patterns (WMI + VDS)
    logger.info("Analyzing WMI+VDS patterns...")
    df['WMI_VDS'] = df['VIN'].str[0:9]
    prefix_counts = df.groupby(['WMI_VDS', 'regionalSpec']).size().unstack(fill_value=0)
    
    # Find prefixes that are 100% GCC or 100% Non-GCC
    gcc_exclusive = prefix_counts[(prefix_counts['GCC'] > 10) & (prefix_counts['NON-GCC'] == 0)].index.tolist()
    ngcc_exclusive = prefix_counts[(prefix_counts['NON-GCC'] > 10) & (prefix_counts['GCC'] == 0)].index.tolist()
    
    # 7. Print and Save findings
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "gcc_non_gcc_rules.txt")
    
    with open(report_path, "w") as f:
        f.write("=== GCC vs NON-GCC REGIONAL SPECIFICATION RULES ===\n\n")
        f.write("1. DECISION TREE CLASSIFICATION RULES:\n")
        f.write(decoded_text_rules)
        f.write("\n\n2. FEATURE IMPORTANCE RANKING:\n")
        importances = dt.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f_idx in indices:
            if importances[f_idx] > 0:
                f.write(f"   {feature_names[f_idx]}: {importances[f_idx]:.2%}\n")
                
        f.write(f"\n\n3. EXCLUSIVE WMI+VDS PREFIX PATTERNS:\n")
        f.write(f"   GCC-Exclusive prefixes (first 9 characters, count={len(gcc_exclusive)}): {gcc_exclusive[:30]}...\n")
        f.write(f"   Non-GCC-Exclusive prefixes (first 9 characters, count={len(ngcc_exclusive)}): {ngcc_exclusive[:30]}...\n")
        
    print("\n" + "="*60)
    print(" GCC vs NON-GCC PATTERN MATCHING RULES EXTRACTED")
    print("="*60)
    print(decoded_text_rules)
    print("\nFeature Importances:")
    for f_idx in indices:
        if importances[f_idx] > 0:
            print(f"  {feature_names[f_idx]}: {importances[f_idx]:.2%}")
            
    print(f"\nSaved full rule report to: {report_path}")
    print("="*60)

if __name__ == "__main__":
    main()
