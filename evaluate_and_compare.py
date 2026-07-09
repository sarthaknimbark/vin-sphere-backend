import os
import sys
import json
import time
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root and subpackages to python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(project_root / "chat_cat(2.0)") not in sys.path:
    sys.path.append(str(project_root / "chat_cat(2.0)"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("evaluate_compare")

# Target attributes to evaluate
ATTRIBUTES = [
    "make", "model", "trim", "body_type", "year", 
    "cylinders", "origin", "no_of_passengers", "weight", 
    "regional_spec", "color"
]

def load_config(config_path: Path) -> dict:
    """Loads evaluation weights and decision thresholds."""
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
    
    # Fallback default configurations
    return {
        "weights": {
            "make": 0.25,
            "model": 0.25,
            "trim": 0.20,
            "body_type": 0.10,
            "year": 0.10,
            "origin": 0.05,
            "cylinders": 0.05
        },
        "metrics": {
            "primary": "accuracy",
            "thresholds": {
                "accuracy": 0.0,
                "f1_weighted": 0.0
            }
        },
        "promotion_rules": {
            "require_overall_improvement": true,
            "allow_regression_tolerance": 0.01,
            "critical_attributes": ["make", "model"]
        }
    }

def get_fixed_dataset(dataset_path: Path) -> pd.DataFrame:
    """Loads fixed evaluation dataset. Generates it deterministically from PostgreSQL if missing."""
    if dataset_path.exists():
        logger.info(f"Loaded existing evaluation dataset from: {dataset_path}")
        return pd.read_csv(dataset_path)

    logger.info("Evaluation dataset not found. Generating deterministically from PostgreSQL database...")
    try:
        from database.connection import get_db_connection
        conn = get_db_connection()
        dfs = []
        
        # 1. 17-char Standard VINs (500 rows)
        q_17 = """
            SELECT 
                chassisnumber AS "chassisNumber",
                year, make, model, trim, 
                bodytype AS "bodyType", 
                regionalspec AS "regionalSpec", 
                cylinders, color, origin, 
                weightinkg AS "weightInKg", 
                noofpassengers AS "noOfPassengers"
            FROM master_data
            WHERE LENGTH(chassisnumber) = 17
            ORDER BY chassisnumber
            LIMIT 500;
        """
        df_17 = pd.read_sql_query(q_17, conn)
        dfs.append(df_17)
        
        # 2. Short VINs (100 rows per length: 10, 11, 12)
        for length in (10, 11, 12):
            q_short = f"""
                SELECT 
                    chassisnumber AS "chassisNumber",
                    year, make, model, trim, 
                    bodytype AS "bodyType", 
                    regionalspec AS "regionalSpec", 
                    cylinders, color, origin, 
                    weightinkg AS "weightInKg", 
                    noofpassengers AS "noOfPassengers"
                FROM master_data_{length}
                ORDER BY chassisnumber
                LIMIT 100;
            """
            df_short = pd.read_sql_query(q_short, conn)
            dfs.append(df_short)
            
        conn.close()
        
        # Merge all into one evaluation set
        full_df = pd.concat(dfs, ignore_index=True)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(dataset_path, index=False)
        logger.info(f"Deterministic fixed evaluation dataset generated and saved to: {dataset_path} ({len(full_df)} rows)")
        return full_df
        
    except Exception as e:
        logger.error(f"Failed to generate evaluation dataset from DB: {e}")
        # Build a dummy dataset if DB is unreachable to fail safely
        raise RuntimeError("No database connectivity to build evaluation dataset.")

def normalize_pred_dict(d: dict) -> dict:
    """Normalizes keys and string values to allow uniform comparisons."""
    normalized = {}
    for k, v in d.items():
        k_lower = k.lower().replace(" ", "_")
        if k_lower == "regional_specs":
            k_lower = "regional_spec"
            
        # Standardize value type
        if v is None or pd.isna(v):
            v_val = "UNKNOWN"
        elif isinstance(v, (int, float)):
            v_val = str(int(v))
        else:
            v_val = str(v).upper().strip()
            
        if v_val in ("NAN", "NONE", "", "NAT", "UNDEFINED"):
            v_val = "UNKNOWN"
            
        normalized[k_lower] = v_val
    return normalized

def load_short_pipeline(pipeline_path: Path):
    """Safely loads a short VIN pipeline using joblib."""
    if not pipeline_path.exists():
        return None
    try:
        import joblib
        return joblib.load(pipeline_path)
    except Exception as e:
        logger.error(f"Failed to load short VIN pipeline at {pipeline_path}: {e}")
        return None

def run_predictions(df: pd.DataFrame, model_dir: Path, is_candidate: bool) -> Tuple[list, float]:
    """Runs predictions over the dataframe using models loaded from the specified folder."""
    import predict
    
    # Pre-load short VIN pipelines
    short_pipelines = {}
    for length in (10, 11, 12):
        if is_candidate:
            path = model_dir / "short" / f"{length:02d}" / "vin_decoder_pipeline.pkl"
        else:
            path = project_root / f"chat_cat_short_vin_{length if length != 10 else ''}".strip("_") / "vin_decoder_pipeline.pkl"
            
        short_pipelines[length] = load_short_pipeline(path)
        
    results = []
    start_time = time.time()
    
    for idx, row in df.iterrows():
        vin = str(row["chassisNumber"]).upper().strip()
        length = len(vin)
        
        # 1. Standard 17-char prediction
        if length == 17:
            # Clear standard predict.py resource cache to load correctly
            predict._resources = None
            try:
                res = predict.decode_vin(vin, model_dir=str(model_dir if is_candidate else "models"))
                results.append(normalize_pred_dict(res))
            except Exception as e:
                logger.error(f"Standard inference failed for VIN {vin}: {e}")
                results.append({attr: "UNKNOWN" for attr in ATTRIBUTES})
                
        # 2. Short VIN prediction (10, 11, 12)
        elif length in (10, 11, 12) and short_pipelines[length] is not None:
            try:
                res = short_pipelines[length].predict(vin)
                results.append(normalize_pred_dict(res))
            except Exception as e:
                logger.error(f"Short {length}-char inference failed for chassis {vin}: {e}")
                results.append({attr: "UNKNOWN" for attr in ATTRIBUTES})
        else:
            results.append({attr: "UNKNOWN" for attr in ATTRIBUTES})
            
    total_time = time.time() - start_time
    avg_inference_ms = (total_time / len(df)) * 1000
    return results, avg_inference_ms

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Computes all classification metrics for a single attribute."""
    # Convert lists to clean strings
    y_true_clean = [str(x).upper().strip() if pd.notna(x) else "UNKNOWN" for x in y_true]
    y_pred_clean = [str(x).upper().strip() if pd.notna(x) else "UNKNOWN" for x in y_pred]
    
    accuracy = accuracy_score(y_true_clean, y_pred_clean)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true_clean, y_pred_clean)
    
    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_true_clean))
    }

def save_confusion_matrix_plots(y_true: list, y_pred: list, attr: str, out_dir: Path):
    """Optionally draws and saves confusion matrix plots using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        y_true_clean = [str(x).upper().strip() if pd.notna(x) else "UNKNOWN" for x in y_true]
        y_pred_clean = [str(x).upper().strip() if pd.notna(x) else "UNKNOWN" for x in y_pred]
        
        labels = sorted(list(set(y_true_clean).union(set(y_pred_clean))))
        # Limit label plotting counts if too large
        if len(labels) > 20:
            labels = labels[:20]
            
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix - {attr.upper()}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"confusion_matrix_{attr}.png", bbox_inches="tight")
        plt.close()
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Automated Model Evaluation & Comparison Pipeline")
    parser.add_argument("--candidate_dir", type=str, required=True, help="Versioned folder (e.g. models/training_001)")
    parser.add_argument("--production_dir", type=str, default="models", help="Current production models folder")
    parser.add_argument("--dataset_path", type=str, default="Data/master-comparison.csv", help="Fixed comparison dataset")
    parser.add_argument("--config_path", type=str, default="evaluation_config.json", help="Decision configuration file")
    
    args = parser.parse_args()
    
    candidate_dir = Path(args.candidate_dir)
    production_dir = Path(args.production_dir)
    dataset_path = Path(args.dataset_path)
    config_path = Path(args.config_path)
    
    logger.info("Initializing Evaluation and Comparison Pipeline...")
    
    # 1. Load config and fixed dataset
    config = load_config(config_path)
    # Using master-comparison.csv if it exists, otherwise generate database evaluation dataset
    df_eval = get_fixed_dataset(dataset_path)
    
    # 2. Run predictions for both Production and Candidate
    logger.info("Running inference using CURRENT PRODUCTION models...")
    prod_preds, prod_inf_time = run_predictions(df_eval, production_dir, is_candidate=false)
    logger.info(f"Production inference complete. Average latency: {prod_inf_time:.2f} ms/VIN")
    
    logger.info("Running inference using LATEST RETRAINED CANDIDATE models...")
    cand_preds, cand_inf_time = run_predictions(df_eval, candidate_dir, is_candidate=true)
    logger.info(f"Candidate inference complete. Average latency: {cand_inf_time:.2f} ms/VIN")
    
    # 3. Calculate metrics per attribute
    metrics_report = {"production": {}, "candidate": {}}
    comparison_table = []
    
    eval_dir = candidate_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    for attr in ATTRIBUTES:
        logger.info(f"Evaluating attribute: {attr.upper()}")
        
        # Ground truth mapping
        gt_col = "bodyType" if attr == "body_type" else "regionalSpec" if attr == "regional_spec" else "weightInKg" if attr == "weight" else "noOfPassengers" if attr == "no_of_passengers" else attr
        
        y_true = df_eval[gt_col].tolist()
        y_prod = [p[attr] for p in prod_preds]
        y_cand = [p[attr] for p in cand_preds]
        
        prod_metrics = compute_metrics(y_true, y_prod)
        cand_metrics = compute_metrics(y_true, y_cand)
        
        metrics_report["production"][attr] = prod_metrics
        metrics_report["candidate"][attr] = cand_metrics
        
        # Save plots
        save_confusion_matrix_plots(y_true, y_cand, attr, eval_dir / "plots")
        
        # Add to comparison table
        prod_score = prod_metrics["accuracy"] * 100
        cand_score = cand_metrics["accuracy"] * 100
        winner = "Candidate" if cand_score > prod_score else "Production" if prod_score > cand_score else "Tie"
        
        comparison_table.append({
            "attribute": attr,
            "production_acc": round(prod_score, 2),
            "candidate_acc": round(cand_score, 2),
            "winner": winner
        })
        
    # 4. Compute Overall Weighted Scores
    weights = config.get("weights", {})
    prod_weighted_score = 0.0
    cand_weighted_score = 0.0
    total_weight = 0.0
    
    for attr in ATTRIBUTES:
        weight = weights.get(attr, 0.0)
        if weight > 0:
            prod_weighted_score += metrics_report["production"][attr]["accuracy"] * weight
            cand_weighted_score += metrics_report["candidate"][attr]["accuracy"] * weight
            total_weight += weight
            
    if total_weight > 0:
        prod_final_score = (prod_weighted_score / total_weight) * 100
        cand_final_score = (cand_weighted_score / total_weight) * 100
    else:
        prod_final_score = 0.0
        cand_final_score = 0.0
        
    # 5. Decision Engine
    rules = config.get("promotion_rules", {})
    status = "APPROVED"
    reasons = []
    
    # Rule A: Require overall weighted score improvement
    if rules.get("require_overall_improvement", True):
        if cand_final_score <= prod_final_score:
            status = "REJECTED"
            reasons.append(f"Candidate overall weighted score ({cand_final_score:.2f}%) did not outperform Production ({prod_final_score:.2f}%).")
            
    # Rule B: Regression tolerance on critical attributes
    regression_tolerance = rules.get("allow_regression_tolerance", 0.01)
    critical_attributes = rules.get("critical_attributes", ["make", "model"])
    for attr in critical_attributes:
        prod_acc = metrics_report["production"][attr]["accuracy"]
        cand_acc = metrics_report["candidate"][attr]["accuracy"]
        if cand_acc < (prod_acc - regression_tolerance):
            status = "REJECTED"
            reasons.append(f"Regression detected on critical attribute '{attr}': Candidate Accuracy ({cand_acc*100:.2f}%) fell below Production ({prod_acc*100:.2f}%) by more than the allowed tolerance ({regression_tolerance*100}%).")
            
    # 6. Save Evaluation Artifacts
    overall_summary = {
        "status": status,
        "reasons": reasons,
        "weighted_scores": {
            "production": round(prod_final_score, 2),
            "candidate": round(cand_final_score, 2)
        },
        "latencies_ms": {
            "production": round(prod_inf_time, 2),
            "candidate": round(cand_inf_time, 2)
        },
        "attribute_comparison": comparison_table
    }
    
    # Save comparison report JSON
    report_path = eval_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(overall_summary, f, indent=2)
        
    # Save detailed metrics JSON
    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
        
    # Save prediction samples CSV (subset of 50 samples for verification)
    sample_rows = []
    for i in range(min(50, len(df_eval))):
        vin = df_eval.iloc[i]["chassisNumber"]
        for attr in ["make", "model", "trim", "year"]:
            gt_col = "bodyType" if attr == "body_type" else "regionalSpec" if attr == "regional_spec" else "weightInKg" if attr == "weight" else "noOfPassengers" if attr == "no_of_passengers" else attr
            gt_val = df_eval.iloc[i][gt_col]
            prod_val = prod_preds[i][attr]
            cand_val = cand_preds[i][attr]
            sample_rows.append({
                "vin": vin,
                "attribute": attr,
                "ground_truth": gt_val,
                "production_pred": prod_val,
                "candidate_pred": cand_val,
                "is_match_prod": str(gt_val).upper().strip() == str(prod_val).upper().strip(),
                "is_match_cand": str(gt_val).upper().strip() == str(cand_val).upper().strip()
            })
    pd.DataFrame(sample_rows).to_csv(eval_dir / "prediction_samples.csv", index=False)
    
    # Generate comparison_summary.md
    summary_md_path = eval_dir / "comparison_summary.md"
    alert_type = "[!NOTE]" if status == "APPROVED" else "[!WARNING]"
    reasons_str = "\n".join([f"- {r}" for r in reasons])
    
    md_content = f"""# Model Comparison Report

This report compares the performance of the newly retrained candidate models against the currently active production models.

> {alert_type}
> **Promotion Status: {status}**
> Overall recommendation generated by the automatic decision engine.
{f"> Reasons for rejection:\n> {reasons_str}" if status == "REJECTED" else ""}

## Weighted Overall Performance

- **Production Weighted Score**: {prod_final_score:.2f}%
- **Candidate Weighted Score**: {cand_final_score:.2f}%
- **Winner**: {"Candidate" if cand_final_score > prod_final_score else "Production"}

## Inference Latency

- **Production Average Latency**: {prod_inf_time:.2f} ms/VIN
- **Candidate Average Latency**: {cand_inf_time:.2f} ms/VIN

## Attribute-level Accuracies

| Attribute | Production Accuracy (%) | Candidate Accuracy (%) | Winner |
| :--- | :---: | :---: | :---: |
"""
    for row in comparison_table:
        md_content += f"| {row['attribute'].upper()} | {row['production_acc']:.2f}% | {row['candidate_acc']:.2f}% | {row['winner']} |\n"
        
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logger.info(f"Evaluation report written successfully to: {report_path}")
    logger.info(f"Markdown summary saved successfully to: {summary_md_path}")
    logger.info("Evaluation and Comparison Pipeline complete!")

if __name__ == "__main__":
    main()
