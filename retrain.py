import os
import sys
import re
import json
import time
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging to write to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
class DualStream:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        
    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.flush()
        
    def flush(self):
        try:
            self.stream1.flush()
        except Exception:
            pass
        try:
            self.stream2.flush()
        except Exception:
            pass

def get_next_training_dir(models_base_dir: Path) -> str:
    """Finds the next training folder name (e.g., training_001)."""
    os.makedirs(models_base_dir, exist_ok=True)
    existing_dirs = os.listdir(models_base_dir)
    max_num = 0
    pattern = re.compile(r"^training_(\d+)$")
    for d in existing_dirs:
        match = pattern.match(d)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    next_num = max_num + 1
    return f"training_{next_num:03d}"

def get_db_row_counts() -> dict:
    """Queries PostgreSQL database to get the row counts for training data."""
    try:
        from database.connection import get_db_connection
        conn = get_db_connection()
        counts = {}
        with conn.cursor() as cur:
            # Get 17-character count from master_data
            cur.execute("""
                SELECT COUNT(*) 
                FROM master_data 
                WHERE LENGTH(chassisnumber) = 17;
            """)
            counts["standard_17"] = cur.fetchone()[0]
                
            # Get short VIN tables counts
            for length in (10, 11, 12):
                table_name = f"master_data_{length}"
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                    counts[f"short_{length}"] = cur.fetchone()[0]
                except Exception as e:
                    logger.warning(f"Could not get count for table {table_name}: {e}")
                    counts[f"short_{length}"] = 0
        conn.close()
        return counts
    except Exception as e:
        logger.warning(f"Could not fetch database row counts for metadata: {e}")
        return {}

def run_training_step(cmd: list, cwd: Path, log_file) -> bool:
    """Runs a single training step as a subprocess, writing output to log_file."""
    import traceback
    logger.info(f"Executing: {' '.join(cmd)} in CWD: {cwd}")
    log_file.write(f"\n--- EXECUTING COMMAND: {' '.join(cmd)} ---\n")
    log_file.flush()
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        duration = time.time() - start_time
        logger.info(f"Command finished with return code {result.returncode} in {duration:.2f} seconds.")
        log_file.write(f"\n--- FINISHED IN {duration:.2f}s WITH RETURN CODE {result.returncode} ---\n")
        log_file.flush()
        return result.returncode == 0
    except Exception as e:
        duration = time.time() - start_time
        err_msg = f"EXCEPTION OCCURRED: {e}\n{traceback.format_exc()}"
        logger.error(err_msg)
        log_file.write(f"\n--- FAILED IN {duration:.2f}s WITH EXCEPTION ---\n{err_msg}\n")
        log_file.flush()
        return False

def main():
    parser = argparse.ArgumentParser(description="Unified PostgreSQL-Based Automated Retraining Pipeline")
    parser.add_argument("--epochs", type=int, default=15, help="NN Transformer training epochs")
    parser.add_argument("--downsample", type=int, default=200000, help="NN Transformer training downsample size (-1 for none)")
    parser.add_argument("--batch_size", type=int, default=512, help="NN training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="NN learning rate")
    parser.add_argument("--fast", action="store_true", help="Run short VIN classifiers with faster/reduced settings")
    parser.add_argument("--skip_nn", action="store_true", help="Skip Neural Network training phase")
    parser.add_argument("--skip_fallback", action="store_true", help="Skip fallback lookup training phase")
    parser.add_argument("--skip_rules", action="store_true", help="Skip rule extraction phase")
    parser.add_argument("--models_dir", type=str, default="models", help="Base directory for training runs")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    models_base_dir = project_root / args.models_dir
    
    # 1. Determine versioned folder
    training_name = get_next_training_dir(models_base_dir)
    versioned_dir = models_base_dir / training_name
    
    # Create subdirectories
    standard_dir = versioned_dir / "standard"
    short_dir = versioned_dir / "short"
    
    os.makedirs(standard_dir, exist_ok=True)
    for length in ("10", "11", "12"):
        os.makedirs(short_dir / length, exist_ok=True)
        
    log_file_path = versioned_dir / "training.log"
    
    # Open log file and redirect stdout/stderr to capture all print/logger outputs
    log_file = open(log_file_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = DualStream(original_stdout, log_file)
    sys.stderr = DualStream(original_stderr, log_file)
    
    try:
        log_file.write(f"=== RETRAINING RUN: {training_name} ===\n")
        log_file.write(f"Started at: {datetime.now().isoformat()}\n")
        log_file.write(f"Parameters: {vars(args)}\n\n")
        log_file.flush()
        
        logger.info(f"Retraining run started. Outputs will be saved in: {versioned_dir}")
        logger.info(f"Logs will be written to: {log_file_path}")
        
        start_total_time = time.time()
        success_states = {}
        
        # 2. Retrain Standard 17-character VIN model
        logger.info("\n--- Retraining Standard 17-Character Models ---")
        cmd_standard = [
            sys.executable,
            "train_pipeline.py",
            "--data_path", "postgres",
            "--model_dir", str(standard_dir),
            "--epochs", str(args.epochs),
            "--downsample", str(args.downsample),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr)
        ]
        if args.skip_nn:
            cmd_standard.append("--skip_nn")
        if args.skip_fallback:
            cmd_standard.append("--skip_fallback")
        if args.skip_rules:
            cmd_standard.append("--skip_rules")
            
        standard_success = run_training_step(
            cmd_standard, 
            cwd=project_root / "chat_cat(2.0)", 
            log_file=log_file
        )
        success_states["standard_17"] = standard_success
        
        # 3. Retrain Short VIN Models
        short_tasks = [
            ("10", "chat_cat_short_vin"),
            ("11", "chat_cat_short_vin_11"),
            ("12", "chat_cat_short_vin_12")
        ]
        
        for len_str, folder_name in short_tasks:
            logger.info(f"\n--- Retraining Short {len_str}-Character Models ---")
            cmd_short = [
                sys.executable,
                "train.py",
                "--data_path", "postgres",
                "--model_dir", str(short_dir / len_str)
            ]
            if args.fast:
                cmd_short.append("--fast")
                
            short_success = run_training_step(
                cmd_short,
                cwd=project_root / folder_name,
                log_file=log_file
            )
            success_states[f"short_{len_str}"] = short_success
            
        total_duration = time.time() - start_total_time
        logger.info(f"\nAll training tasks complete in {total_duration:.2f} seconds.")
        
        # 4. Load and aggregate evaluation metrics
        metrics_summary = {}
        
        # Load standard metrics
        std_metrics_path = standard_dir / "metrics.json"
        if std_metrics_path.exists():
            try:
                with open(std_metrics_path, "r") as f:
                    metrics_summary["standard"] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load standard metrics: {e}")
                
        # Load short metrics
        metrics_summary["short"] = {}
        for len_str, _ in short_tasks:
            short_metrics_path = short_dir / len_str / "metrics.json"
            if short_metrics_path.exists():
                try:
                    with open(short_metrics_path, "r") as f:
                        metrics_summary["short"][len_str] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load short_{len_str} metrics: {e}")
                    
        # 5. Run evaluation and comparison pipeline
        logger.info("\n--- Phase 5: Automated Model Evaluation and Comparison ---")
        cmd_eval = [
            sys.executable,
            "evaluate_and_compare.py",
            "--candidate_dir", str(versioned_dir),
            "--production_dir", "models"
        ]
        
        eval_success = run_training_step(
            cmd_eval,
            cwd=project_root,
            log_file=log_file
        )
        success_states["evaluation_and_comparison"] = eval_success
        
        # Load comparison report if it exists
        eval_report = {}
        report_path = versioned_dir / "evaluation" / "comparison_report.json"
        if report_path.exists():
            try:
                with open(report_path, "r") as f:
                    eval_report = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load evaluation report: {e}")
                
        # 6. Save final metadata.json
        db_row_counts = get_db_row_counts()
        
        metadata_final = {
            "training_number": int(training_name.split("_")[1]),
            "training_folder": training_name,
            "timestamp": datetime.now().isoformat(),
            "model_type": "Standard (17-char) & Short (10-12 char) VIN decoders",
            "project_version": "3.0",
            "training_parameters": vars(args),
            "database_row_counts": db_row_counts,
            "training_durations_sec": {
                "total_duration": total_duration
            },
            "success_states": success_states,
            "evaluation_metrics": metrics_summary,
            "evaluation_and_comparison_result": eval_report
        }
        
        metadata_file_path = versioned_dir / "metadata.json"
        with open(metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_final, f, indent=2)
            
        logger.info(f"Metadata written successfully to: {metadata_file_path}")
        logger.info("Pipeline execution wrapper complete!")
        
    finally:
        log_file.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

if __name__ == "__main__":
    main()
