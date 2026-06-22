import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import pickle
import json

from torch.utils.data import DataLoader
from dataset import MultiTaskVINDataset, encode_vin
from model import MultiTaskVINTransformer
from trainer import MultiTaskTrainer
from predict import decode_vin, get_resources
from feature_engineering import extract_vin_features

def run_verification():
    print("=== STARTING VIN PIPELINE VERIFICATION ===")
    
    # Create temp directory for testing models
    test_model_dir = "test_models"
    os.makedirs(test_model_dir, exist_ok=True)
    
    try:
        # 1. Test Dataset Tokenization
        print("\n[1/7] Testing Dataset Tokenization...")
        test_vin = "JTFHX02POF0099797"
        encoded = encode_vin(test_vin)
        print(f"VIN: {test_vin}")
        print(f"Encoded shape: {encoded.shape}, values: {encoded}")
        assert encoded.shape == (17,), "Encoded VIN shape must be (17,)"
        assert (encoded > 0).any(), "Tokenization should produce non-zero indices"
        print("OK: Dataset Tokenization passed.")

        # 2. Fit and Save Dummy Encoders
        print("\n[2/7] Saving Dummy Target Encoders...")
        target_keys = ["TRIM", "REGIONAL_SPEC", "BODY_TYPE"]
        num_classes_dict = {}
        for target in target_keys:
            le = LabelEncoder()
            # Fit on a small set of mock classes
            if target == "TRIM":
                classes = ["GXR", "VXR", "UNKNOWN"]
            elif target == "REGIONAL_SPEC":
                classes = ["GCC", "Non-Gcc", "UNKNOWN"]
            else: # BODY_TYPE
                classes = ["SUV", "Sedan", "Coupe", "UNKNOWN"]
                
            le.fit(classes)
            encoder_path = os.path.join(test_model_dir, f"{target.lower()}_encoder.pkl")
            with open(encoder_path, "wb") as f:
                pickle.dump(le, f)
            num_classes_dict[target.lower()] = len(le.classes_)
            print(f"Saved {target} encoder with {len(le.classes_)} classes.")
        print("OK: Dummy Target Encoders saved.")

        # 3. Test Transformer Architecture & Shapes
        print("\n[3/7] Testing Transformer Architecture Shapes...")
        model = MultiTaskVINTransformer(
            num_classes_dict=num_classes_dict,
            vocab_size=39,
            embedding_dim=64, # small dimensions for speed
            num_layers=2,
            num_heads=2,
            dim_feedforward=128
        )
        
        # Mock batch of 4 VIN sequences
        x_mock = torch.randint(1, 38, (4, 17))
        logits_dict = model(x_mock)
        
        for task, logits in logits_dict.items():
            expected_classes = num_classes_dict[task]
            print(f"Task: {task} | Logits shape: {logits.shape} (Expected: (4, {expected_classes}))")
            assert logits.shape == (4, expected_classes), f"Output shape mismatch for head {task}"
        print("OK: Transformer Architecture Shapes passed.")

        # 4. Test Trainer and Mock Training Loop
        print("\n[4/7] Testing Trainer and Mock Training loop...")
        # Create dummy fallback_model.pkl for ensembling testing later
        dummy_fallback = {}
        for tgt in ["MAKE", "MODEL", "TRIM", "BODY_TYPE", "YEAR", "CYLINDERS", "ORIGIN", "NO_OF_PASSENGERS", "WEIGHT", "REGIONAL_SPEC", "COLOR"]:
            # Level 0 dictionary maps WMI to (label, probability)
            dummy_fallback[tgt] = {
                0: {"JTF": ("VXR" if tgt == "TRIM" else "GCC" if tgt == "REGIONAL_SPEC" else "SUV" if tgt == "BODY_TYPE" else "Toyota", 0.9)},
                "_default": ("UNKNOWN", 1.0)
            }
        
        fallback_path = os.path.join(test_model_dir, "fallback_model.pkl")
        with open(fallback_path, "wb") as f:
            pickle.dump(dummy_fallback, f)
        
        # Set up mock datasets
        mock_vins = ["JTFHX02POF0099797"] * 20
        mock_y = {
            "trim": np.zeros(20, dtype=np.int64),
            "regional_spec": np.ones(20, dtype=np.int64),
            "body_type": np.zeros(20, dtype=np.int64)
        }
        dataset = MultiTaskVINDataset(mock_vins, mock_y)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterions = {task: nn.CrossEntropyLoss() for task in num_classes_dict.keys()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        trainer = MultiTaskTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            scheduler=None,
            criterions=criterions,
            device=device,
            save_dir=test_model_dir,
            patience=2,
            model_name="vin_transformer_best.pt"
        )
        
        trainer.fit(max_epochs=2)
        print("OK: Trainer Fit Loop completed successfully.")

        # 5. Check checkpoint generation
        print("\n[5/7] Verifying Checkpoint Creation...")
        checkpoint_file = os.path.join(test_model_dir, "vin_transformer_best.pt")
        assert os.path.exists(checkpoint_file), "Model checkpoint file was not saved!"
        print("OK: Checkpoint saved successfully.")

        # 6. Test Model Load from Checkpoint
        print("\n[6/7] Testing Model Loading...")
        from model_utils import load_vin_transformer
        loaded_model = load_vin_transformer(test_model_dir, device)
        assert loaded_model is not None, "Failed to load model from checkpoint"
        print("OK: Model successfully loaded from checkpoint.")

        # 7. Test Ensembled Inference and Caching
        print("\n[7/7] Testing Ensembled Inference and Caching...")
        # Clear resources cache to force reloading from test_model_dir
        global _resources
        import predict
        predict._resources = None
        
        # Test full ensembled decode
        output = decode_vin(test_vin, model_dir=test_model_dir, w_transformer=0.7, w_lookup=0.3)
        print(f"Result for VIN {test_vin}:")
        print(json.dumps(output, indent=2))
        
        # Assertions on output structure
        assert "vin" in output
        assert "trim" in output
        assert "regional_spec" in output
        assert "confidence" in output
        assert "attribute_confidences" in output
        
        # Verify thresholding (set to high threshold to verify UNKNOWN output)
        high_thresholds = {"trim": 0.99, "regional_spec": 0.99}
        output_thresh = decode_vin(test_vin, model_dir=test_model_dir, thresholds=high_thresholds)
        print("\nResult with high thresholds:")
        print(f"Trim: {output_thresh['trim']}, Regional Spec: {output_thresh['regional_spec']}")
        assert output_thresh["trim"] == "UNKNOWN", "High threshold should force 'UNKNOWN' output"
        assert output_thresh["regional_spec"] == "UNKNOWN", "High threshold should force 'UNKNOWN' output"
        
        print("OK: Ensembled Inference and Caching passed.")
        print("\nSUCCESS: ALL VIN PIPELINE TESTS PASSED!")
        
    finally:
        # Clean up testing outputs
        if os.path.exists(test_model_dir):
            shutil.rmtree(test_model_dir)

if __name__ == "__main__":
    run_verification()
